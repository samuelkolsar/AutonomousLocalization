"""
localizer.py — Particle filter localizer

Extracted from localization_particle_filter.ipynb.
Maintains state across calls — call process_observation() each time
a new sign image is processed.
"""

import pickle
import time
import copy
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm

warnings.filterwarnings('ignore')
ox.settings.log_console = False

# ── Configuration ─────────────────────────────────────────────────────────────

CACHE_DIR           = Path("cache")
GRAPH_FILE          = CACHE_DIR / "estonia_drive.pkl"

N_PARTICLES           = 1000
SCORE_SIGMA_M         = 2000   # forgiving enough for rounded km signs
EDGE_SAMPLE_SPACING_M = 50
RESAMPLE_THRESHOLD    = 0.5
ROUGHENING_SIGMA_M    = 30

rng = np.random.default_rng(42)

# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graph(cache_path: Path = GRAPH_FILE) -> nx.MultiDiGraph:
    if cache_path.exists():
        print("Loading graph from cache ...")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  ({time.time()-t0:.1f}s)")
        return G

    print("Downloading Estonia road network from OSM (one-time, ~5-15 min) ...")
    print("  Fetching graph ...", flush=True)
    t0 = time.time()
    G = ox.graph_from_place("Estonia", network_type="drive", simplify=True)
    print(f"  Graph fetched in {time.time()-t0:.1f}s — adding speeds ...", flush=True)
    G = ox.add_edge_speeds(G)
    print(f"  Speeds added — adding travel times ...", flush=True)
    G = ox.add_edge_travel_times(G)
    print(f"  Done in {time.time()-t0:.1f}s  |  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print("  Saving to cache ...")
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"  Saved to {cache_path}")
    return G


# ── City distance service ─────────────────────────────────────────────────────

class CityDistanceService:
    def __init__(self, G: nx.MultiDiGraph, cache_dir: Path = CACHE_DIR,
                 country: str = "Estonia"):
        self.G         = G
        self.G_u       = G.to_undirected()
        self.cache_dir = cache_dir
        self.country   = country
        self._distances    = {}
        self._origin_nodes = {}
        self._geocodes     = {}

    def load(self, city: str) -> bool:
        """Returns False if city could not be geocoded — caller should skip it."""
        if city in self._distances:
            return True
        safe       = city.replace(" ", "_").replace("/", "_")
        cache_file = self.cache_dir / f"city_distances_{safe}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                geocode, origin_node, distances = pickle.load(f)
            print(f"  [{city}] loaded from cache")
        else:
            print(f"  [{city}] geocoding ...", end=" ", flush=True)
            try:
                geocode = ox.geocode(f"{city}, {self.country}")
            except Exception:
                print(f"FAILED — skipping '{city}'")
                return False
            origin_node = ox.nearest_nodes(self.G, geocode[1], geocode[0])
            print(f"→ {geocode}")
            print(f"  [{city}] running Dijkstra ...", end=" ", flush=True)
            t0        = time.time()
            distances = dict(nx.single_source_dijkstra_path_length(
                self.G_u, origin_node, weight="length"
            ))
            print(f"done in {time.time()-t0:.1f}s")
            with open(cache_file, "wb") as f:
                pickle.dump((geocode, origin_node, distances), f)

        self._geocodes[city]     = geocode
        self._origin_nodes[city] = origin_node
        self._distances[city]    = distances
        return True

    def distance_at_node(self, city: str, node_id: int) -> Optional[float]:
        return self._distances[city].get(node_id)

    def origin_node(self, city: str) -> int:
        return self._origin_nodes[city]

    def geocode(self, city: str) -> tuple:
        return self._geocodes[city]


# ── Particle ──────────────────────────────────────────────────────────────────

@dataclass
class Particle:
    lat:    float
    lon:    float
    edge_u: int
    edge_v: int
    t:      float
    weight: float = 1.0


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_particle(p: Particle, observation: list[dict],
                   city_svc: CityDistanceService,
                   sigma: float = SCORE_SIGMA_M) -> float:
    total = 1.0
    for det in observation:
        city   = det["city"]
        d_sign = det["distance_km"] * 1000
        conf   = det["ocr_confidence"]
        du     = city_svc.distance_at_node(city, p.edge_u)
        dv     = city_svc.distance_at_node(city, p.edge_v)
        if du is None or dv is None:
            return 0.0
        d_actual = du * (1 - p.t) + dv * p.t
        total   *= conf * np.exp(-0.5 * ((d_actual - d_sign) / sigma) ** 2)
    return total


# ── Particle filter ───────────────────────────────────────────────────────────

class ParticleFilter:
    def __init__(self, G: nx.MultiDiGraph, edges_gdf: gpd.GeoDataFrame,
                 city_svc: CityDistanceService):
        self.G          = G
        self.edges_gdf  = edges_gdf
        self.city_svc   = city_svc
        self.particles: list[Particle] = []
        self.history:   list[dict]     = []
        self.step_index = 0
        self.initialised = False

    def initialise(self, observation: list[dict]) -> None:
        print("\n[Filter] Initialising ...")

        # Load cities, dropping any that fail geocoding
        observation = [d for d in observation if self.city_svc.load(d["city"])]
        if not observation:
            raise RuntimeError("No valid cities in observation after geocoding.")
        cities = [d["city"] for d in observation]
        print(f"  Using {len(cities)} valid cities: {cities}")

        pool_lats, pool_lons = [], []
        pool_us,   pool_vs   = [], []
        pool_ts, pool_scores = [], []

        edges_list = list(self.edges_gdf.itertuples())
        for edge in tqdm(edges_list, desc="Sampling edges"):
            u, v  = edge.Index[0], edge.Index[1]
            geom  = edge.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                edge_len_m = edge.length
            except AttributeError:
                edge_len_m = geom.length * 111_320

            endpoint_du, endpoint_dv = {}, {}
            skip = False
            for city in cities:
                du = self.city_svc.distance_at_node(city, u)
                dv = self.city_svc.distance_at_node(city, v)
                if du is None and dv is None:
                    skip = True; break
                if du is None: du = dv
                if dv is None: dv = du
                endpoint_du[city] = du
                endpoint_dv[city] = dv
            if skip:
                continue

            top = max(observation, key=lambda d: d["ocr_confidence"])
            tc  = top["city"]
            tol = SCORE_SIGMA_M * 4
            if (max(endpoint_du[tc], endpoint_dv[tc]) < top["distance_km"]*1000 - tol or
                min(endpoint_du[tc], endpoint_dv[tc]) > top["distance_km"]*1000 + tol):
                continue

            n_samples = max(2, int(edge_len_m / EDGE_SAMPLE_SPACING_M))
            for t in np.linspace(0.0, 1.0, n_samples):
                pt = geom.interpolate(t, normalized=True)
                s  = 1.0
                for det in observation:
                    city   = det["city"]
                    d_sign = det["distance_km"] * 1000
                    conf   = det["ocr_confidence"]
                    d_act  = endpoint_du[city]*(1-t) + endpoint_dv[city]*t
                    s     *= conf * np.exp(-0.5*((d_act - d_sign)/SCORE_SIGMA_M)**2)
                if s > 1e-12:
                    pool_lats.append(pt.y);  pool_lons.append(pt.x)
                    pool_us.append(u);       pool_vs.append(v)
                    pool_ts.append(t);       pool_scores.append(s)

        if not pool_scores:
            raise RuntimeError("No candidate points found. Try increasing SCORE_SIGMA_M.")

        print(f"  Candidate pool: {len(pool_scores):,} points")
        scores_arr = np.array(pool_scores)
        probs      = scores_arr / scores_arr.sum()
        indices    = rng.choice(len(pool_scores), size=N_PARTICLES, replace=True, p=probs)

        self.particles = [
            Particle(lat=pool_lats[i], lon=pool_lons[i],
                     edge_u=pool_us[i], edge_v=pool_vs[i],
                     t=pool_ts[i], weight=1.0/N_PARTICLES)
            for i in indices
        ]
        self.initialised = True
        self._save_snapshot("init", observation)
        self.step_index = 1
        lat, lon, unc = self.estimate()
        print(f"  Initial estimate: ({lat:.5f}, {lon:.5f})  uncertainty ≈ {unc:.0f} m")

    def predict(self, delta_distance_m: float = 0.0,
                delta_heading_deg: float = 0.0) -> None:
        """Odometry stub — implement when odometry data is available."""
        if delta_distance_m == 0.0:
            return
        raise NotImplementedError("Odometry not yet implemented.")

    def update(self, observation: list[dict]) -> None:
        self.step_index += 1
        print(f"\n[Filter] Update #{self.step_index} ...")

        # Load cities, dropping any that fail geocoding
        observation = [d for d in observation if self.city_svc.load(d["city"])]
        if not observation:
            print("  WARNING: no valid cities in observation — skipping update")
            return

        new_weights = np.array([
            p.weight * score_particle(p, observation, self.city_svc)
            for p in tqdm(self.particles, desc="Reweighting")
        ])
        total_w = new_weights.sum()
        if total_w == 0:
            print("  WARNING: all weights zero — keeping previous distribution")
            return

        new_weights /= total_w
        for p, w in zip(self.particles, new_weights):
            p.weight = w

        ess       = 1.0 / np.sum(new_weights ** 2)
        ess_ratio = ess / N_PARTICLES
        print(f"  ESS = {ess:.0f} / {N_PARTICLES}  ({100*ess_ratio:.1f}%)")

        if ess_ratio < RESAMPLE_THRESHOLD:
            self._systematic_resample(new_weights)

        self._save_snapshot(f"update_{self.step_index}", observation)
        lat, lon, unc = self.estimate()
        print(f"  Updated estimate: ({lat:.5f}, {lon:.5f})  uncertainty ≈ {unc:.0f} m")

    def _systematic_resample(self, weights: np.ndarray) -> None:
        N      = N_PARTICLES
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        positions  = (rng.random() + np.arange(N)) / N
        indices    = np.searchsorted(cumsum, positions)
        jitter_deg = ROUGHENING_SIGMA_M / 111_320

        self.particles = [
            Particle(
                lat    = self.particles[i].lat + rng.normal(0, jitter_deg),
                lon    = self.particles[i].lon + rng.normal(0, jitter_deg / np.cos(np.radians(self.particles[i].lat))),
                edge_u = self.particles[i].edge_u,
                edge_v = self.particles[i].edge_v,
                t      = self.particles[i].t,
                weight = 1.0 / N,
            )
            for i in indices
        ]
        print(f"  Resampled → {N} particles")

    def estimate(self) -> tuple[float, float, float]:
        weights = np.array([p.weight for p in self.particles])
        lats    = np.array([p.lat    for p in self.particles])
        lons    = np.array([p.lon    for p in self.particles])
        w_lat   = np.average(lats, weights=weights)
        w_lon   = np.average(lons, weights=weights)
        var_lat = np.average((lats - w_lat)**2, weights=weights)
        var_lon = np.average((lons - w_lon)**2, weights=weights)
        unc_m   = np.sqrt(var_lat + var_lon) * 111_320
        return w_lat, w_lon, unc_m

    def _save_snapshot(self, label: str, observation: list[dict]) -> None:
        lat, lon, unc = self.estimate()
        self.history.append({
            "label":       label,
            "observation": observation,
            "particles":   copy.deepcopy(self.particles),
            "estimate":    {"lat": lat, "lon": lon, "uncertainty_m": unc},
        })


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_particle_map(pf: ParticleFilter, G: nx.MultiDiGraph,
                      city_svc: CityDistanceService,
                      output_dir: Path = Path("results"),
                      margin: float = 0.15,
                      figsize: tuple = (14, 10)) -> None:
    final   = pf.history[-1]
    est     = final["estimate"]
    parts   = final["particles"]
    weights = np.array([p.weight for p in parts])
    max_w   = weights.max()
    lons    = np.array([p.lon for p in parts])
    lats    = np.array([p.lat for p in parts])

    lon_min = lons.min() - margin;  lon_max = lons.max() + margin
    lat_min = lats.min() - margin;  lat_max = lats.max() + margin
    for snap in pf.history:
        e = snap["estimate"]
        lon_min = min(lon_min, e["lon"] - margin)
        lon_max = max(lon_max, e["lon"] + margin)
        lat_min = min(lat_min, e["lat"] - margin)
        lat_max = max(lat_max, e["lat"] + margin)

    tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    title = (f"Step {pf.step_index}  |  {len(parts)} particles  |  "
             f"({est['lat']:.4f}, {est['lon']:.4f})  |  "
             f"uncertainty ≈ {est['uncertainty_m']:.0f} m")

    def _overlay(ax):
        ax.scatter(lons, lats, c=weights/max_w, cmap='plasma',
                   s=12, alpha=0.7, zorder=3, linewidths=0)
        for snap in pf.history:
            e = snap["estimate"]
            ax.scatter(e["lon"], e["lat"], s=100, color='cyan',
                       marker='+', linewidths=2.5, zorder=5)
        if len(pf.history) > 1:
            ax.plot([s["estimate"]["lon"] for s in pf.history],
                    [s["estimate"]["lat"] for s in pf.history],
                    color='cyan', linewidth=1.5, alpha=0.7, zorder=4)
        for j, det in enumerate(final["observation"]):
            nd = G.nodes[city_svc.origin_node(det["city"])]
            ax.scatter(nd['x'], nd['y'], s=80, color=tab10[j % 4],
                       marker='D', zorder=5, edgecolors='white', linewidths=0.5)
            ax.annotate(det["city"], (nd['x'], nd['y']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=tab10[j % 4], fontweight='bold')

    # Full map
    fig1, ax1 = ox.plot_graph(G, show=False, close=False, bgcolor='#0e1117',
                               node_size=0, edge_color='#333844',
                               edge_linewidth=0.5, figsize=figsize)
    _overlay(ax1)
    ax1.set_title("Full map — " + title, color='white', fontsize=9)
    p1 = output_dir / f"map_full_step{pf.step_index}.png"
    plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()

    # Zoomed map
    bbox = (lat_max, lat_min, lon_max, lon_min)
    fig2, ax2 = ox.plot_graph(G, show=False, close=False, bbox=bbox,
                               bgcolor='#0e1117', node_size=0,
                               edge_color='#333844', edge_linewidth=0.8,
                               figsize=figsize)
    _overlay(ax2)
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_title("Zoomed — " + title, color='white', fontsize=9)
    p2 = output_dir / f"map_zoomed_step{pf.step_index}.png"
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()

    print(f"  Maps saved → {p1.name}, {p2.name}")

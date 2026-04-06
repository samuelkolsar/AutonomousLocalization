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

HIGHWAY_TYPES = {
    "motorway", "motorway_link",
    "trunk",    "trunk_link",
    "primary",  "primary_link",
}

N_PARTICLES           = 1000
SCORE_SIGMA_M         = 2000   # base sigma (m); tune on validation (e.g. 1000–3000)
EDGE_SAMPLE_SPACING_M = 50
RESAMPLE_THRESHOLD    = 0.5
ROUGHENING_SIGMA_M    = 30
# Observation model: low-confidence readings get wider sigma (less influence)
CONFIDENCE_FLOOR      = 0.2    # avoid sigma_eff exploding for very low conf
MIN_LIKELIHOOD        = 1e-3   # cap per-observation factor so one bad read doesn't kill weights
# Motion bias at junctions: prefer straight, discourage U-turns and tiny side roads.
TURN_SIGMA_DEG        = 35.0   # sigma when heading is inferred from edge geometry
HEADING_SIGMA_DEG     = 15.0   # tighter sigma when a measured heading is supplied
UTURN_PENALTY         = 0.02
REF_EDGE_LENGTH_M     = 80.0
MIN_EDGE_LENGTH_BIAS  = 0.25

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


def _highway_weight(edge_data: dict) -> float:
    """
    Motion-model preference weight for an edge based on its road type.
    Particles strongly prefer motorway/trunk, moderately prefer primary,
    and almost never take minor roads — but the full graph stays connected.
    """
    hw = edge_data.get("highway") or ""
    tags = [hw] if isinstance(hw, str) else list(hw)
    if any(t in {"motorway", "motorway_link", "trunk", "trunk_link"} for t in tags):
        return 1.0
    if any(t in {"primary", "primary_link"} for t in tags):
        return 0.7
    if any(t in {"secondary", "secondary_link"} for t in tags):
        return 0.05
    return 0.01


def _is_highway_edge(hw_value) -> bool:
    """Return True if an edge's highway attribute is a major road type."""
    if hw_value is None:
        return False
    try:
        import math
        if math.isnan(float(hw_value)):
            return False
    except (TypeError, ValueError):
        pass
    tags = [hw_value] if isinstance(hw_value, str) else list(hw_value)
    return any(t in HIGHWAY_TYPES for t in tags)


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

def _observation_factor(d_actual: float, d_sign: float, conf: float,
                        sigma: float = SCORE_SIGMA_M) -> float:
    """
    Per-observation likelihood factor. Low OCR confidence widens sigma so
    that reading has less influence; factor is capped to avoid one bad read
    killing weights.
    """
    sigma_eff = sigma / np.sqrt(max(conf, CONFIDENCE_FLOOR))
    raw = np.exp(-0.5 * ((d_actual - d_sign) / sigma_eff) ** 2)
    return max(raw, MIN_LIKELIHOOD)


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
        total   *= _observation_factor(d_actual, d_sign, conf, sigma)
    return total


# ── Particle filter ───────────────────────────────────────────────────────────

class ParticleFilter:
    def __init__(self, G: nx.MultiDiGraph, edges_gdf: gpd.GeoDataFrame,
                 city_svc: CityDistanceService):
        self.G          = G
        self.city_svc   = city_svc
        self.particles: list[Particle] = []
        self.history:   list[dict]     = []
        self.step_index = 0
        self.initialised = False

        # Pre-filter highway edges once at construction time — cheap on the
        # full graph and avoids iterating 190k rows on every initialise() call.
        print("Pre-filtering highway edges for initialisation ...")
        t0 = time.time()
        self._highway_edges = [
            e for e in edges_gdf.itertuples()
            if _is_highway_edge(getattr(e, 'highway', None))
        ]
        print(f"  {len(self._highway_edges):,} highway edges  ({time.time()-t0:.1f}s)")

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

        for edge in tqdm(self._highway_edges, desc="Sampling highway edges"):
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
                    s     *= _observation_factor(d_act, d_sign, conf, SCORE_SIGMA_M)
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
        map_lat, map_lon, map_unc = self.estimate_map()
        print(f"  Initial estimate: ({lat:.5f}, {lon:.5f})  uncertainty ≈ {unc:.0f} m")
        print(f"  Initial MAP:      ({map_lat:.5f}, {map_lon:.5f})  spread ≈ {map_unc:.0f} m")
        if unc > 30_000:
            print(f"  WARNING: initial uncertainty too high ({unc:.0f} m) — "
                  f"observation likely contains a misread (e.g. route code as distance). "
                  f"Rejecting this initialisation; waiting for a better sign.")
            self.particles  = []
            self.history    = []
            self.step_index = 0
            self.initialised = False

    def predict(self, delta_distance_m: float = 0.0,
                heading_deg: Optional[float] = None) -> None:
        """
        Propagate each particle forward along the road network by
        delta_distance_m metres (plus Gaussian noise for odometry error).

        At each junction the particle picks an outgoing edge using a simple
        transition bias: prefer continuing direction, penalise immediate
        U-turns, and penalise very short side-road edges.

        heading_deg — optional compass bearing (0 = North, 90 = East, clockwise).
        When supplied, junction selection uses the measured heading with a
        tighter sigma instead of inferring direction from edge geometry.
        """
        if delta_distance_m <= 0.0:
            return

        # Convert compass bearing to math angle (East=0, CCW) used by atan2
        measured_heading_rad: Optional[float] = None
        if heading_deg is not None:
            measured_heading_rad = np.radians(90.0 - heading_deg)

        heading_str = f"  heading {heading_deg:.0f}°" if heading_deg is not None else ""
        print(f"\n[Filter] Predict — advancing particles {delta_distance_m:.0f} m{heading_str} ...")

        # Odometry noise: 5% of distance travelled, minimum 10 m
        noise_std_m = max(10.0, delta_distance_m * 0.05)

        # Pre-build adjacency for fast junction lookup (include data for road-type weighting)
        adj = {}
        for u, v, key, data in self.G.edges(keys=True, data=True):
            length = data.get("length", 1.0)
            if u not in adj:
                adj[u] = []
            adj[u].append((v, length, data.get("geometry"), data))

        new_particles = []
        for p in self.particles:
            travel = max(0.0, delta_distance_m + rng.normal(0, noise_std_m))

            cur_u, cur_v, cur_t = p.edge_u, p.edge_v, p.t

            # Get current edge length
            edge_data = self.G.get_edge_data(cur_u, cur_v)
            if edge_data is None:
                new_particles.append(p)
                continue
            if isinstance(edge_data, dict):
                edge_data = edge_data.get(0) or next(iter(edge_data.values()))
            cur_len   = edge_data.get("length", 1.0)
            remaining = cur_len * (1.0 - cur_t)

            while travel > 0:
                if travel < remaining:
                    cur_t  = cur_t + (travel / cur_len)
                    cur_t  = min(cur_t, 1.0)
                    travel = 0.0
                else:
                    travel   -= remaining
                    prev_u    = cur_u
                    cur_u     = cur_v
                    cur_t     = 0.0
                    neighbors = adj.get(cur_u, [])
                    if not neighbors:
                        break
                    nv, nlen = self._choose_next_edge(prev_u, cur_u, neighbors,
                                                        measured_heading_rad)
                    cur_v     = nv
                    cur_len   = nlen
                    remaining = cur_len

            # Resolve coordinates at new position
            ed = self.G.get_edge_data(cur_u, cur_v)
            if ed is None:
                new_particles.append(p)
                continue
            if isinstance(ed, dict):
                ed = ed.get(0) or next(iter(ed.values()))
            geom = ed.get("geometry")
            if geom is not None:
                pt  = geom.interpolate(cur_t, normalized=True)
                lat, lon = pt.y, pt.x
            else:
                nd_u = self.G.nodes[cur_u]
                nd_v = self.G.nodes[cur_v]
                lat  = nd_u["y"] * (1 - cur_t) + nd_v["y"] * cur_t
                lon  = nd_u["x"] * (1 - cur_t) + nd_v["x"] * cur_t

            new_particles.append(Particle(
                lat=lat, lon=lon,
                edge_u=cur_u, edge_v=cur_v,
                t=cur_t, weight=p.weight
            ))

        self.particles = new_particles
        print(f"  Particles propagated.")

    @staticmethod
    def _angle_diff_rad(a: float, b: float) -> float:
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return abs(d)

    def _choose_next_edge(self, prev_u: int, cur_u: int,
                          neighbors: list[tuple[int, float, Optional[object], dict]],
                          measured_heading_rad: Optional[float] = None) -> tuple[int, float]:
        """Sample next edge using heading continuity + U-turn + short-edge + road-type biases.

        measured_heading_rad — if supplied (already in math-angle radians), use it
        directly with HEADING_SIGMA_DEG instead of inferring from edge geometry.
        """
        node_cur = self.G.nodes.get(cur_u)
        if node_cur is None:
            nv, nlen, _, _ = neighbors[rng.integers(len(neighbors))]
            return nv, nlen

        if measured_heading_rad is not None:
            in_heading = measured_heading_rad
            sigma_rad  = np.radians(HEADING_SIGMA_DEG)
        else:
            node_prev = self.G.nodes.get(prev_u)
            if node_prev is None:
                nv, nlen, _, _ = neighbors[rng.integers(len(neighbors))]
                return nv, nlen
            in_heading = np.arctan2(node_cur["y"] - node_prev["y"],
                                    node_cur["x"] - node_prev["x"])
            sigma_rad  = np.radians(TURN_SIGMA_DEG)

        scores = []
        for nv, nlen, _, edge_data in neighbors:
            node_next = self.G.nodes.get(nv)
            if node_next is None:
                scores.append(0.0)
                continue

            out_heading = np.arctan2(node_next["y"] - node_cur["y"], node_next["x"] - node_cur["x"])
            turn    = self._angle_diff_rad(out_heading, in_heading)
            w_dir   = np.exp(-0.5 * (turn / max(sigma_rad, 1e-6)) ** 2)
            w_uturn = UTURN_PENALTY if nv == prev_u else 1.0
            w_len   = np.clip(nlen / max(REF_EDGE_LENGTH_M, 1.0), MIN_EDGE_LENGTH_BIAS, 1.0)
            w_road  = _highway_weight(edge_data)
            scores.append(float(w_dir * w_uturn * w_len * w_road))

        total = float(np.sum(scores))
        if total <= 1e-12:
            nv, nlen, _, _ = neighbors[rng.integers(len(neighbors))]
            return nv, nlen

        probs = np.array(scores, dtype=float) / total
        idx = int(rng.choice(len(neighbors), p=probs))
        nv, nlen, _, _ = neighbors[idx]
        return nv, nlen

    def update(self, observation: list[dict]) -> None:
        target_step = self.step_index + 1
        print(f"\n[Filter] Update #{target_step} ...")

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

        # Sanity check: reject observation if weighted mean would jump > 100 km.
        # This guards against bad OCR readings that would teleport all particles.
        cur_lat, cur_lon, _ = self.estimate()
        new_lat = float(np.average([p.lat for p in self.particles], weights=new_weights))
        new_lon = float(np.average([p.lon for p in self.particles], weights=new_weights))
        jump_m = 111_320 * np.sqrt(
            (new_lat - cur_lat) ** 2 +
            ((new_lon - cur_lon) * np.cos(np.radians(cur_lat))) ** 2
        )
        if jump_m > 100_000:
            print(f"  WARNING: observation would cause {jump_m/1000:.0f} km jump — rejected")
            return

        for p, w in zip(self.particles, new_weights):
            p.weight = w

        ess       = 1.0 / np.sum(new_weights ** 2)
        ess_ratio = ess / N_PARTICLES
        print(f"  ESS = {ess:.0f} / {N_PARTICLES}  ({100*ess_ratio:.1f}%)")

        if ess_ratio < RESAMPLE_THRESHOLD:
            self._systematic_resample(new_weights)

        self.step_index = target_step
        self._save_snapshot(f"update_{self.step_index}", observation)
        lat, lon, unc = self.estimate()
        map_lat, map_lon, map_unc = self.estimate_map()
        print(f"  Updated estimate: ({lat:.5f}, {lon:.5f})  uncertainty ≈ {unc:.0f} m")
        print(f"  Updated MAP:      ({map_lat:.5f}, {map_lon:.5f})  spread ≈ {map_unc:.0f} m")

    def _get_edge_length_and_geom(self, u: int, v: int) -> tuple[float, Optional[object]]:
        """Return (length_m, geometry) for edge (u, v). Length in metres."""
        ed = self.G.get_edge_data(u, v)
        if ed is None:
            return 1.0, None
        if isinstance(ed, dict):
            ed = ed.get(0) or next(iter(ed.values()))
        length = ed.get("length", 1.0)
        geom = ed.get("geometry")
        return length, geom

    def _systematic_resample(self, weights: np.ndarray) -> None:
        N      = N_PARTICLES
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        positions  = (rng.random() + np.arange(N)) / N
        indices    = np.searchsorted(cumsum, positions)

        new_particles = []
        for i in indices:
            p = self.particles[i]
            edge_len, geom = self._get_edge_length_and_geom(p.edge_u, p.edge_v)
            # Roughen along the edge (constraint-preserving): jitter t, then recompute lat/lon
            sigma_t = ROUGHENING_SIGMA_M / max(edge_len, 1.0)
            sigma_t = min(sigma_t, 0.15)  # cap so we don't jump too far along short edges
            t_new = np.clip(p.t + rng.normal(0, sigma_t), 0.0, 1.0)
            if geom is not None:
                pt = geom.interpolate(t_new, normalized=True)
                lat, lon = pt.y, pt.x
            else:
                nd_u = self.G.nodes[p.edge_u]
                nd_v = self.G.nodes[p.edge_v]
                lat = nd_u["y"] * (1 - t_new) + nd_v["y"] * t_new
                lon = nd_u["x"] * (1 - t_new) + nd_v["x"] * t_new
            new_particles.append(Particle(
                lat=lat, lon=lon,
                edge_u=p.edge_u, edge_v=p.edge_v,
                t=t_new, weight=1.0 / N,
            ))
        self.particles = new_particles
        print(f"  Resampled → {N} particles")

    def estimate(self) -> tuple[float, float, float]:
        weights = np.array([p.weight for p in self.particles])
        lats    = np.array([p.lat    for p in self.particles])
        lons    = np.array([p.lon    for p in self.particles])
        w_lat   = np.average(lats, weights=weights)
        w_lon   = np.average(lons, weights=weights)
        var_lat = np.average((lats - w_lat)**2, weights=weights)
        var_lon = np.average((lons - w_lon)**2, weights=weights)
        # Convert degree variances to metres: 1° lat ≈ 111320 m; 1° lon ≈ 111320*cos(lat) m
        cos_lat = np.cos(np.radians(w_lat))
        unc_m   = 111_320 * np.sqrt(var_lat + var_lon * (cos_lat ** 2))
        return w_lat, w_lon, unc_m

    def estimate_map(self) -> tuple[float, float, float]:
        """
        MAP-style estimate using the highest-weight particle.
        Returns (lat, lon, local spread in metres around MAP).
        """
        if not self.particles:
            raise RuntimeError("No particles available for MAP estimate.")

        weights = np.array([p.weight for p in self.particles])
        lats    = np.array([p.lat for p in self.particles])
        lons    = np.array([p.lon for p in self.particles])

        idx = int(np.argmax(weights))
        p0  = self.particles[idx]

        # Local ENU-style approximation around MAP point for spread.
        dy = 111_320.0 * (lats - p0.lat)
        dx = 111_320.0 * np.cos(np.radians(p0.lat)) * (lons - p0.lon)
        d2 = dx * dx + dy * dy
        spread_m = float(np.sqrt(np.average(d2, weights=weights)))
        return p0.lat, p0.lon, spread_m

    def _save_snapshot(self, label: str, observation: list[dict]) -> None:
        lat, lon, unc = self.estimate()
        map_lat, map_lon, map_unc = self.estimate_map()
        self.history.append({
            "label":       label,
            "observation": observation,
            "particles":   copy.deepcopy(self.particles),
            "estimate":    {"lat": lat, "lon": lon, "uncertainty_m": unc},
            "estimate_map": {"lat": map_lat, "lon": map_lon, "uncertainty_m": map_unc},
        })


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_particle_map(pf: ParticleFilter, G: nx.MultiDiGraph,
                      city_svc: CityDistanceService,
                      output_dir: Path = Path("results"),
                      margin: float = 0.15,
                      figsize: tuple = (14, 10)) -> None:
    final   = pf.history[-1]
    est     = final["estimate_map"]
    parts   = final["particles"]
    weights = np.array([p.weight for p in parts])
    max_w   = weights.max()
    lons    = np.array([p.lon for p in parts])
    lats    = np.array([p.lat for p in parts])

    lon_min = lons.min() - margin;  lon_max = lons.max() + margin
    lat_min = lats.min() - margin;  lat_max = lats.max() + margin
    for snap in pf.history:
        e = snap["estimate_map"]
        lon_min = min(lon_min, e["lon"] - margin)
        lon_max = max(lon_max, e["lon"] + margin)
        lat_min = min(lat_min, e["lat"] - margin)
        lat_max = max(lat_max, e["lat"] + margin)

    tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    title = (f"Step {pf.step_index}  |  {len(parts)} particles  |  "
             f"({est['lat']:.4f}, {est['lon']:.4f})  |  "
             f"uncertainty ≈ {est['uncertainty_m']:.0f} m")

    # Reference cities shown on every map for orientation
    REFERENCE_CITIES = {
        "Tallinn": (59.4370, 24.7536),
        "Tartu":   (58.3780, 26.7290),
        "Pärnu":   (58.3859, 24.4971),
        "Narva":   (59.3772, 28.1910),
        "Viljandi":(58.3639, 25.5897),
        "Rakvere": (59.3469, 26.3551),
        "Paide":   (58.8858, 25.5572),
    }

    def _overlay(ax):
        ax.scatter(lons, lats, c=weights/max_w, cmap='plasma',
                   s=12, alpha=0.7, zorder=3, linewidths=0)
        if len(pf.history) > 1:
            ax.plot([s["estimate_map"]["lon"] for s in pf.history],
                    [s["estimate_map"]["lat"] for s in pf.history],
                    color='cyan', linewidth=1.5, alpha=0.7, zorder=4)
        # Current position cross (latest only)
        ax.scatter(est["lon"], est["lat"], s=120, color='cyan',
                   marker='+', linewidths=2.5, zorder=6)
        for j, det in enumerate(final["observation"]):
            geocode = city_svc.geocode(det["city"])  # (lat, lon)
            ax.scatter(geocode[1], geocode[0], s=80, color=tab10[j % 4],
                       marker='D', zorder=5, edgecolors='white', linewidths=0.5)
            ax.annotate(det["city"], (geocode[1], geocode[0]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=tab10[j % 4], fontweight='bold')
        # Fixed reference city markers for orientation
        for city, (rlat, rlon) in REFERENCE_CITIES.items():
            ax.scatter(rlon, rlat, s=40, color='white', marker='o',
                       alpha=0.5, zorder=4, linewidths=0)
            ax.annotate(city, (rlon, rlat), xytext=(3, 3),
                        textcoords='offset points', fontsize=7,
                        color='white', alpha=0.6)

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


def plot_latest_map(pf: ParticleFilter, G: nx.MultiDiGraph,
                    city_svc: CityDistanceService,
                    output_dir: Path = Path("results"),
                    zoom: float = 0.08,
                    figsize: tuple = (10, 7)) -> None:
    """
    Lightweight zoomed map updated on every frame (including predict-only steps).
    Always saved as latest_map.png — overwrites each time.
    """
    if not pf.history:
        return

    est    = pf.history[-1]["estimate_map"]
    parts  = pf.history[-1]["particles"]
    weights = np.array([p.weight for p in parts])
    max_w  = weights.max() if weights.max() > 0 else 1.0
    lons   = np.array([p.lon for p in parts])
    lats   = np.array([p.lat for p in parts])

    lat_min = est["lat"] - zoom;  lat_max = est["lat"] + zoom
    lon_min = est["lon"] - zoom;  lon_max = est["lon"] + zoom

    title = (f"Step {pf.step_index}  |  "
             f"({est['lat']:.4f}, {est['lon']:.4f})  |  "
             f"spread ≈ {est['uncertainty_m']:.0f} m")

    bbox = (lat_max, lat_min, lon_max, lon_min)
    fig, ax = ox.plot_graph(G, show=False, close=False, bbox=bbox,
                            bgcolor='#0e1117', node_size=0,
                            edge_color='#333844', edge_linewidth=0.8,
                            figsize=figsize)

    # Particles
    mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
    if mask.any():
        ax.scatter(lons[mask], lats[mask], c=(weights/max_w)[mask],
                   cmap='plasma', s=10, alpha=0.7, zorder=3, linewidths=0)

    # Trail
    if len(pf.history) > 1:
        ax.plot([s["estimate_map"]["lon"] for s in pf.history],
                [s["estimate_map"]["lat"] for s in pf.history],
                color='cyan', linewidth=1.5, alpha=0.7, zorder=4)

    # Current position cross
    ax.scatter(est["lon"], est["lat"], s=120, color='cyan',
               marker='+', linewidths=2.5, zorder=6)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title("Latest — " + title, color='white', fontsize=9)

    out = output_dir / "latest_map.png"
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
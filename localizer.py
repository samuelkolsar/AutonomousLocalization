"""
localizer.py: Particle filter for GPS-free vehicle localization.

Uses road sign distance readings (e.g. "TARTU 82 km") combined with
vehicle odometry and compass heading to estimate position on the
Estonian road network via a particle filter.

Each particle lives on a directed graph edge and is propagated along
the network. Observations (city + distance pairs from OCR) update
particle weights using Dijkstra-based road distances.
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
from tqdm import tqdm

warnings.filterwarnings('ignore')
ox.settings.log_console = False

# Constants
CACHE_DIR  = Path("cache")
GRAPH_FILE = CACHE_DIR / "estonia_drive.pkl"

HIGHWAY_TYPES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
}

# Particle filter parameters
N_PARTICLES        = 500
SCORE_SIGMA_M      = 2000     # Gaussian sigma for distance-based scoring (metres)
SAMPLE_SPACING_M   = 50       # spacing between sample points on edges during init
RESAMPLE_THRESHOLD = 0.5      # ESS ratio below which we resample
ROUGHENING_SIGMA_M = 15       # jitter added during resampling (metres)
CONFIDENCE_FLOOR   = 0.2      # minimum OCR confidence used in scoring
MIN_LIKELIHOOD     = 1e-3     # floor for per-observation likelihood factor

# Heading parameters
HEADING_SIGMA_DEG  = 10.0     # tight sigma when compass heading is available
TURN_SIGMA_DEG     = 35.0     # looser sigma when heading is inferred from edges
HEADING_CORRECTION_DEG = 115  # threshold for snapping wrong-direction particles

rng = np.random.default_rng(42)


# Graph loading
def load_graph(cache_path: Path = GRAPH_FILE) -> nx.MultiDiGraph:
    """Load the Estonia road graph from cache, or download from OSM if missing."""
    if cache_path.exists():
        print("Loading graph from cache ...")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
              f"({time.time()-t0:.1f}s)")
        return G

    print("Downloading Estonia road network from OSM ...")
    t0 = time.time()
    G = ox.graph_from_place("Estonia", network_type="drive", simplify=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"  Saved to {cache_path}")
    return G


def _is_highway_edge(hw_value) -> bool:
    """Check if an edge's highway tag indicates a major road."""
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


def _heading_factor(u: int, v: int, G: nx.MultiDiGraph,
                    measured_heading_rad: float) -> float:
    """
    Score how well a road edge (u -> v) aligns with the measured compass heading.
    Tests both forward and reverse directions since OSM edge direction is arbitrary.
    Returns a value in [0, 1].
    """
    nd_u = G.nodes.get(u)
    nd_v = G.nodes.get(v)
    if nd_u is None or nd_v is None:
        return 1.0

    cos_lat = np.cos(np.radians((nd_u["y"] + nd_v["y"]) / 2))
    dy = nd_v["y"] - nd_u["y"]
    dx = (nd_v["x"] - nd_u["x"]) * cos_lat

    fwd = np.arctan2(dy, dx)
    rev = np.arctan2(-dy, -dx)

    sigma = np.radians(TURN_SIGMA_DEG)
    diff_fwd = abs((measured_heading_rad - fwd + np.pi) % (2 * np.pi) - np.pi)
    diff_rev = abs((measured_heading_rad - rev + np.pi) % (2 * np.pi) - np.pi)
    return float(np.exp(-0.5 * (min(diff_fwd, diff_rev) / sigma) ** 2))


# City distance service
class CityDistanceService:
    """
    Precomputes and caches shortest-path distances from city centres to
    every node in the graph using Dijkstra's algorithm. This lets us
    quickly look up "how far is node X from Tartu?" during scoring.
    """

    def __init__(self, G: nx.MultiDiGraph, cache_dir: Path = CACHE_DIR,
                 country: str = "Estonia"):
        self.G         = G
        self.G_u       = G.to_undirected()
        self.cache_dir = cache_dir
        self.country   = country
        self._distances: dict[str, dict]    = {}
        self._origin_nodes: dict[str, int]  = {}
        self._geocodes: dict[str, tuple]    = {}

    def load(self, city: str) -> bool:
        """Load (or compute) distances for a city. Returns False if geocoding fails."""
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
                print(f"FAILED")
                return False
            origin_node = ox.nearest_nodes(self.G, geocode[1], geocode[0])
            print(f"-> {geocode}")
            print(f"  [{city}] running Dijkstra ...", end=" ", flush=True)
            t0 = time.time()
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
        """Road distance (metres) from city centre to a graph node."""
        return self._distances[city].get(node_id)

    def origin_node(self, city: str) -> int:
        return self._origin_nodes[city]

    def geocode(self, city: str) -> tuple:
        return self._geocodes[city]


# Particle
@dataclass
class Particle:
    """A single particle representing a hypothesis about the vehicle's position."""
    lat:    float        # latitude
    lon:    float        # longitude
    edge_u: int          # start node of the current road edge
    edge_v: int          # end node of the current road edge
    t:      float        # position along the edge [0.0 = at u, 1.0 = at v]
    weight: float = 1.0


# Observation validation
def cross_check_observation(observation: list[dict],
                            city_svc: CityDistanceService,
                            tolerance_km: float = 30.0) -> list[dict]:
    """
    Validate multi-city observations by checking pairwise consistency.

    If a sign shows "TARTU 82" and "VORU 150", the difference (68 km) should
    roughly match the actual road distance between Tartu and Voru. If not,
    the lower-confidence reading is dropped.
    """
    if len(observation) < 2:
        return observation

    valid = [d for d in observation if city_svc.load(d["city"])]
    if len(valid) < 2:
        return valid

    keep = set(range(len(valid)))
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            d_a, d_b = a["distance_km"], b["distance_km"]

            node_a = city_svc.origin_node(a["city"])
            dist_ab = city_svc.distance_at_node(b["city"], node_a)
            if dist_ab is None:
                continue
            dist_ab_km = dist_ab / 1000.0

            diff_km = abs(d_a - d_b)
            if diff_km > dist_ab_km + tolerance_km:
                drop = i if a["ocr_confidence"] <= b["ocr_confidence"] else j
                keep.discard(drop)
                print(f"  [cross-check] {a['city']} {d_a}km vs {b['city']} {d_b}km: "
                      f"diff={diff_km:.0f}km but road={dist_ab_km:.0f}km "
                      f"-- dropping {valid[drop]['city']}")

    result = [valid[i] for i in sorted(keep)]
    if len(result) < len(valid):
        dropped = [valid[i]["city"] for i in range(len(valid)) if i not in keep]
        print(f"  [cross-check] kept {[d['city'] for d in result]}, dropped {dropped}")
    return result


# Scoring
def _observation_factor(d_actual: float, d_sign: float, conf: float,
                        sigma: float = SCORE_SIGMA_M) -> float:
    """
    Gaussian likelihood of observing d_sign given actual distance d_actual.
    Low OCR confidence widens the sigma (less influence on weight).
    Capped at MIN_LIKELIHOOD so one bad reading can't zero out a particle.
    """
    sigma_eff = sigma / np.sqrt(max(conf, CONFIDENCE_FLOOR))
    raw = np.exp(-0.5 * ((d_actual - d_sign) / sigma_eff) ** 2)
    return max(raw, MIN_LIKELIHOOD)


def score_particle(p: Particle, observation: list[dict],
                   city_svc: CityDistanceService) -> float:
    """Compute the observation likelihood for a single particle."""
    total = 1.0
    for det in observation:
        city   = det["city"]
        d_sign = det["distance_km"] * 1000
        conf   = det["ocr_confidence"]
        du = city_svc.distance_at_node(city, p.edge_u)
        dv = city_svc.distance_at_node(city, p.edge_v)
        if du is None or dv is None:
            return 0.0
        d_actual = du * (1 - p.t) + dv * p.t
        total *= _observation_factor(d_actual, d_sign, conf)
    return total


# Particle filter
class ParticleFilter:
    def __init__(self, G: nx.MultiDiGraph, edges_gdf: gpd.GeoDataFrame,
                 city_svc: CityDistanceService):
        self.G          = G
        self.city_svc   = city_svc
        self.particles: list[Particle] = []
        self.history:   list[dict]     = []
        self.step_index = 0
        self.initialised = False

        # Pre-filter highway edges for faster initialisation
        print("Pre-filtering highway edges ...")
        t0 = time.time()
        self._highway_edges = [
            e for e in edges_gdf.itertuples()
            if _is_highway_edge(getattr(e, 'highway', None))
        ]
        print(f"  {len(self._highway_edges):,} highway edges ({time.time()-t0:.1f}s)")

        # Pre-build adjacency dict for O(1) neighbour lookup during predict
        print("Pre-building adjacency dict ...")
        t0 = time.time()
        self._adj: dict[int, list[tuple]] = {}
        for u, v, data in self.G.edges(data=True):
            length = data.get("length", 1.0)
            if u not in self._adj:
                self._adj[u] = []
            self._adj[u].append((v, length, data.get("geometry")))
        print(f"  {len(self._adj):,} nodes ({time.time()-t0:.1f}s)")

    # Initialisation
    def initialise(self, observation: list[dict]) -> None:
        """
        Create the initial particle distribution by sampling highway edges
        weighted by how well they match the first sign observation.
        """
        print("\n[Filter] Initialising ...")

        observation = [d for d in observation if self.city_svc.load(d["city"])]
        if not observation:
            raise RuntimeError("No valid cities in observation.")
        print(f"  Cities: {[d['city'] for d in observation]}")

        # Score candidate positions along all highway edges
        pool_lats, pool_lons = [], []
        pool_us, pool_vs     = [], []
        pool_ts, pool_scores = [], []

        for edge in tqdm(self._highway_edges, desc="Sampling highway edges"):
            u, v = edge.Index[0], edge.Index[1]
            geom = edge.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                edge_len_m = edge.length
            except AttributeError:
                edge_len_m = geom.length * 111_320

            # Get road distances at both endpoints for each city
            endpoint_du, endpoint_dv = {}, {}
            skip = False
            for city in [d["city"] for d in observation]:
                du = self.city_svc.distance_at_node(city, u)
                dv = self.city_svc.distance_at_node(city, v)
                if du is None and dv is None:
                    skip = True
                    break
                if du is None: du = dv
                if dv is None: dv = du
                endpoint_du[city] = du
                endpoint_dv[city] = dv
            if skip:
                continue

            # Quick pre-filter: skip edges far from the most confident reading
            top = max(observation, key=lambda d: d["ocr_confidence"])
            tol = SCORE_SIGMA_M * 4
            d_top_u = endpoint_du[top["city"]]
            d_top_v = endpoint_dv[top["city"]]
            d_sign  = top["distance_km"] * 1000
            if max(d_top_u, d_top_v) < d_sign - tol or \
               min(d_top_u, d_top_v) > d_sign + tol:
                continue

            # Sample points along the edge and score each one
            n_samples = max(2, int(edge_len_m / SAMPLE_SPACING_M))
            for t in np.linspace(0.0, 1.0, n_samples):
                pt = geom.interpolate(t, normalized=True)
                score = 1.0
                for det in observation:
                    d_actual = endpoint_du[det["city"]] * (1-t) + \
                               endpoint_dv[det["city"]] * t
                    score *= _observation_factor(
                        d_actual, det["distance_km"] * 1000,
                        det["ocr_confidence"])
                if score > 1e-12:
                    pool_lats.append(pt.y)
                    pool_lons.append(pt.x)
                    pool_us.append(u)
                    pool_vs.append(v)
                    pool_ts.append(t)
                    pool_scores.append(score)

        if not pool_scores:
            raise RuntimeError("No candidate points found.")

        # Draw particles proportional to scores
        print(f"  {len(pool_scores):,} candidate points")
        scores = np.array(pool_scores)
        probs  = scores / scores.sum()
        indices = rng.choice(len(pool_scores), size=N_PARTICLES,
                             replace=True, p=probs)

        self.particles = [
            Particle(lat=pool_lats[i], lon=pool_lons[i],
                     edge_u=pool_us[i], edge_v=pool_vs[i],
                     t=pool_ts[i], weight=1.0 / N_PARTICLES)
            for i in indices
        ]
        self.initialised = True
        self._save_snapshot("init", observation)
        self.step_index = 1

        lat, lon, unc = self.estimate()
        map_lat, map_lon, map_unc = self.estimate_map()
        print(f"  Initial estimate: ({lat:.5f}, {lon:.5f})  unc={unc:.0f}m")
        print(f"  Initial MAP:      ({map_lat:.5f}, {map_lon:.5f})  spread={map_unc:.0f}m")

        # Reject if initial spread is too large (likely a misread sign)
        if unc > 30_000:
            print(f"  WARNING: uncertainty too high ({unc:.0f}m), rejecting init")
            self.particles  = []
            self.history    = []
            self.step_index = 0
            self.initialised = False

    # Predict (motion model)
    def predict(self, delta_distance_m: float = 0.0,
                heading_deg: Optional[float] = None) -> None:
        """
        Propagate each particle forward along the road network.
        At junctions, the particle chooses an outgoing edge biased toward
        the current heading direction.
        """
        if delta_distance_m <= 0.0:
            return

        # Convert compass bearing to math angle (East=0, CCW)
        measured_heading_rad = None
        if heading_deg is not None:
            measured_heading_rad = np.radians(90.0 - heading_deg)

        heading_str = f"  heading {heading_deg:.0f}" if heading_deg is not None else ""
        print(f"\n[Filter] Predict: {delta_distance_m:.0f}m{heading_str} ...")

        noise_std = max(10.0, delta_distance_m * 0.05)

        new_particles = []
        for p in self.particles:
            travel = max(0.0, delta_distance_m + rng.normal(0, noise_std))
            cur_u, cur_v, cur_t = p.edge_u, p.edge_v, p.t

            edge_data = self._get_edge_data(cur_u, cur_v)
            if edge_data is None:
                new_particles.append(p)
                continue
            cur_len = edge_data.get("length", 1.0)
            remaining = cur_len * (1.0 - cur_t)

            # Walk along the network until travel distance is consumed
            while travel > 0:
                if travel < remaining:
                    cur_t = cur_t + (travel / cur_len)
                    cur_t = min(cur_t, 1.0)
                    travel = 0.0
                else:
                    travel -= remaining
                    prev_u = cur_u
                    cur_u  = cur_v
                    cur_t  = 0.0
                    neighbors = self._adj.get(cur_u, [])
                    if not neighbors:
                        break
                    nv, nlen = self._choose_next_edge(
                        prev_u, cur_u, neighbors, measured_heading_rad)
                    cur_v   = nv
                    cur_len = nlen
                    remaining = cur_len

            # Compute lat/lon at the new position on the edge
            lat, lon = self._position_on_edge(cur_u, cur_v, cur_t)
            new_particles.append(Particle(
                lat=lat, lon=lon,
                edge_u=cur_u, edge_v=cur_v,
                t=cur_t, weight=p.weight
            ))

        self.particles = new_particles

        # Snap particles that ended up on wrong-direction edges
        if measured_heading_rad is not None:
            self.particles = self._correct_headings(
                self.particles, measured_heading_rad)

        print(f"  Particles propagated.")

    def _get_edge_data(self, u: int, v: int) -> Optional[dict]:
        """Get edge data dict, handling MultiDiGraph key structure."""
        ed = self.G.get_edge_data(u, v)
        if ed is None:
            return None
        if isinstance(ed, dict) and 0 in ed:
            return ed[0]
        if isinstance(ed, dict):
            first = next(iter(ed.values()), None)
            if isinstance(first, dict):
                return first
        return ed

    def _position_on_edge(self, u: int, v: int, t: float) -> tuple[float, float]:
        """Compute (lat, lon) at position t along edge (u, v)."""
        ed = self._get_edge_data(u, v)
        if ed is not None:
            geom = ed.get("geometry")
            if geom is not None:
                pt = geom.interpolate(t, normalized=True)
                return pt.y, pt.x
        # Fallback: linear interpolation between node coordinates
        nd_u = self.G.nodes[u]
        nd_v = self.G.nodes[v]
        lat = nd_u["y"] * (1 - t) + nd_v["y"] * t
        lon = nd_u["x"] * (1 - t) + nd_v["x"] * t
        return lat, lon

    def _choose_next_edge(self, prev_u, cur_u, neighbors, measured_heading_rad):
        """
        At a junction, pick the next edge weighted by heading alignment.
        Uses compass heading if available (tight sigma), otherwise infers
        direction from the previous edge (looser sigma).
        """
        node_cur = self.G.nodes.get(cur_u)
        if node_cur is None:
            nv, nlen, _ = neighbors[rng.integers(len(neighbors))]
            return nv, nlen

        cos_lat = np.cos(np.radians(node_cur["y"]))

        if measured_heading_rad is not None:
            in_heading = measured_heading_rad
            sigma_rad  = np.radians(HEADING_SIGMA_DEG)
        else:
            node_prev = self.G.nodes.get(prev_u)
            if node_prev is None:
                nv, nlen, _ = neighbors[rng.integers(len(neighbors))]
                return nv, nlen
            dy = node_cur["y"] - node_prev["y"]
            dx = (node_cur["x"] - node_prev["x"]) * cos_lat
            in_heading = np.arctan2(dy, dx)
            sigma_rad  = np.radians(TURN_SIGMA_DEG)

        # Score each outgoing edge by heading alignment
        scores = []
        for nv, nlen, _ in neighbors:
            node_next = self.G.nodes.get(nv)
            if node_next is None:
                scores.append(0.0)
                continue
            dy = node_next["y"] - node_cur["y"]
            dx = (node_next["x"] - node_cur["x"]) * cos_lat
            out_heading = np.arctan2(dy, dx)
            diff = abs((in_heading - out_heading + np.pi) % (2 * np.pi) - np.pi)
            scores.append(float(np.exp(-0.5 * (diff / max(sigma_rad, 1e-6)) ** 2)))

        total = sum(scores)
        if total <= 1e-12:
            nv, nlen, _ = neighbors[rng.integers(len(neighbors))]
            return nv, nlen

        probs = np.array(scores) / total
        idx = int(rng.choice(len(neighbors), p=probs))
        nv, nlen, _ = neighbors[idx]
        return nv, nlen

    def _correct_headings(self, particles, heading_rad):
        """
        Post-predict correction: if a particle's edge direction disagrees with
        the measured heading by more than HEADING_CORRECTION_DEG, try to snap
        it to a better-aligned neighbouring edge.
        """
        threshold = np.radians(HEADING_CORRECTION_DEG)
        sigma = np.radians(HEADING_SIGMA_DEG)
        min_snap_score = 0.3

        result = []
        n_snapped = 0
        for p in particles:
            nd_u = self.G.nodes.get(p.edge_u)
            nd_v = self.G.nodes.get(p.edge_v)
            if nd_u is None or nd_v is None:
                result.append(p)
                continue

            # Check angle between edge direction and measured heading
            cos_lat = np.cos(np.radians((nd_u["y"] + nd_v["y"]) / 2))
            dy = nd_v["y"] - nd_u["y"]
            dx = (nd_v["x"] - nd_u["x"]) * cos_lat
            edge_rad = np.arctan2(dy, dx)
            diff = abs((heading_rad - edge_rad + np.pi) % (2 * np.pi) - np.pi)

            if diff <= threshold:
                result.append(p)
                continue

            # Search 1-hop neighbours from both ends for a better-aligned edge
            best_score = -1.0
            best_hub = best_nv = best_nlen = best_geom = None
            for hub in (p.edge_u, p.edge_v):
                nd_hub = self.G.nodes.get(hub)
                if nd_hub is None:
                    continue
                cos_h = np.cos(np.radians(nd_hub["y"]))
                for nv, nlen, geom in self._adj.get(hub, []):
                    nd_nv = self.G.nodes.get(nv)
                    if nd_nv is None:
                        continue
                    edy = nd_nv["y"] - nd_hub["y"]
                    edx = (nd_nv["x"] - nd_hub["x"]) * cos_h
                    e_rad = np.arctan2(edy, edx)
                    ediff = abs((heading_rad - e_rad + np.pi) % (2*np.pi) - np.pi)
                    score = float(np.exp(-0.5 * (ediff / sigma) ** 2))
                    if score > best_score:
                        best_score = score
                        best_hub, best_nv = hub, nv
                        best_nlen, best_geom = nlen, geom

            if best_score >= min_snap_score and best_hub is not None:
                nd_hub = self.G.nodes[best_hub]
                if best_geom is not None:
                    pt = best_geom.interpolate(0.0, normalized=True)
                    new_lat, new_lon = pt.y, pt.x
                else:
                    new_lat, new_lon = nd_hub["y"], nd_hub["x"]
                result.append(Particle(
                    lat=new_lat, lon=new_lon,
                    edge_u=best_hub, edge_v=best_nv,
                    t=0.0, weight=p.weight))
                n_snapped += 1
            else:
                result.append(p)

        if n_snapped:
            print(f"  Heading correction: {n_snapped}/{len(particles)} snapped")
        return result

    # Update (observation model)
    def update(self, observation: list[dict]) -> None:
        """Reweight particles based on observed city distances from a sign."""
        target_step = self.step_index + 1
        print(f"\n[Filter] Update #{target_step} ...")

        observation = [d for d in observation if self.city_svc.load(d["city"])]
        if not observation:
            print("  WARNING: no valid cities, skipping update")
            return

        new_weights = np.array([
            p.weight * score_particle(p, observation, self.city_svc)
            for p in tqdm(self.particles, desc="Reweighting")
        ])
        total_w = new_weights.sum()
        if total_w == 0:
            print("  WARNING: all weights zero, keeping previous distribution")
            return
        new_weights /= total_w

        # Reject if observation would teleport the estimate > 100 km
        cur_lat, cur_lon, _ = self.estimate()
        new_lat = float(np.average([p.lat for p in self.particles], weights=new_weights))
        new_lon = float(np.average([p.lon for p in self.particles], weights=new_weights))
        jump_m = 111_320 * np.sqrt(
            (new_lat - cur_lat) ** 2 +
            ((new_lon - cur_lon) * np.cos(np.radians(cur_lat))) ** 2)
        if jump_m > 100_000:
            print(f"  WARNING: {jump_m/1000:.0f}km jump, observation rejected")
            return

        for p, w in zip(self.particles, new_weights):
            p.weight = w

        ess = 1.0 / np.sum(new_weights ** 2)
        ess_ratio = ess / N_PARTICLES
        print(f"  ESS = {ess:.0f}/{N_PARTICLES} ({100*ess_ratio:.1f}%)")

        if ess_ratio < RESAMPLE_THRESHOLD:
            self._systematic_resample(new_weights)

        self.step_index = target_step
        self._save_snapshot(f"update_{self.step_index}", observation)
        lat, lon, unc = self.estimate()
        map_lat, map_lon, map_unc = self.estimate_map()
        print(f"  Estimate: ({lat:.5f}, {lon:.5f})  unc={unc:.0f}m")
        print(f"  MAP:      ({map_lat:.5f}, {map_lon:.5f})  spread={map_unc:.0f}m")

    # Resampling
    def _systematic_resample(self, weights: np.ndarray) -> None:
        """
        Systematic resampling with roughening: duplicate high-weight particles,
        remove low-weight ones, then jitter positions along their edges to
        maintain diversity.
        """
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        positions = (rng.random() + np.arange(N_PARTICLES)) / N_PARTICLES
        indices   = np.searchsorted(cumsum, positions)

        new_particles = []
        for i in indices:
            p = self.particles[i]
            ed = self._get_edge_data(p.edge_u, p.edge_v)
            edge_len = ed.get("length", 1.0) if ed else 1.0
            geom = ed.get("geometry") if ed else None

            # Jitter position along the edge
            sigma_t = min(ROUGHENING_SIGMA_M / max(edge_len, 1.0), 0.15)
            t_new = np.clip(p.t + rng.normal(0, sigma_t), 0.0, 1.0)
            lat, lon = self._position_on_edge(p.edge_u, p.edge_v, t_new)

            new_particles.append(Particle(
                lat=lat, lon=lon,
                edge_u=p.edge_u, edge_v=p.edge_v,
                t=t_new, weight=1.0 / N_PARTICLES))

        self.particles = new_particles
        print(f"  Resampled {N_PARTICLES} particles")

    # Estimation
    def estimate(self) -> tuple[float, float, float]:
        """Weighted mean estimate. Returns (lat, lon, uncertainty_m)."""
        weights = np.array([p.weight for p in self.particles])
        lats    = np.array([p.lat for p in self.particles])
        lons    = np.array([p.lon for p in self.particles])
        w_lat = np.average(lats, weights=weights)
        w_lon = np.average(lons, weights=weights)
        var_lat = np.average((lats - w_lat)**2, weights=weights)
        var_lon = np.average((lons - w_lon)**2, weights=weights)
        cos_lat = np.cos(np.radians(w_lat))
        unc_m = 111_320 * np.sqrt(var_lat + var_lon * cos_lat**2)
        return w_lat, w_lon, unc_m

    def estimate_map(self) -> tuple[float, float, float]:
        """
        MAP estimate: the highest-weight particle's position.
        When weights are uniform (after resampling), picks the particle
        closest to the mean instead.
        Returns (lat, lon, spread_m).
        """
        weights = np.array([p.weight for p in self.particles])
        lats    = np.array([p.lat for p in self.particles])
        lons    = np.array([p.lon for p in self.particles])

        if weights.max() - weights.min() < 1e-12:
            # Uniform weights, pick particle closest to mean
            mean_lat, mean_lon = float(np.mean(lats)), float(np.mean(lons))
            cos_lat = np.cos(np.radians(mean_lat))
            dy = 111_320.0 * (lats - mean_lat)
            dx = 111_320.0 * cos_lat * (lons - mean_lon)
            idx = int(np.argmin(dy*dy + dx*dx))
        else:
            idx = int(np.argmax(weights))

        p0 = self.particles[idx]
        dy = 111_320.0 * (lats - p0.lat)
        dx = 111_320.0 * np.cos(np.radians(p0.lat)) * (lons - p0.lon)
        spread_m = float(np.sqrt(np.average(dy*dy + dx*dx, weights=weights)))
        return p0.lat, p0.lon, spread_m

    def predicted_distance_km(self, city: str) -> Optional[float]:
        """
        Weighted-mean predicted road distance (km) from the particle
        distribution to a city. Used for single-city observation validation.
        """
        if not self.city_svc.load(city):
            return None
        weights = np.array([p.weight for p in self.particles])
        dists = []
        for p in self.particles:
            du = self.city_svc.distance_at_node(city, p.edge_u)
            dv = self.city_svc.distance_at_node(city, p.edge_v)
            if du is None or dv is None:
                dists.append(np.nan)
            else:
                dists.append(du * (1 - p.t) + dv * p.t)
        dists = np.array(dists)
        valid = ~np.isnan(dists)
        if not valid.any():
            return None
        w = weights[valid]
        return float(np.average(dists[valid], weights=w / w.sum())) / 1000.0

    # History
    def _save_snapshot(self, label: str, observation: list[dict]) -> None:
        lat, lon, unc = self.estimate()
        map_lat, map_lon, map_unc = self.estimate_map()
        self.history.append({
            "label":        label,
            "observation":  observation,
            "particles":    copy.deepcopy(self.particles),
            "estimate":     {"lat": lat, "lon": lon, "uncertainty_m": unc},
            "estimate_map": {"lat": map_lat, "lon": map_lon, "uncertainty_m": map_unc},
        })


# Visualisation
REFERENCE_CITIES = {
    "Tallinn":  (59.4370, 24.7536),
    "Tartu":    (58.3780, 26.7290),
    "Parnu":    (58.3859, 24.4971),
    "Narva":    (59.3772, 28.1910),
    "Viljandi": (58.3639, 25.5897),
    "Rakvere":  (59.3469, 26.3551),
    "Paide":    (58.8858, 25.5572),
}


def _draw_overlay(ax, pf, city_svc, particles, est, weights):
    """Shared drawing logic for particle maps."""
    max_w = weights.max() if weights.max() > 0 else 1.0
    lats = np.array([p.lat for p in particles])
    lons = np.array([p.lon for p in particles])

    ax.scatter(lons, lats, c=weights/max_w, cmap='plasma',
               s=12, alpha=0.7, zorder=3, linewidths=0)

    # Trajectory trail
    if len(pf.history) > 1:
        ax.plot([s["estimate_map"]["lon"] for s in pf.history],
                [s["estimate_map"]["lat"] for s in pf.history],
                color='cyan', linewidth=1.5, alpha=0.7, zorder=4)

    # Current position marker
    ax.scatter(est["lon"], est["lat"], s=120, color='cyan',
               marker='+', linewidths=2.5, zorder=6)

    # Observed city markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    final = pf.history[-1]
    for j, det in enumerate(final["observation"]):
        gc = city_svc.geocode(det["city"])
        ax.scatter(gc[1], gc[0], s=80, color=colors[j % 4],
                   marker='D', zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(det["city"], (gc[1], gc[0]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8,
                    color=colors[j % 4], fontweight='bold')

    # Reference city labels
    for city, (rlat, rlon) in REFERENCE_CITIES.items():
        ax.scatter(rlon, rlat, s=40, color='white', marker='o',
                   alpha=0.5, zorder=4, linewidths=0)
        ax.annotate(city, (rlon, rlat), xytext=(3, 3),
                    textcoords='offset points', fontsize=7,
                    color='white', alpha=0.6)


def plot_particle_map(pf: ParticleFilter, G: nx.MultiDiGraph,
                      city_svc: CityDistanceService,
                      output_dir: Path = Path("results"),
                      margin: float = 0.15) -> None:
    """Save full-country and zoomed particle maps for the current step."""
    final   = pf.history[-1]
    est     = final["estimate_map"]
    parts   = final["particles"]
    weights = np.array([p.weight for p in parts])
    lons    = np.array([p.lon for p in parts])
    lats    = np.array([p.lat for p in parts])

    title = (f"Step {pf.step_index}  |  {len(parts)} particles  |  "
             f"({est['lat']:.4f}, {est['lon']:.4f})  |  "
             f"unc={est['uncertainty_m']:.0f}m")

    # Full map
    fig1, ax1 = ox.plot_graph(G, show=False, close=False, bgcolor='#0e1117',
                              node_size=0, edge_color='#333844',
                              edge_linewidth=0.5, figsize=(14, 10))
    _draw_overlay(ax1, pf, city_svc, parts, est, weights)
    ax1.set_title("Full, " + title, color='white', fontsize=9)
    p1 = output_dir / f"map_full_step{pf.step_index}.png"
    plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()

    # Zoomed map
    lon_min, lon_max = lons.min() - margin, lons.max() + margin
    lat_min, lat_max = lats.min() - margin, lats.max() + margin
    for snap in pf.history:
        e = snap["estimate_map"]
        lon_min = min(lon_min, e["lon"] - margin)
        lon_max = max(lon_max, e["lon"] + margin)
        lat_min = min(lat_min, e["lat"] - margin)
        lat_max = max(lat_max, e["lat"] + margin)

    bbox = (lat_max, lat_min, lon_max, lon_min)
    fig2, ax2 = ox.plot_graph(G, show=False, close=False, bbox=bbox,
                              bgcolor='#0e1117', node_size=0,
                              edge_color='#333844', edge_linewidth=0.8,
                              figsize=(14, 10))
    _draw_overlay(ax2, pf, city_svc, parts, est, weights)
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_title("Zoomed, " + title, color='white', fontsize=9)
    p2 = output_dir / f"map_zoomed_step{pf.step_index}.png"
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()

    print(f"  Maps saved: {p1.name}, {p2.name}")


def plot_latest_map(pf: ParticleFilter, G: nx.MultiDiGraph,
                    city_svc: CityDistanceService,
                    output_dir: Path = Path("results"),
                    zoom: float = 0.08) -> None:
    """Lightweight zoomed map updated every frame, saved as latest_map.png."""
    if not pf.history:
        return

    est   = pf.history[-1]["estimate_map"]
    parts = pf.history[-1]["particles"]
    weights = np.array([p.weight for p in parts])
    max_w = weights.max() if weights.max() > 0 else 1.0
    lons  = np.array([p.lon for p in parts])
    lats  = np.array([p.lat for p in parts])

    lat_min, lat_max = est["lat"] - zoom, est["lat"] + zoom
    lon_min, lon_max = est["lon"] - zoom, est["lon"] + zoom

    title = (f"Step {pf.step_index}  |  "
             f"({est['lat']:.4f}, {est['lon']:.4f})  |  "
             f"spread={est['uncertainty_m']:.0f}m")

    bbox = (lat_max, lat_min, lon_max, lon_min)
    fig, ax = ox.plot_graph(G, show=False, close=False, bbox=bbox,
                            bgcolor='#0e1117', node_size=0,
                            edge_color='#333844', edge_linewidth=0.8,
                            figsize=(10, 7))

    mask = ((lons >= lon_min) & (lons <= lon_max) &
            (lats >= lat_min) & (lats <= lat_max))
    if mask.any():
        ax.scatter(lons[mask], lats[mask], c=(weights/max_w)[mask],
                   cmap='plasma', s=10, alpha=0.7, zorder=3, linewidths=0)

    if len(pf.history) > 1:
        ax.plot([s["estimate_map"]["lon"] for s in pf.history],
                [s["estimate_map"]["lat"] for s in pf.history],
                color='cyan', linewidth=1.5, alpha=0.7, zorder=4)

    ax.scatter(est["lon"], est["lat"], s=120, color='cyan',
               marker='+', linewidths=2.5, zorder=6)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title("Latest, " + title, color='white', fontsize=9)

    out = output_dir / "latest_map.png"
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0e1117')
    plt.close()

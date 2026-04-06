"""
bag_replay.py — Replay a ROS1 .bag file through the particle filter.

Reads camera frames and NovAtel INSPVA telemetry (heading + speed) from a
.bag file and feeds them into the same OCR + particle filter pipeline as
app.py, without any manual input.

Usage:
    python bag_replay.py path/to/file.bag
    python bag_replay.py path/to/file.bag --camera /interfacea/link3/image/compressed
    python bag_replay.py path/to/file.bag --reset
"""

import argparse
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
CACHE_DIR   = Path("cache")

for d in [RESULTS_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

# ── Topic names ───────────────────────────────────────────────────────────────

DEFAULT_CAMERA_TOPIC = "/interfacea/link3/image/compressed"
INSPVA_TOPIC         = "/novatel/oem7/inspva"


# ── Telemetry helpers ─────────────────────────────────────────────────────────

def _load_inspva(bag_path: str) -> list[dict]:
    """Extract INSPVA telemetry via bagpy (handles custom NovAtel types reliably)."""
    import bagpy
    b        = bagpy.bagreader(bag_path)
    csv_path = b.message_by_topic(INSPVA_TOPIC)
    df       = pd.read_csv(csv_path)
    records  = []
    for _, row in df.iterrows():
        t_ns  = int(row["Time"] * 1e9)
        speed = math.sqrt(row["north_velocity"] ** 2 + row["east_velocity"] ** 2)
        records.append({
            "t":       t_ns,
            "heading": float(row["azimuth"]),   # degrees, 0 = North, clockwise
            "speed":   speed,                   # m/s
        })
    records.sort(key=lambda r: r["t"])
    return records


def _nearest_inspva(records: list[dict], t_ns: int) -> dict | None:
    """Binary search for the INSPVA record nearest to t_ns."""
    if not records:
        return None
    lo, hi = 0, len(records) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if records[mid]["t"] < t_ns:
            lo = mid + 1
        else:
            hi = mid
    best = records[lo]
    if lo > 0 and abs(records[lo - 1]["t"] - t_ns) < abs(best["t"] - t_ns):
        best = records[lo - 1]
    return best


def _distance_between(records: list[dict], t_start_ns: int, t_end_ns: int) -> float:
    """
    Estimate distance (metres) travelled between two timestamps by averaging
    speed across all INSPVA samples in that window × elapsed time.
    """
    window = [r for r in records if t_start_ns <= r["t"] <= t_end_ns]
    if not window:
        return 0.0
    avg_speed = sum(r["speed"] for r in window) / len(window)
    dt_s      = (t_end_ns - t_start_ns) / 1e9
    return avg_speed * dt_s


# ── Image helpers ─────────────────────────────────────────────────────────────

def _decode_compressed(data: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes from a CompressedImage message to a BGR array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── Observation deduplication (mirrors app.py) ────────────────────────────────

def _deduplicate(destinations: list[dict]) -> list[dict]:
    by_conf = sorted(destinations, key=lambda d: -d["ocr_confidence"])
    seen, out = set(), []
    for d in by_conf:
        if d["city"] not in seen:
            seen.add(d["city"])
            out.append(d)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Path to .bag file")
    parser.add_argument("--camera", default=DEFAULT_CAMERA_TOPIC,
                        help="Compressed image topic (default: %(default)s)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear results before starting")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"ERROR: bag file not found: {bag_path}")
        return

    if args.reset:
        removed = sum(1 for _ in RESULTS_DIR.iterdir())
        for entry in RESULTS_DIR.iterdir():
            entry.unlink() if entry.is_file() else shutil.rmtree(entry)
        print(f"Results cleared ({removed} item(s)).")

    # ── Load models and graphs ────────────────────────────────────────────────
    print("Loading OCR and localisation modules ...")
    from ocr import process_image
    from localizer import (load_graph,
                           CityDistanceService, ParticleFilter,
                           plot_particle_map, plot_latest_map)
    import osmnx as ox

    print("Loading road graph ...")
    G            = load_graph()
    _, edges_gdf = ox.graph_to_gdfs(G)
    city_svc     = CityDistanceService(G)
    pf           = ParticleFilter(G, edges_gdf, city_svc)

    # ── Open bag ──────────────────────────────────────────────────────────────
    print(f"\nOpening bag: {bag_path}")

    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    # Load INSPVA telemetry via bagpy before opening the rosbags Reader
    print("Loading INSPVA telemetry ...")
    inspva = _load_inspva(str(bag_path))
    if inspva:
        print(f"  {len(inspva):,} INSPVA records  "
              f"({inspva[0]['t']/1e9:.1f}s – {inspva[-1]['t']/1e9:.1f}s)")
    else:
        print("  WARNING: no INSPVA records found")

    # Standard ROS1 Noetic typestore is sufficient for sensor_msgs/CompressedImage
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:

        camera_conns = [c for c in reader.connections if c.topic == args.camera]
        if not camera_conns:
            available = [c.topic for c in reader.connections]
            print(f"ERROR: camera topic '{args.camera}' not found.")
            print("Available topics:", available)
            return

        print(f"Replaying: {args.camera}")
        print(f"{'═'*55}\n")

        last_t_ns = None
        frame_idx = 0
        obs_count = 0   # number of sign-observation saves

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img = Path(tmpdir) / "frame.jpg"

            for connection, timestamp, rawdata in reader.messages(connections=camera_conns):
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                img = _decode_compressed(bytes(msg.data))
                if img is None:
                    print(f"  [frame {frame_idx}] decode failed — skipping.")
                    last_t_ns = timestamp
                    frame_idx += 1
                    continue

                cv2.imwrite(str(tmp_img), img)

                print(f"{'─'*55}")
                print(f"  Frame {frame_idx}  |  bag_t={timestamp/1e9:.2f}s")

                # Telemetry
                tel        = _nearest_inspva(inspva, timestamp)
                heading    = tel["heading"] if tel else None
                distance_m = 0.0
                if last_t_ns is not None:
                    distance_m = _distance_between(inspva, last_t_ns, timestamp)
                if tel:
                    print(f"  distance={distance_m:.1f}m  heading={heading:.1f}°  "
                          f"speed={tel['speed']:.1f}m/s")

                # OCR
                try:
                    destinations = _deduplicate(process_image(str(tmp_img)))
                except Exception as e:
                    print(f"  OCR error: {e}")
                    destinations = []

                # Particle filter
                try:
                    if not destinations:
                        if not pf.initialised:
                            print("  No sign and not initialised — skipping.")
                            last_t_ns = timestamp
                            frame_idx += 1
                            continue
                        print("  No sign — predict only.")
                        pf.predict(delta_distance_m=distance_m, heading_deg=heading)
                        pf.step_index += 1
                        pf._save_snapshot(f"predict_{pf.step_index}", [])
                    elif not pf.initialised:
                        pf.initialise(destinations)
                    else:
                        pf.predict(delta_distance_m=distance_m, heading_deg=heading)
                        pf.update(destinations)
                except Exception as e:
                    print(f"  Filter error: {e}")
                    last_t_ns = timestamp
                    frame_idx += 1
                    continue

                # If initialisation was rejected (bad observation), skip saving.
                if not pf.initialised:
                    last_t_ns = timestamp
                    frame_idx += 1
                    continue

                # Build result dict (used for both latest.json and obs files).
                # MAP estimate (best particle) is primary — always on a road.
                # Weighted mean is secondary — can fall off-road between clusters.
                map_lat, map_lon, map_unc = pf.estimate_map()
                mean_lat, mean_lon, mean_unc = pf.estimate()

                result = {
                    "filter_step": pf.step_index,
                    "timestamp":   datetime.now().isoformat(),
                    "bag_time_s":  timestamp / 1e9,
                    "frame":       frame_idx,
                    "observation": destinations,
                    "telemetry": {
                        "distance_m": round(distance_m, 2),
                        "heading_deg": round(heading, 2) if heading is not None else None,
                        "speed_ms":   round(tel["speed"], 2) if tel else None,
                    },
                    "estimate":      {"lat": map_lat,  "lon": map_lon,  "uncertainty_m": map_unc},
                    "estimate_mean": {"lat": mean_lat, "lon": mean_lon, "uncertainty_m": mean_unc},
                }

                # Always update latest.json (current live position)
                with open(RESULTS_DIR / "latest.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                print(f"  → lat={map_lat:.6f}  lon={map_lon:.6f}  unc≈{map_unc:.0f}m")

                # Only save a numbered file when a sign observation was used
                if destinations:
                    obs_count += 1
                    out = RESULTS_DIR / f"obs{obs_count:04d}_frame{frame_idx:04d}.json"
                    with open(out, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"  Observation #{obs_count} saved → {out.name}")

                    try:
                        plot_particle_map(pf, G, city_svc, output_dir=RESULTS_DIR)
                    except Exception as e:
                        print(f"  Map warning: {e}")

                # Always update latest_map.png (every frame)
                try:
                    plot_latest_map(pf, G, city_svc, output_dir=RESULTS_DIR)
                except Exception as e:
                    print(f"  Latest map warning: {e}")

                last_t_ns = timestamp
                frame_idx += 1

    print(f"\n{'═'*55}")
    print(f"Done. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()

"""
bag_replay.py — Replay a ROS1 .bag file through the particle filter.

Reads camera frames and NovAtel INSPVA telemetry (heading + speed) from a
.bag file and feeds them into the OCR + particle filter pipeline.

Usage:
    python bag_replay.py path/to/file.bag
    python bag_replay.py path/to/file.bag --camera /interfacea/link3/image/compressed
    python bag_replay.py path/to/file.bag --reset
"""

import argparse
import json
import math
import signal
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Paths and topics ─────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
CACHE_DIR   = Path("cache")

for d in [RESULTS_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

DEFAULT_CAMERA_TOPIC = "/interfacea/link3/image/compressed"
INSPVA_TOPIC         = "/novatel/oem7/inspva"


# ── Telemetry ────────────────────────────────────────────────────────────────

def _load_inspva(bag_path: str) -> list[dict]:
    """Extract INSPVA telemetry (heading, speed, position) via bagpy."""
    import bagpy
    import pandas as pd
    b        = bagpy.bagreader(bag_path)
    csv_path = b.message_by_topic(INSPVA_TOPIC)
    df       = pd.read_csv(csv_path)
    records  = []
    for _, row in df.iterrows():
        speed = math.sqrt(row["north_velocity"]**2 + row["east_velocity"]**2)
        records.append({
            "t":       int(row["Time"] * 1e9),
            "heading": float(row["azimuth"]),
            "speed":   speed,
            "lat":     float(row["latitude"]),
            "lon":     float(row["longitude"]),
        })
    records.sort(key=lambda r: r["t"])
    return records


def _nearest_inspva(records: list[dict], t_ns: int) -> dict | None:
    """Find the INSPVA record nearest to timestamp t_ns (binary search)."""
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
    if lo > 0 and abs(records[lo-1]["t"] - t_ns) < abs(best["t"] - t_ns):
        best = records[lo - 1]
    return best


def _distance_between(records: list[dict], t_start_ns: int, t_end_ns: int) -> float:
    """Estimate distance travelled between two timestamps using average speed."""
    window = [r for r in records if t_start_ns <= r["t"] <= t_end_ns]
    if not window:
        return 0.0
    avg_speed = sum(r["speed"] for r in window) / len(window)
    dt_s = (t_end_ns - t_start_ns) / 1e9
    return avg_speed * dt_s


# ── Image decoding ───────────────────────────────────────────────────────────

def _decode_compressed(data: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes from a CompressedImage message."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── OCR deduplication ────────────────────────────────────────────────────────

def _deduplicate(destinations: list[dict]) -> list[dict]:
    """Keep only the highest-confidence reading per city."""
    by_conf = sorted(destinations, key=lambda d: -d["ocr_confidence"])
    seen, out = set(), []
    for d in by_conf:
        if d["city"] not in seen:
            seen.add(d["city"])
            out.append(d)
    return out


# ── Temporal sign averaging ──────────────────────────────────────────────────

def _flush_sign_buffer(sign_buffer: dict) -> list[dict]:
    """
    Fuse buffered readings per city into a single observation.
    Distance is a confidence-weighted average across frames.
    """
    if not sign_buffer:
        return []
    fused = []
    for city, readings in sign_buffer.items():
        confs = [r["ocr_confidence"] for r in readings]
        total_conf = sum(confs)
        if total_conf > 0:
            w_dist = sum(r["distance_km"] * r["ocr_confidence"]
                         for r in readings) / total_conf
        else:
            w_dist = sum(r["distance_km"] for r in readings) / len(readings)
        avg_conf = total_conf / len(readings)
        fused.append({
            "city":           city,
            "distance_km":    round(w_dist, 1),
            "ocr_confidence": round(avg_conf, 3),
        })
        print(f"  [fused {len(readings)} frames] {city}: "
              f"{w_dist:.1f}km  conf={avg_conf:.2f}")
    sign_buffer.clear()
    return fused


# ── Evaluation plots ─────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def generate_evaluation(track: list[dict], G, output_dir: Path) -> None:
    """
    Generate two evaluation outputs:
      1. evaluation.png  -- predicted track and GPS track side by side on road graph
      2. error_plot.png  -- position error % vs distance travelled
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    if not track:
        print("  No evaluation data.")
        return

    gps_lats  = [r["gps_lat"]  for r in track]
    gps_lons  = [r["gps_lon"]  for r in track]
    pred_lats = [r["pred_lat"] for r in track]
    pred_lons = [r["pred_lon"] for r in track]

    # Compute errors and cumulative distance
    elapsed, errors = [], []
    t_acc = 0.0
    for row in track:
        t_acc += row["dist_m"]
        elapsed.append(t_acc)
        errors.append(_haversine_m(
            row["pred_lat"], row["pred_lon"],
            row["gps_lat"],  row["gps_lon"]))

    max_err = max(errors) if errors else 1.0
    errors_pct = [e / max_err * 100 for e in errors]

    # Shared bounding box covering both tracks
    all_lats = gps_lats + pred_lats
    all_lons = gps_lons + pred_lons
    margin = 0.05
    bbox = (max(all_lats) + margin, min(all_lats) - margin,
            max(all_lons) + margin, min(all_lons) - margin)
    lon_min, lon_max = min(all_lons) - margin, max(all_lons) + margin
    lat_min, lat_max = min(all_lats) - margin, max(all_lats) + margin

    # ── evaluation.png: side-by-side maps on road graph ─────────────────
    # osmnx creates its own figure, so render each panel separately
    # then combine into one figure using their pixel data.
    import io
    from PIL import Image

    def _render_track(lons, lats, color, title):
        fig_, ax_ = ox.plot_graph(G, show=False, close=False, bbox=bbox,
                                  bgcolor='#0e1117', node_size=0,
                                  edge_color='#2a2f3a', edge_linewidth=0.5,
                                  figsize=(9, 9))
        ax_.plot(lons, lats, color=color, linewidth=2.5,
                 alpha=0.9, zorder=4, solid_capstyle='round')
        ax_.set_xlim(lon_min, lon_max)
        ax_.set_ylim(lat_min, lat_max)
        ax_.set_title(title, color='white', fontsize=13,
                      fontweight='bold', pad=10)
        buf = io.BytesIO()
        fig_.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#0e1117')
        plt.close(fig_)
        buf.seek(0)
        return Image.open(buf).copy()

    img_pred = _render_track(pred_lons, pred_lats, '#3a9bdc', 'Predicted track')
    img_gps  = _render_track(gps_lons,  gps_lats,  '#e07b39', 'GPS ground truth')

    combined_w = img_pred.width + img_gps.width
    combined_h = max(img_pred.height, img_gps.height)
    combined = Image.new('RGB', (combined_w, combined_h), (14, 17, 23))
    combined.paste(img_pred, (0, 0))
    combined.paste(img_gps,  (img_pred.width, 0))

    p1 = output_dir / "evaluation.png"
    combined.save(str(p1), dpi=(150, 150))
    print(f"  Evaluation map saved: {p1.name}")

    # ── error_plot.png: error % vs time elapsed ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(elapsed, errors_pct, color='black', linewidth=1.5)
    ax.set_xlabel('Time elapsed', fontsize=11)
    ax.set_ylabel('Error %', fontsize=11)
    ax.set_ylim(0, 105)
    ax.tick_params(left=True, bottom=False, labelbottom=False)
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    p2 = output_dir / "error_plot.png"
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Error plot saved: {p2.name}")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a ROS1 bag through the particle filter pipeline")
    parser.add_argument("bag", help="Path to .bag file")
    parser.add_argument("--camera", default=DEFAULT_CAMERA_TOPIC,
                        help="Camera topic (default: %(default)s)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear results before starting")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"ERROR: bag file not found: {bag_path}")
        return

    if args.reset:
        for entry in RESULTS_DIR.iterdir():
            entry.unlink() if entry.is_file() else shutil.rmtree(entry)
        print("Results cleared.")

    # ── Load models ──────────────────────────────────────────────────────
    print("Loading OCR and localization modules ...")
    from ocr import process_image
    from localizer import (load_graph, CityDistanceService, ParticleFilter,
                           cross_check_observation,
                           plot_particle_map)
    import osmnx as ox

    print("Loading road graph ...")
    G            = load_graph()
    _, edges_gdf = ox.graph_to_gdfs(G)
    city_svc     = CityDistanceService(G)
    pf           = ParticleFilter(G, edges_gdf, city_svc)

    # ── Load telemetry ───────────────────────────────────────────────────
    print(f"\nOpening bag: {bag_path}")
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    print("Loading INSPVA telemetry ...")
    inspva = _load_inspva(str(bag_path))
    if inspva:
        print(f"  {len(inspva):,} records")
    else:
        print("  WARNING: no INSPVA records found")

    typestore = get_typestore(Stores.ROS1_NOETIC)

    # State variables (declared before try so Ctrl+C handler can reach them)
    frame_idx   = 0
    eval_track: list[dict]          = []
    sign_buffer: dict[str, list]    = {}

    # ── Replay loop ──────────────────────────────────────────────────────
    MIN_SPEED_MS        = 1.0     # below this, distance = 0
    MIN_SPEED_HEADING   = 3.0     # below this, heading is unreliable
    SIGN_BUFFER_FRAMES  = 10      # max frames to buffer a visible sign
    PREDICT_EVERY       = 5       # run predict every N frames
    SINGLE_CITY_TOL_KM  = 30      # reject single-city obs if off by more

    with Reader(bag_path) as reader:
        camera_conns = [c for c in reader.connections if c.topic == args.camera]
        if not camera_conns:
            print(f"ERROR: topic '{args.camera}' not found.")
            return

        print(f"Replaying: {args.camera}")
        print(f"{'='*55}\n")

        last_t_ns        = None
        obs_count        = 0
        sign_buf_frames  = 0
        accum_dist       = 0.0
        frames_since_pred = 0
        last_heading     = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img = Path(tmpdir) / "frame.jpg"

            # Graceful Ctrl+C: set flag, loop checks it each iteration
            stop_requested = False
            def handle_sigint(signum, frame):
                nonlocal stop_requested
                stop_requested = True
                print("\n  [Ctrl+C -- stopping after this frame]")
            prev_handler = signal.signal(signal.SIGINT, handle_sigint)

            for conn, timestamp, rawdata in reader.messages(connections=camera_conns):
                if stop_requested:
                    print(f"\nStopping at frame {frame_idx}.")
                    break

                # Decode image
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                img = _decode_compressed(bytes(msg.data))
                if img is None:
                    last_t_ns = timestamp
                    frame_idx += 1
                    continue
                cv2.imwrite(str(tmp_img), img)

                print(f"{'~'*55}")
                print(f"  Frame {frame_idx}  |  t={timestamp/1e9:.2f}s")

                # ── Telemetry ────────────────────────────────────────
                tel = _nearest_inspva(inspva, timestamp)
                spd = tel["speed"] if tel else 0.0

                distance_m = 0.0
                if last_t_ns is not None and spd >= MIN_SPEED_MS:
                    distance_m = _distance_between(inspva, last_t_ns, timestamp)

                heading = (tel["heading"]
                           if tel and spd >= MIN_SPEED_HEADING
                           else None)

                if tel:
                    hdg_str = (f"{heading:.1f}" if heading is not None
                               else f"suppressed ({spd:.1f}m/s)")
                    print(f"  dist={distance_m:.1f}m  hdg={hdg_str}  spd={spd:.1f}m/s")

                # ── OCR ──────────────────────────────────────────────
                try:
                    raw_dests = _deduplicate(process_image(str(tmp_img)))
                except Exception as e:
                    print(f"  OCR error: {e}")
                    raw_dests = []

                # ── Temporal averaging ───────────────────────────────
                # Buffer sign readings across frames, flush when sign
                # leaves view or buffer is full.
                destinations = []
                if not raw_dests:
                    if sign_buffer:
                        print("  Sign left view -- flushing buffer.")
                        destinations = _flush_sign_buffer(sign_buffer)
                        sign_buf_frames = 0
                else:
                    for det in raw_dests:
                        sign_buffer.setdefault(det["city"], []).append(det)
                    sign_buf_frames += 1
                    if sign_buf_frames >= SIGN_BUFFER_FRAMES:
                        print(f"  Buffer full -- flushing.")
                        destinations = _flush_sign_buffer(sign_buffer)
                        sign_buf_frames = 0

                # ── Multi-city cross-check ───────────────────────────
                if len(destinations) >= 2 and pf.initialised:
                    destinations = cross_check_observation(destinations, city_svc)

                # ── Single-city plausibility check ───────────────────
                # Reject if OCR distance is far from what particles predict
                if len(destinations) == 1 and pf.initialised:
                    det = destinations[0]
                    pred_km = pf.predicted_distance_km(det["city"])
                    if pred_km is not None:
                        diff = abs(det["distance_km"] - pred_km)
                        if diff > SINGLE_CITY_TOL_KM:
                            print(f"  [single-city] {det['city']}: "
                                  f"OCR={det['distance_km']:.0f}km vs "
                                  f"pred={pred_km:.0f}km -- rejected")
                            destinations = []

                # ── Particle filter step ─────────────────────────────
                accum_dist += distance_m
                frames_since_pred += 1
                if heading is not None:
                    last_heading = heading

                try:
                    if destinations:
                        if not pf.initialised:
                            pf.initialise(destinations)
                        else:
                            pf.predict(delta_distance_m=accum_dist,
                                       heading_deg=last_heading)
                            pf.step_index += 1
                            pf.update(destinations)
                        accum_dist = 0.0
                        frames_since_pred = 0

                    elif frames_since_pred >= PREDICT_EVERY:
                        if not pf.initialised:
                            print("  Not initialised -- skipping.")
                            accum_dist = 0.0
                            frames_since_pred = 0
                            last_t_ns = timestamp
                            frame_idx += 1
                            continue
                        print(f"  Predict batch ({frames_since_pred} frames, "
                              f"{accum_dist:.0f}m)")
                        pf.predict(delta_distance_m=accum_dist,
                                   heading_deg=last_heading)
                        pf.step_index += 1
                        pf._save_snapshot(f"predict_{pf.step_index}", [])
                        accum_dist = 0.0
                        frames_since_pred = 0
                    else:
                        last_t_ns = timestamp
                        frame_idx += 1
                        continue

                except Exception as e:
                    print(f"  Filter error: {e}")
                    last_t_ns = timestamp
                    frame_idx += 1
                    continue

                if not pf.initialised:
                    last_t_ns = timestamp
                    frame_idx += 1
                    continue

                # ── Save results ─────────────────────────────────────
                map_lat, map_lon, map_unc = pf.estimate_map()
                mean_lat, mean_lon, mean_unc = pf.estimate()

                result = {
                    "filter_step": pf.step_index,
                    "timestamp":   datetime.now().isoformat(),
                    "bag_time_s":  timestamp / 1e9,
                    "frame":       frame_idx,
                    "observation": destinations,
                    "telemetry": {
                        "distance_m":  round(distance_m, 2),
                        "heading_deg": round(heading, 2) if heading else None,
                        "speed_ms":    round(tel["speed"], 2) if tel else None,
                    },
                    "gps": {
                        "lat": tel["lat"], "lon": tel["lon"],
                    } if tel and "lat" in tel else None,
                    "estimate":      {"lat": map_lat,  "lon": map_lon,
                                      "uncertainty_m": map_unc},
                    "estimate_mean": {"lat": mean_lat, "lon": mean_lon,
                                      "uncertainty_m": mean_unc},
                }

                with open(RESULTS_DIR / "latest.json", "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                print(f"  -> lat={map_lat:.6f}  lon={map_lon:.6f}  unc={map_unc:.0f}m")

                # Record for evaluation (GPS vs prediction)
                if tel and "lat" in tel:
                    eval_track.append({
                        "dist_m":   distance_m,
                        "pred_lat": map_lat,  "pred_lon": map_lon,
                        "gps_lat":  tel["lat"], "gps_lon": tel["lon"],
                        "obs":      bool(destinations),
                    })

                if destinations:
                    obs_count += 1
                    out = RESULTS_DIR / f"obs{obs_count:04d}_frame{frame_idx:04d}.json"
                    with open(out, "w") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"  Observation #{obs_count} saved: {out.name}")
                    try:
                        plot_particle_map(pf, G, city_svc, output_dir=RESULTS_DIR)
                    except Exception as e:
                        print(f"  Map warning: {e}")


                last_t_ns = timestamp
                frame_idx += 1

            signal.signal(signal.SIGINT, prev_handler)

    # ── Post-processing ──────────────────────────────────────────────────
    if sign_buffer and pf.initialised:
        print("\nFlushing remaining sign buffer.")
        remaining = _flush_sign_buffer(sign_buffer)
        if remaining:
            pf.predict(delta_distance_m=0.0, heading_deg=None)
            pf.update(remaining)

    if eval_track:
        # Save the full track to disk so plots can be regenerated later
        track_file = RESULTS_DIR / "eval_track.json"
        with open(track_file, "w") as f:
            json.dump(eval_track, f)
        print(f"\nEval track saved: {len(eval_track)} points")

        print("Generating evaluation plots ...")
        try:
            generate_evaluation(eval_track, G, RESULTS_DIR)
        except Exception as e:
            print(f"  Evaluation error: {e}")

    print(f"\n{'='*55}")
    print(f"Done. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()

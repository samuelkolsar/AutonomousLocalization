"""
app.py — Main application loop

Drop images into the `images/` folder while this script is running.
Each new image is automatically processed through OCR + particle filter.
For each image after the first one, you are asked in terminal how many
metres were travelled since the previous image.
Results (coordinates + maps) are written to `results/`.

Usage:
    python app.py

To reset the filter (start fresh):
    python app.py --reset
"""

import argparse
import json
import shutil
import time
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGES_DIR  = Path("images")
RESULTS_DIR = Path("results")
CACHE_DIR   = Path("cache")

for d in [IMAGES_DIR, RESULTS_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

PROCESSED_LOG = CACHE_DIR / "processed_images.json"
POLL_INTERVAL = 2.0   # seconds between folder scans

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ── Observation cleaning ────────────────────────────────────────────────────

def deduplicate_destinations_by_city(destinations: list[dict]) -> list[dict]:
    """
    When the same city appears multiple times (e.g. from two crops or two lines),
    keep the reading with highest OCR confidence per city.
    """
    if not destinations:
        return []
    # Sort by confidence descending; first occurrence of each city wins
    by_conf = sorted(destinations, key=lambda d: -d["ocr_confidence"])
    seen = set()
    out = []
    for d in by_conf:
        if d["city"] not in seen:
            seen.add(d["city"])
            out.append(d)
    if len(out) < len(destinations):
        print(f"  Deduplicated: {len(destinations)} → {len(out)} (by city)")
    return out


# ── Travel distance input ────────────────────────────────────────────────────

def prompt_travel_distance(image_path: Path) -> float:
    """Ask user for travelled distance since previous processed image."""
    prompt = f"  Distance since previous sign image before '{image_path.name}' (m): "
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            print("  Input unavailable; defaulting distance to 0 m.")
            return 0.0

        if not raw:
            print("  Please enter a number (e.g. 850).")
            continue

        try:
            distance_m = float(raw)
        except ValueError:
            print("  Invalid number. Try again.")
            continue

        if distance_m < 0.0:
            print("  Distance cannot be negative.")
            continue

        print(f"  Odometry: {distance_m:.0f} m since last image")
        return distance_m


def prompt_heading() -> float | None:
    """
    Ask user for compass bearing (0 = North, 90 = East, clockwise).
    Returns None if skipped — filter falls back to edge-geometry inference.
    """
    while True:
        try:
            raw = input("  Heading / compass bearing (0–360°, or Enter to skip): ").strip()
        except EOFError:
            return None

        if not raw:
            return None

        try:
            heading = float(raw)
        except ValueError:
            print("  Invalid number. Try again or press Enter to skip.")
            continue

        if not (0.0 <= heading < 360.0):
            print("  Bearing must be between 0 and 360.")
            continue

        print(f"  Heading: {heading:.0f}°")
        return heading


# ── Processed image tracking ─────────────────────────────────────────────────

def load_processed() -> set:
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            return set(json.load(f))
    return set()

def save_processed(processed: set) -> None:
    with open(PROCESSED_LOG, "w") as f:
        json.dump(list(processed), f)

def clear_directory_contents(path: Path) -> int:
    """Delete all files and subdirectories in a directory; return removed count."""
    removed = 0
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed += 1
    return removed


# ── Result saving ─────────────────────────────────────────────────────────────

_obs_count = 0   # module-level counter for observation saves


def save_result(image_name: str, observation: list[dict],
                map_lat: float, map_lon: float, map_uncertainty_m: float,
                mean_lat: float, mean_lon: float, mean_uncertainty_m: float,
                step: int) -> None:
    global _obs_count
    # MAP estimate (best particle) is primary — always on a road.
    # Weighted mean is secondary — can fall off-road between clusters.
    result = {
        "filter_step":   step,
        "timestamp":     datetime.now().isoformat(),
        "image":         image_name,
        "observation":   observation,
        "estimate": {
            "lat":           map_lat,
            "lon":           map_lon,
            "uncertainty_m": map_uncertainty_m,
        },
        "estimate_mean": {
            "lat":           mean_lat,
            "lon":           mean_lon,
            "uncertainty_m": mean_uncertainty_m,
        },
    }

    # Always update latest.json (current live position)
    with open(RESULTS_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Coordinates (MAP):  lat={map_lat:.6f}  lon={map_lon:.6f}  spread≈{map_uncertainty_m:.0f}m")
    print(f"  Coordinates (mean): lat={mean_lat:.6f}  lon={mean_lon:.6f}  uncertainty≈{mean_uncertainty_m:.0f}m")

    # Only save a numbered file when a sign observation was used
    if observation:
        _obs_count += 1
        out_path = RESULTS_DIR / f"obs{_obs_count:04d}_{Path(image_name).stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Observation #{_obs_count} saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Clear processed log and restart the filter")
    args = parser.parse_args()

    if args.reset:
        global _obs_count
        _obs_count = 0
        if PROCESSED_LOG.exists():
            PROCESSED_LOG.unlink()
        removed = clear_directory_contents(RESULTS_DIR)
        print(f"Filter reset. Processed image log cleared, results cleared ({removed} item(s)).")

    # Lazy imports — heavy models only load when first needed
    print("Loading OCR and localization modules ...")
    from ocr import process_image
    from localizer import load_graph, CityDistanceService, ParticleFilter, plot_particle_map
    import osmnx as ox

    print("Loading road graph ...")
    G            = load_graph()
    _, edges_gdf = ox.graph_to_gdfs(G)
    city_svc     = CityDistanceService(G)

    pf        = ParticleFilter(G, edges_gdf, city_svc)
    processed = load_processed()

    print(f"\n{'═'*55}")
    print(f"  Watching {IMAGES_DIR.resolve()} for new images ...")
    print(f"  Results  → {RESULTS_DIR.resolve()}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'═'*55}\n")

    try:
        while True:
            # Find new images not yet processed, sorted by filename (chronological)
            all_images = sorted(
                p for p in IMAGES_DIR.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            new_images = [p for p in all_images if p.name not in processed]

            for image_path in new_images:
                print(f"\n{'─'*55}")
                print(f"  New image: {image_path.name}")
                print(f"{'─'*55}")

                # Step 1 — OCR
                try:
                    destinations = process_image(str(image_path))
                except Exception as e:
                    print(f"  ERROR during OCR: {e}")
                    print("  Will retry this image on the next scan.")
                    continue

                destinations = deduplicate_destinations_by_city(destinations)

                if not destinations:
                    if not pf.initialised:
                        print("  No destinations found and filter not initialised — skipping.")
                        processed.add(image_path.name)
                        save_processed(processed)
                        continue

                    # Predict-only: advance particles without an observation update
                    print("  No sign detected — running predict-only step.")
                    distance_m = prompt_travel_distance(image_path)
                    heading    = prompt_heading()
                    try:
                        pf.predict(delta_distance_m=distance_m, heading_deg=heading)
                        pf.step_index += 1
                        pf._save_snapshot(f"predict_{pf.step_index}", [])
                    except Exception as e:
                        print(f"  ERROR during predict: {e}")
                        print("  Will retry this image on the next scan.")
                        continue

                    map_lat, map_lon, map_unc = pf.estimate_map()
                    mean_lat, mean_lon, mean_unc = pf.estimate()
                    save_result(image_path.name, [],
                                map_lat, map_lon, map_unc,
                                mean_lat, mean_lon, mean_unc, pf.step_index)
                    processed.add(image_path.name)
                    save_processed(processed)
                    continue

                # Step 2 — Travel distance + heading (only relevant after first observation)
                distance_m = 0.0
                heading    = None
                if pf.initialised:
                    distance_m = prompt_travel_distance(image_path)
                    heading    = prompt_heading()

                # Step 3 — Particle filter
                try:
                    if not pf.initialised:
                        pf.initialise(destinations)
                    else:
                        pf.predict(delta_distance_m=distance_m, heading_deg=heading)
                        pf.update(destinations)
                except Exception as e:
                    print(f"  ERROR during localisation: {e}")
                    print("  Will retry this image on the next scan.")
                    continue

                # If initialisation was rejected (bad observation), skip and retry.
                if not pf.initialised:
                    processed.add(image_path.name)
                    save_processed(processed)
                    continue

                # Step 4 — Save results
                map_lat, map_lon, map_unc = pf.estimate_map()
                mean_lat, mean_lon, mean_unc = pf.estimate()
                save_result(
                    image_path.name, destinations,
                    map_lat, map_lon, map_unc,
                    mean_lat, mean_lon, mean_unc,
                    pf.step_index
                )

                # Step 5 — Generate maps
                try:
                    plot_particle_map(pf, G, city_svc, output_dir=RESULTS_DIR)
                except Exception as e:
                    print(f"  WARNING: map generation failed: {e}")

                processed.add(image_path.name)
                save_processed(processed)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
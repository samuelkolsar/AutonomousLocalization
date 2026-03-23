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
    """
    Ask user for travelled distance since previous processed image.
    The value is used as odometry for the particle filter predict step.
    """
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

def save_result(image_name: str, observation: list[dict],
                lat: float, lon: float, uncertainty_m: float,
                map_lat: float, map_lon: float, map_uncertainty_m: float,
                step: int) -> None:
    result = {
        "step":          step,
        "timestamp":     datetime.now().isoformat(),
        "image":         image_name,
        "observation":   observation,
        "estimate": {
            "lat":           lat,
            "lon":           lon,
            "uncertainty_m": uncertainty_m,
        },
        "estimate_map": {
            "lat":           map_lat,
            "lon":           map_lon,
            "uncertainty_m": map_uncertainty_m,
        },
    }
    out_path = RESULTS_DIR / f"result_step{step:03d}_{Path(image_name).stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also overwrite a single "latest.json" for easy external reading
    with open(RESULTS_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Result saved → {out_path.name}")
    print(f"  Coordinates (mean): lat={lat:.6f}  lon={lon:.6f}  uncertainty≈{uncertainty_m:.0f}m")
    print(f"  Coordinates (MAP):  lat={map_lat:.6f}  lon={map_lon:.6f}  spread≈{map_uncertainty_m:.0f}m")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Clear processed log and restart the filter")
    args = parser.parse_args()

    if args.reset:
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
    G               = load_graph()
    G_undirected    = G.to_undirected()
    _, edges_gdf    = ox.graph_to_gdfs(G)
    city_svc        = CityDistanceService(G)

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
                    print("  No destinations found — skipping.")
                    processed.add(image_path.name)
                    save_processed(processed)
                    continue

                # Step 2 — Travel distance (only relevant after first observation)
                distance_m = 0.0
                if pf.initialised:
                    distance_m = prompt_travel_distance(image_path)

                # Step 3 — Particle filter
                try:
                    if not pf.initialised:
                        pf.initialise(destinations)
                    else:
                        pf.predict(delta_distance_m=distance_m)
                        pf.update(destinations)
                except Exception as e:
                    print(f"  ERROR during localisation: {e}")
                    print("  Will retry this image on the next scan.")
                    continue

                # Step 4 — Save results
                lat, lon, unc = pf.estimate()
                map_lat, map_lon, map_unc = pf.estimate_map()
                save_result(
                    image_path.name, destinations,
                    lat, lon, unc,
                    map_lat, map_lon, map_unc,
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
"""
app.py — Main application loop

Drop images into the `images/` folder while this script is running.
Each new image is automatically processed through OCR + particle filter.
Results (coordinates + maps) are written to `results/`.

Usage:
    python app.py

To reset the filter (start fresh):
    python app.py --reset
"""

import argparse
import json
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


# ── Processed image tracking ─────────────────────────────────────────────────

def load_processed() -> set:
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            return set(json.load(f))
    return set()

def save_processed(processed: set) -> None:
    with open(PROCESSED_LOG, "w") as f:
        json.dump(list(processed), f)


# ── Result saving ─────────────────────────────────────────────────────────────

def save_result(image_name: str, observation: list[dict],
                lat: float, lon: float, uncertainty_m: float,
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
        }
    }
    out_path = RESULTS_DIR / f"result_step{step:03d}_{Path(image_name).stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also overwrite a single "latest.json" for easy external reading
    with open(RESULTS_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Result saved → {out_path.name}")
    print(f"  Coordinates: lat={lat:.6f}  lon={lon:.6f}  uncertainty≈{uncertainty_m:.0f}m")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Clear processed log and restart the filter")
    args = parser.parse_args()

    if args.reset:
        if PROCESSED_LOG.exists():
            PROCESSED_LOG.unlink()
        print("Filter reset. Processed image log cleared.")

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
                    processed.add(image_path.name)
                    save_processed(processed)
                    continue

                if not destinations:
                    print("  No destinations found — skipping.")
                    processed.add(image_path.name)
                    save_processed(processed)
                    continue

                # Step 2 — Particle filter
                try:
                    if not pf.initialised:
                        pf.initialise(destinations)
                    else:
                        pf.predict(delta_distance_m=0)   # odometry hook
                        pf.update(destinations)
                except Exception as e:
                    print(f"  ERROR during localisation: {e}")
                    processed.add(image_path.name)
                    save_processed(processed)
                    continue

                # Step 3 — Save results
                lat, lon, unc = pf.estimate()
                save_result(image_path.name, destinations, lat, lon, unc, pf.step_index)

                # Step 4 — Generate maps
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

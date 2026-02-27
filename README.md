# Road Sign Localization — Pipeline Application

## Structure

```
pipeline/
├── app.py          ← main loop, run this
├── ocr.py          ← YOLO detection + EasyOCR + parsing
├── localizer.py    ← particle filter + visualisation
├── images/         ← DROP YOUR IMAGES HERE
├── results/        ← coordinates JSON + map PNGs written here
└── cache/          ← road graph + Dijkstra cache (auto-managed)
```

## Setup

```bash
pip install ultralytics easyocr rapidfuzz osmnx networkx geopandas shapely scipy matplotlib tqdm
```

Make sure `minu_mudel_best.pt` (your trained YOLO model) is in the same folder as `app.py`.

## Running

```bash
python app.py
```

Then drop images into the `images/` folder. Each new image is automatically:
1. Detected + OCR'd for city/distance pairs
2. Fed into the particle filter
3. Saved as `results/result_stepNNN_<image>.json` + two map PNGs

The filter maintains state across images — the more images you feed in,
the more the particle cloud converges.

## Resetting the filter

```bash
python app.py --reset
```

Clears the processed image log so everything starts fresh.

## Output

Every processed image produces:
- `results/result_stepNNN_<name>.json` — coordinates + observation + uncertainty
- `results/latest.json` — always the most recent result (easy to poll externally)
- `results/map_full_stepN.png` — full Estonia road map with particle cloud
- `results/map_zoomed_stepN.png` — zoomed view around the particle cluster

## Adding odometry

In `localizer.py`, find the `predict()` method in `ParticleFilter`.
The stub is there with instructions. When you have odometry data,
pass it in `app.py` here:

```python
pf.predict(delta_distance_m=500, delta_heading_deg=0)
```

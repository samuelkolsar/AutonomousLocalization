# Autonomous Localisation via Road Sign OCR

GPS-free vehicle localisation on the Estonian road network. A particle filter
estimates the vehicle's position by matching road sign distance readings
(e.g. "TARTU 82 km") against precomputed Dijkstra distances from city centres
to every node in the OSM road graph. Odometry and compass heading from a
NovAtel INSPVA unit propagate the particles between sign observations.

## Pipeline

```
ROS1 .bag file
      │
      ├── Camera frames ──► YOLO detection ──► EasyOCR ──► city + distance pairs
      │                                                              │
      └── INSPVA telemetry ──► speed + heading                       │
                                    │                                │
                                    ▼                                ▼
                              Predict step                     Update step
                         (move particles along           (reweight particles by
                          road network)                   distance likelihood)
                                    │
                                    ▼
                          results/latest.json + map PNGs
```

## Files

| File | Purpose |
|---|---|
| `bag_replay.py` | Main script — replays a ROS1 bag through the pipeline |
| `ocr.py` | YOLO sign detection + EasyOCR + city/distance parsing |
| `localizer.py` | Particle filter, city distance service, visualisation |
| `place_names.py` | List of Estonian place names for OCR fuzzy correction |

## Setup

```bash
pip install -r requirements.txt
```

Place your trained YOLO weights (`sign_detector.pt`) in the project root.

The Estonia road graph is downloaded from OpenStreetMap automatically on first
run and cached to `cache/estonia_drive.pkl` (~5–15 minutes, one-time).
City Dijkstra distances are also cached per city in `cache/`.

## Running

```bash
python bag_replay.py path/to/recording.bag
```

Optional arguments:

```bash
python bag_replay.py recording.bag --reset           # clear previous results first
python bag_replay.py recording.bag --camera /topic   # specify camera topic
```

Press **Ctrl+C** at any time to stop — evaluation plots are generated from
whatever data was collected up to that point.

## Output

All output is written to `results/`:

| File | Description |
|---|---|
| `latest.json` | Most recent position estimate (updated every step) |
| `latest_map.png` | Live zoomed particle map (updated every step) |
| `obs####_frame####.json` | Full result at each sign observation |
| `map_full_stepN.png` | Full Estonia map with particle cloud |
| `map_zoomed_stepN.png` | Zoomed view around the particle cluster |
| `error_plot.png` | Position error vs distance travelled (vs GPS ground truth) |
| `gps_map.png` | GPS track vs particle filter track overlay |

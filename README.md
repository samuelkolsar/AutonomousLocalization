# Autonomous Localization via Road Sign OCR

GPS-free vehicle localization on the Estonian road network. Uses a particle-filter-inspired localization filter to estimate position by matching road sign distances (e.g., "TARTU 82 km") against precomputed road distances from city centers to map nodes. Vehicle motion (odometry and heading) moves particles between sign readings.

## Simple Workflow

The system runs in a loop:
1. **Read a sign observation**: OCR extracts city names and distances from camera images.
2. **Move guesses forward**: Propagate particles along roads using traveled distance and compass heading.
3. **Score guesses**: Check how well each particle's road distance to the cities matches the sign distances.
4. **Keep the best guesses**: Resample particles to focus on high-scoring ones, adding small random shifts for variety.
5. **Estimate position**: Compute the most likely location from the particle weights.

Particles are guesses about the vehicle's position on road edges. Each guess includes latitude, longitude, edge ID, and position along the edge.

## Pipeline Overview

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
| `bag_replay.py` | Main script, replays a ROS1 bag through the pipeline |
| `ocr.py` | YOLO sign detection + EasyOCR + city/distance parsing |
| `localizer.py` | Localization filter (`ParticleFilter` class), city distance service, visualisation |
| `place_names.py` | List of Estonian place names for OCR fuzzy correction |

## Setup

```bash
pip install -r requirements.txt
```

Model weights are included in the `models/` folder (`sign_detector_v11.pt` for YOLOv11, `sign_detector_v8.pt` for YOLOv8).

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

Press **Ctrl+C** at any time to stop, evaluation plots are generated from
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

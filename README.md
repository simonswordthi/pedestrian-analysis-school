# pedestrian-analysis-school

> **Pedestrian Crossing Trajectory Analysis Tool** – drone BEV video → trajectories → behavior labels → swarm metrics → paper-ready Plotly figures, all wrapped in a Gradio UI.

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/simonswordthi/pedestrian-analysis-school.git
cd pedestrian-analysis-school
pip install -r requirements.txt
```

### 2. Create the project structure

Run the bootstrap script once to generate all source files:

```bash
python init_project.py
```

This creates the `pedestrian_analysis/` package with the full directory tree shown below.

### 3. Launch the app

```bash
cd pedestrian_analysis
python app.py
# → Open http://localhost:7860
```

---

## Project Structure

```
pedestrian_analysis/
├── app.py                        # Gradio UI – entry point  (python app.py)
├── config.py                     # Global thresholds & plot settings
├── pipeline/
│   ├── calibration.py            # Homography: pixel → metre
│   ├── tracker.py                # YOLOv8 + ByteTrack → trajectory DataFrame
│   ├── trajectory_io.py          # Load / save CSV trajectories
│   ├── pedpy_analysis.py         # PedPy wrapper: speed & heading
│   ├── behavior_labeling.py      # FSM: waiting / approaching / crossing / crossed
│   ├── group_analysis.py         # DBSCAN group detection + temporal smoothing
│   └── swarm_analysis.py         # Reynolds rules: cohesion, alignment, separation
├── visualization/
│   ├── plot_trajectories.py      # Plotly trajectory lines + street overlay
│   ├── plot_heatmap.py           # 2-D density heatmap
│   ├── plot_behavior.py          # Gantt-style behavior timeline
│   ├── plot_swarm.py             # Cohesion / alignment / separation subplots
│   └── export.py                 # PNG / PDF export + ZIP bundle
├── data/
│   ├── videos/                   # Raw videos (git-ignored)
│   ├── trajectories/             # Extracted .csv trajectories
│   └── calibration/              # Homography matrices (.npy)
├── outputs/figures/              # Exported paper plots
└── tests/test_pipeline.py        # Unit tests (pytest)
```

---

## Workflow

| Step | Tab | What happens |
|------|-----|--------------|
| 1 | 📷 Kalibrierung | Upload a video frame, mark 4 pixel/metre point pairs, download `homography.npy` |
| 2 | 🎬 Trajektorienextraktion | Upload video + `.npy`, run YOLOv8+ByteTrack, full pipeline runs, download CSV |
| 3 | 📊 Analyse & Plots | Upload CSV, explore 6 sub-tabs, export all figures as ZIP |

---

## Running tests

```bash
cd pedestrian_analysis
pytest tests/ -v
```

---

## Key dependencies

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 person detection |
| `supervision` | ByteTrack multi-object tracking |
| `pedpy` | Pedestrian kinematics (speed, density) |
| `scikit-learn` | DBSCAN group detection |
| `scipy` | Circular statistics for heading angles |
| `plotly` | Interactive & paper-ready figures |
| `gradio` | Web UI |
| `kaleido` | Plotly → PNG/PDF export |
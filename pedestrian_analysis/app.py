"""Gradio Blocks UI – entry point for the Pedestrian Analysis tool.

Run with:
    python app.py
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import numpy as np
import pandas as pd

# Ensure the package root is on sys.path when running directly
import sys
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import pedestrian_analysis.config as cfg
from pedestrian_analysis.pipeline.calibration import (
    compute_homography,
    load_calibration,
    pixel_to_meter,
    save_calibration,
    validate_calibration,
)
from pedestrian_analysis.pipeline.tracker import extract_trajectories_from_video
from pedestrian_analysis.pipeline.trajectory_io import load_trajectories, save_trajectories
from pedestrian_analysis.pipeline.pedpy_analysis import compute_kinematics
from pedestrian_analysis.pipeline.behavior_labeling import (
    compute_crossing_events,
    label_behaviors,
)
from pedestrian_analysis.pipeline.group_analysis import (
    compute_group_statistics,
    detect_groups_per_frame,
)
from pedestrian_analysis.pipeline.swarm_analysis import (
    compute_cohesion,
    compute_alignment,
    compute_separation,
    compute_leader_follower,
)
from pedestrian_analysis.visualization.plot_trajectories import plot_trajectories
from pedestrian_analysis.visualization.plot_heatmap import plot_density_heatmap
from pedestrian_analysis.visualization.plot_behavior import plot_behavior_timeline
from pedestrian_analysis.visualization.plot_swarm import plot_swarm_metrics
from pedestrian_analysis.visualization.export import export_all_figures

import cv2


# ---------------------------------------------------------------------------
# Tab 1 helpers
# ---------------------------------------------------------------------------

def _run_calibration(
    frame_img: np.ndarray | None,
    pixel_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> tuple[np.ndarray | None, str | None]:
    """Compute homography and return (BEV image, calibration .npy path)."""
    if frame_img is None:
        raise gr.Error("Please upload a video frame image.")

    try:
        src = pixel_df[["px", "py"]].values.astype(np.float32)
        dst = meter_df[["x_m", "y_m"]].values.astype(np.float32)
        H = compute_homography(src, dst)
    except Exception as exc:
        raise gr.Error(f"Calibration failed: {exc}") from exc

    bev = validate_calibration(frame_img, H, grid_spacing_m=grid_spacing)

    npy_path = os.path.join(tempfile.mkdtemp(), "homography.npy")
    save_calibration(H, npy_path)
    return bev, npy_path


# ---------------------------------------------------------------------------
# Tab 2 helpers
# ---------------------------------------------------------------------------

def _run_analysis(
    video_file: str | None,
    calib_file: str | None,
    confidence: float,
    frame_skip: int,
    fps: float,
    street_start: float,
    street_end: float,
    speed_threshold: float,
    group_radius: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    """Full pipeline: track → kinematics → behavior → groups → swarm."""
    if video_file is None:
        raise gr.Error("Please upload a video file.")
    if calib_file is None:
        raise gr.Error("Please upload a calibration .npy file.")

    try:
        H = load_calibration(calib_file)
    except Exception as exc:
        raise gr.Error(f"Cannot load calibration: {exc}") from exc

    progress(0, desc="Extracting trajectories…")

    def _prog_cb(current: int, total: int) -> None:
        progress(current / max(total, 1), desc=f"Tracking frame {current}/{total}")

    try:
        df = extract_trajectories_from_video(
            video_path=video_file,
            H=H,
            confidence_threshold=confidence,
            frame_skip=int(frame_skip),
            progress_callback=_prog_cb,
        )
    except Exception as exc:
        raise gr.Error(f"Tracking failed: {exc}") from exc

    progress(0.6, desc="Computing kinematics…")
    df = compute_kinematics(df, fps=fps)

    progress(0.7, desc="Labeling behaviors…")
    df = label_behaviors(
        df,
        street_start_m=street_start,
        street_end_m=street_end,
        speed_threshold_ms=speed_threshold,
    )

    progress(0.8, desc="Detecting groups…")
    df = detect_groups_per_frame(df, proximity_threshold_m=group_radius)

    progress(0.9, desc="Swarm analysis…")
    df = compute_cohesion(df)
    df = compute_alignment(df)
    df = compute_separation(df)

    csv_path = os.path.join(tempfile.mkdtemp(), "trajectories.csv")
    save_trajectories(df, csv_path)

    progress(1.0, desc="Done.")
    return csv_path, csv_path


# ---------------------------------------------------------------------------
# Tab 3 helpers
# ---------------------------------------------------------------------------

def _load_and_plot_trajectories(
    csv_file: str | None,
    street_start: float,
    street_end: float,
) -> go.Figure | None:
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    return plot_trajectories(df, street_start, street_end)


def _load_and_plot_speed(csv_file: str | None, fps: float):
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    if "speed_ms" not in df.columns:
        raise gr.Error("CSV does not contain 'speed_ms'. Run analysis first.")
    import plotly.express as px
    df["time_s"] = df["frame"] / fps
    fig = px.line(
        df, x="time_s", y="speed_ms", color="id",
        labels={"time_s": "Time (s)", "speed_ms": "Speed (m/s)"},
        template=cfg.PLOT_TEMPLATE,
    )
    fig.update_layout(
        font=dict(family=cfg.PLOT_FONT_FAMILY, size=cfg.PLOT_FONT_SIZE),
        width=cfg.PLOT_WIDTH_PX, height=cfg.PLOT_HEIGHT_PX,
    )
    return fig


def _load_and_plot_behavior(csv_file: str | None, fps: float):
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    if "behavior" not in df.columns:
        raise gr.Error("CSV does not contain 'behavior'. Run analysis first.")
    return plot_behavior_timeline(df, fps)


def _load_and_plot_groups(csv_file: str | None):
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    if "group_id" not in df.columns:
        raise gr.Error("CSV does not contain 'group_id'. Run analysis first.")
    stats = compute_group_statistics(df)
    sizes = stats["size_histogram"]
    import plotly.graph_objects as go
    fig = go.Figure(
        go.Bar(
            x=list(sizes.keys()),
            y=list(sizes.values()),
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        template=cfg.PLOT_TEMPLATE,
        xaxis_title="Group Size (persons)",
        yaxis_title="Count",
        font=dict(family=cfg.PLOT_FONT_FAMILY, size=cfg.PLOT_FONT_SIZE),
        width=cfg.PLOT_WIDTH_PX, height=cfg.PLOT_HEIGHT_PX,
    )
    return fig


def _load_and_plot_swarm(csv_file: str | None, fps: float):
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    for col in ["cohesion_m", "alignment_deg", "min_separation_m"]:
        if col not in df.columns:
            raise gr.Error(f"CSV does not contain '{col}'. Run analysis first.")
    return plot_swarm_metrics(df, fps)


def _compute_statistics(csv_file: str | None, fps: float) -> pd.DataFrame:
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    rows: list[dict] = []

    rows.append({"Metric": "Total pedestrians", "Value": df["id"].nunique()})
    rows.append({"Metric": "Total frames", "Value": df["frame"].nunique()})
    rows.append({"Metric": "Duration (s)", "Value": round(df["frame"].max() / fps, 2)})

    if "speed_ms" in df.columns:
        rows.append({"Metric": "Mean speed (m/s)", "Value": round(df["speed_ms"].mean(), 3)})
        rows.append({"Metric": "Std speed (m/s)", "Value": round(df["speed_ms"].std(), 3)})

    if "group_id" in df.columns:
        stats = compute_group_statistics(df)
        rows.append({"Metric": "Number of groups", "Value": stats["group_count"]})
        rows.append({"Metric": "Partial crossing rate", "Value": round(stats["partial_crossing_rate"], 3)})
        rows.append({"Metric": "Median waiting time (s)", "Value": round(stats["median_waiting_time_s"], 2)})

    if "behavior" in df.columns and "crossing_start_frame" in df.columns:
        events = compute_crossing_events(df, fps)
        if not events.empty:
            rows.append({"Metric": "Crossing events", "Value": len(events)})
            rows.append({"Metric": "Mean crossing duration (s)", "Value": round(events["crossing_duration_s"].mean(), 2)})

    return pd.DataFrame(rows)


def _export_all(
    csv_file: str | None,
    fps: float,
    street_start: float,
    street_end: float,
) -> str | None:
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    figs: dict = {}
    try:
        figs["trajectories"] = plot_trajectories(df, street_start, street_end)
    except Exception:
        pass
    try:
        figs["heatmap"] = plot_density_heatmap(df)
    except Exception:
        pass
    if "behavior" in df.columns:
        try:
            figs["behavior"] = plot_behavior_timeline(df, fps)
        except Exception:
            pass
    if all(c in df.columns for c in ["cohesion_m", "alignment_deg", "min_separation_m"]):
        try:
            figs["swarm"] = plot_swarm_metrics(df, fps)
        except Exception:
            pass

    out_dir = os.path.join(tempfile.mkdtemp(), "figures")
    zip_path = export_all_figures(figs, out_dir)
    return zip_path


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

import plotly.graph_objects as go  # noqa: E402

_DEFAULT_PIXEL_PTS = pd.DataFrame(
    {"px": [100.0, 500.0, 500.0, 100.0], "py": [100.0, 100.0, 400.0, 400.0]}
)
_DEFAULT_METER_PTS = pd.DataFrame(
    {"x_m": [0.0, 4.0, 4.0, 0.0], "y_m": [0.0, 0.0, 3.0, 3.0]}
)

with gr.Blocks(title="Pedestrian Crossing Analysis", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🚶 Pedestrian Crossing Trajectory Analysis")
    gr.Markdown(
        "Analyse drone-captured BEV footage of pedestrian crossing scenarios. "
        "Three-step pipeline: calibration → tracking → analysis."
    )

    # ── Tab 1: Calibration ──────────────────────────────────────────────────
    with gr.Tab("📷 Kalibrierung"):
        gr.Markdown("### Camera → Ground-plane Homography Calibration")
        with gr.Row():
            with gr.Column():
                calib_frame = gr.Image(label="Video Frame (upload as image)", type="numpy")
                pixel_pts_df = gr.Dataframe(
                    value=_DEFAULT_PIXEL_PTS,
                    label="4 Pixel Points (px, py)",
                    col_count=(2, "fixed"),
                    interactive=True,
                )
                meter_pts_df = gr.Dataframe(
                    value=_DEFAULT_METER_PTS,
                    label="4 Meter Points (x_m, y_m)",
                    col_count=(2, "fixed"),
                    interactive=True,
                )
                grid_slider = gr.Slider(
                    minimum=0.5, maximum=5.0, step=0.5, value=1.0,
                    label="Grid spacing (m)",
                )
                calib_btn = gr.Button("Kalibrierung berechnen", variant="primary")
            with gr.Column():
                bev_output = gr.Image(label="BEV Validation Grid")
                calib_file_out = gr.File(label="Download Homography (.npy)")

        calib_btn.click(
            fn=_run_calibration,
            inputs=[calib_frame, pixel_pts_df, meter_pts_df, grid_slider],
            outputs=[bev_output, calib_file_out],
        )

    # ── Tab 2: Extraction ───────────────────────────────────────────────────
    with gr.Tab("🎬 Trajektorienextraktion"):
        gr.Markdown("### Video Tracking & Full Pipeline")
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Upload Video (.mp4/.mov/.avi)", file_types=[".mp4", ".mov", ".avi"])
                calib_input = gr.File(label="Upload Calibration (.npy)", file_types=[".npy"])
                with gr.Row():
                    conf_slider = gr.Slider(0.1, 0.9, step=0.05, value=0.4, label="Confidence Threshold")
                    skip_slider = gr.Slider(1, 10, step=1, value=1, label="Frame Skip")
                with gr.Row():
                    fps_num = gr.Number(value=25.0, label="FPS")
                    speed_thr_num = gr.Number(value=cfg.SPEED_THRESHOLD_MS, label="Speed threshold (m/s)")
                with gr.Row():
                    street_start_num2 = gr.Number(value=2.0, label="Street start X (m)")
                    street_end_num2 = gr.Number(value=6.0, label="Street end X (m)")
                group_radius_num = gr.Number(value=cfg.PROXIMITY_THRESHOLD_M, label="DBSCAN group radius (m)")
                run_btn = gr.Button("▶ Analyse starten", variant="primary")
            with gr.Column():
                csv_out_file = gr.File(label="Download Trajectories CSV")
                status_text = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=_run_analysis,
            inputs=[
                video_input, calib_input, conf_slider, skip_slider,
                fps_num, street_start_num2, street_end_num2,
                speed_thr_num, group_radius_num,
            ],
            outputs=[csv_out_file, status_text],
        )

    # ── Tab 3: Analysis & Plots ──────────────────────────────────────────────
    with gr.Tab("📊 Analyse & Plots"):
        gr.Markdown("### Load a trajectories CSV to explore results")
        with gr.Row():
            csv_upload = gr.File(label="Upload Trajectories CSV", file_types=[".csv"])
            with gr.Column():
                fps_num3 = gr.Number(value=25.0, label="FPS")
                street_start_num3 = gr.Number(value=2.0, label="Street start X (m)")
                street_end_num3 = gr.Number(value=6.0, label="Street end X (m)")

        with gr.Tabs():
            with gr.Tab("Trajektorien"):
                traj_btn = gr.Button("Plot Trajectories")
                traj_plot = gr.Plot()
                traj_btn.click(
                    fn=_load_and_plot_trajectories,
                    inputs=[csv_upload, street_start_num3, street_end_num3],
                    outputs=traj_plot,
                )

            with gr.Tab("Geschwindigkeit"):
                speed_btn = gr.Button("Plot Speed")
                speed_plot = gr.Plot()
                speed_btn.click(
                    fn=_load_and_plot_speed,
                    inputs=[csv_upload, fps_num3],
                    outputs=speed_plot,
                )

            with gr.Tab("Behavior Labels"):
                beh_btn = gr.Button("Plot Behavior Timeline")
                beh_plot = gr.Plot()
                beh_btn.click(
                    fn=_load_and_plot_behavior,
                    inputs=[csv_upload, fps_num3],
                    outputs=beh_plot,
                )

            with gr.Tab("Gruppen"):
                grp_btn = gr.Button("Plot Group Sizes")
                grp_plot = gr.Plot()
                grp_btn.click(
                    fn=_load_and_plot_groups,
                    inputs=[csv_upload],
                    outputs=grp_plot,
                )

            with gr.Tab("Schwarmverhalten"):
                swarm_btn = gr.Button("Plot Swarm Metrics")
                swarm_plot = gr.Plot()
                swarm_btn.click(
                    fn=_load_and_plot_swarm,
                    inputs=[csv_upload, fps_num3],
                    outputs=swarm_plot,
                )

            with gr.Tab("Statistiken"):
                stats_btn = gr.Button("Compute Statistics")
                stats_table = gr.DataFrame()
                stats_btn.click(
                    fn=_compute_statistics,
                    inputs=[csv_upload, fps_num3],
                    outputs=stats_table,
                )

        with gr.Row():
            export_btn = gr.Button("📥 Alle Plots exportieren (PNG)")
            export_zip = gr.File(label="Download ZIP")
        export_btn.click(
            fn=_export_all,
            inputs=[csv_upload, fps_num3, street_start_num3, street_end_num3],
            outputs=export_zip,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

"""Gradio Blocks UI – entry point for the Pedestrian Analysis tool.

Run with:
    python app.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime

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


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")
_CALIBRATION_DIR = os.path.join(_DATA_DIR, "calibration")
_TRAJECTORY_DIR = os.path.join(_DATA_DIR, "trajectories")
_VIDEO_DIR = os.path.join(_DATA_DIR, "videos")
_UPLOAD_DIR = os.path.join(_DATA_DIR, "uploads")
_CALIBRATION_STATE_FILE = os.path.join(_CALIBRATION_DIR, "calibration_state.json")
_CALIBRATION_IMAGE_FILE = os.path.join(_CALIBRATION_DIR, "current_calibration_image.png")


def _ensure_project_dirs() -> None:
    for path in (_CALIBRATION_DIR, _TRAJECTORY_DIR, _VIDEO_DIR, _UPLOAD_DIR):
        os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _as_path(value: str | os.PathLike[str] | None) -> str | None:
    return None if value is None else os.fspath(value)


def _copy_uploaded_file(src_path: str | os.PathLike[str], dest_dir: str, prefix: str) -> str:
    _ensure_project_dirs()
    source = os.fspath(src_path)
    _, ext = os.path.splitext(source)
    dest_path = os.path.join(dest_dir, f"{prefix}_{_timestamp()}{ext or '.bin'}")
    shutil.copy2(source, dest_path)
    return dest_path


def _empty_pixel_points_df() -> pd.DataFrame:
    return pd.DataFrame({"px": [np.nan] * 4, "py": [np.nan] * 4})


def _points_to_dataframe(points: list[list[float]]) -> pd.DataFrame:
    rows = points[:4] + [[np.nan, np.nan]] * max(0, 4 - len(points))
    return pd.DataFrame(rows, columns=["px", "py"])


def _parse_click_state(points_json: str | None) -> list[list[float]]:
    if not points_json:
        return []
    try:
        points = json.loads(points_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(points, list):
        return []
    cleaned: list[list[float]] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                cleaned.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue
    return cleaned[:4]


def _image_to_data_url(image_path: str | None) -> str:
    if image_path is None or not os.path.isfile(image_path):
        return ""
    import base64

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    _, ext = os.path.splitext(image_path)
    mime = "image/png" if ext.lower() in {"", ".png"} else f"image/{ext.lower().lstrip('.')}"
    return f"data:{mime};base64,{encoded}"


def _save_calibration_image(source_path: str | None) -> str | None:
    source = _as_path(source_path)
    if source is None or not os.path.isfile(source):
        return None
    image = cv2.imread(source)
    if image is None:
        return None
    _ensure_project_dirs()
    cv2.imwrite(_CALIBRATION_IMAGE_FILE, image)
    return _CALIBRATION_IMAGE_FILE


def _load_calibration_state() -> dict:
    default_state = {
        "image_path": None,
        "pixel_points": [],
        "meter_points": _DEFAULT_METER_PTS.to_dict(orient="records"),
        "grid_spacing_m": 1.0,
    }
    if not os.path.isfile(_CALIBRATION_STATE_FILE):
        return default_state
    try:
        with open(_CALIBRATION_STATE_FILE, "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except Exception:
        return default_state
    if not isinstance(state, dict):
        return default_state
    for key, value in default_state.items():
        state.setdefault(key, value)
    return state


def _persist_calibration_state(
    image_path: str | None,
    pixel_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> None:
    _ensure_project_dirs()
    payload = {
        "image_path": image_path,
        "pixel_points": pixel_df[["px", "py"]].to_dict(orient="records"),
        "meter_points": meter_df[["x_m", "y_m"]].to_dict(orient="records"),
        "grid_spacing_m": float(grid_spacing),
        "updated_at": _timestamp(),
    }
    with open(_CALIBRATION_STATE_FILE, "w", encoding="utf-8") as state_file:
        json.dump(payload, state_file, ensure_ascii=False, indent=2)


def _build_calibration_preview_html(image_path: str | None, points: list[list[float]] | None = None) -> str:
    image_url = _image_to_data_url(image_path)
    points_json = json.dumps(points or [], ensure_ascii=False)
    if image_url:
        image_block = (
            '<canvas id="calib-preview-canvas" style="width:100%;max-width:100%;border:1px solid #d0d7de;'
            'border-radius:12px;display:block;background:#111;"></canvas>'
        )
        status_text = "Klicke nacheinander auf 4 Referenzpunkte."
    else:
        image_block = (
            '<div style="height:420px;display:flex;align-items:center;justify-content:center;'
            'border:1px dashed #999;border-radius:12px;color:#666;background:#fafafa;">'
            'Noch kein Bild geladen.</div>'
        )
        status_text = "Lade ein Bild in die Vorschau."
    return (
        f'<div id="calib-preview-root" data-image-url="{image_url}" data-points="{points_json}">'
        f'<div style="margin-bottom:0.5rem;font-weight:600;">Kalibrierungs-Vorschau</div>'
        f'<div style="margin-bottom:0.75rem;color:#666;font-size:0.92rem;">{status_text}</div>'
        f'{image_block}'
        f'</div>'
    )


def _load_calibration_ui_state() -> tuple[str, pd.DataFrame, pd.DataFrame, str, str, str]:
    state = _load_calibration_state()
    image_path = state.get("image_path")
    pixel_points = state.get("pixel_points", [])
    meter_points = state.get("meter_points", _DEFAULT_METER_PTS.to_dict(orient="records"))
    pixel_df = _points_to_dataframe(pixel_points) if pixel_points else _empty_pixel_points_df()
    meter_df = pd.DataFrame(meter_points)
    if list(meter_df.columns) != ["x_m", "y_m"]:
        meter_df = _DEFAULT_METER_PTS.copy()
    preview_html = _build_calibration_preview_html(image_path, pixel_points)
    points_json = json.dumps(pixel_points, ensure_ascii=False)
    status = f"Gespeichert: {len(pixel_points)}/4 Referenzpunkte" if pixel_points else "Noch keine Referenzpunkte gespeichert."
    return preview_html, pixel_df, meter_df, points_json, str(image_path or ""), status


CALIBRATION_PREVIEW_HEAD = r"""
<script>
(function () {
  function parsePoints(value) {
    try { return value ? JSON.parse(value) : []; } catch (error) { return []; }
  }

  function drawPreview(root) {
    const canvas = root.querySelector('#calib-preview-canvas');
    const imageUrl = root.dataset.imageUrl || '';
    if (!canvas || !imageUrl) return;

    const points = parsePoints(root.dataset.points);
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      points.forEach((point, index) => {
        const x = point[0];
        const y = point[1];
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#ff3b30';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 18px sans-serif';
        ctx.fillText(String(index + 1), x + 12, y - 12);
      });
    };
    img.src = imageUrl;

    if (canvas.dataset.bound === '1') return;
    canvas.dataset.bound = '1';
    canvas.style.cursor = 'crosshair';
    canvas.addEventListener('click', function (event) {
      const root = document.getElementById('calib-preview-root');
      if (!root) return;
      const currentPoints = parsePoints(root.dataset.points);
      if (currentPoints.length >= 4) return;

      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = Math.round((event.clientX - rect.left) * scaleX);
      const y = Math.round((event.clientY - rect.top) * scaleY);
      currentPoints.push([x, y]);
      root.dataset.points = JSON.stringify(currentPoints);

      const pointsInput = document.querySelector('#calib-points-state textarea, #calib-points-state input');
      if (pointsInput) {
        pointsInput.value = JSON.stringify(currentPoints);
        pointsInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      drawPreview(root);
    });
  }

  function bind() {
    const root = document.getElementById('calib-preview-root');
    if (root) drawPreview(root);
  }

  setInterval(bind, 250);
  document.addEventListener('DOMContentLoaded', bind);
  bind();
})();
</script>
"""


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")
_CALIBRATION_DIR = os.path.join(_DATA_DIR, "calibration")
_TRAJECTORY_DIR = os.path.join(_DATA_DIR, "trajectories")
_VIDEO_DIR = os.path.join(_DATA_DIR, "videos")
_UPLOAD_DIR = os.path.join(_DATA_DIR, "uploads")
_CALIBRATION_STATE_FILE = os.path.join(_CALIBRATION_DIR, "calibration_state.json")
_CALIBRATION_IMAGE_FILE = os.path.join(_CALIBRATION_DIR, "current_calibration_image.png")


def _ensure_project_dirs() -> None:
    for path in (_CALIBRATION_DIR, _TRAJECTORY_DIR, _VIDEO_DIR, _UPLOAD_DIR):
        os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _as_path(value: str | os.PathLike[str] | None) -> str | None:
    return None if value is None else os.fspath(value)


def _copy_uploaded_file(src_path: str | os.PathLike[str], dest_dir: str, prefix: str) -> str:
    _ensure_project_dirs()
    source = os.fspath(src_path)
    _, ext = os.path.splitext(source)
    dest_path = os.path.join(dest_dir, f"{prefix}_{_timestamp()}{ext or '.bin'}")
    shutil.copy2(source, dest_path)
    return dest_path


def _empty_pixel_points_df() -> pd.DataFrame:
    return pd.DataFrame({"px": [np.nan] * 4, "py": [np.nan] * 4})


def _points_to_dataframe(points: list[list[float]]) -> pd.DataFrame:
    rows = points[:4] + [[np.nan, np.nan]] * max(0, 4 - len(points))
    return pd.DataFrame(rows, columns=["px", "py"])


def _parse_click_state(points_json: str | None) -> list[list[float]]:
    if not points_json:
        return []
    try:
        points = json.loads(points_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(points, list):
        return []
    cleaned: list[list[float]] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                cleaned.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue
    return cleaned[:4]


def _image_to_data_url(image_path: str | None) -> str:
    if image_path is None or not os.path.isfile(image_path):
        return ""
    import base64

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    _, ext = os.path.splitext(image_path)
    mime = "image/png" if ext.lower() in {"", ".png"} else f"image/{ext.lower().lstrip('.') }"
    return f"data:{mime};base64,{encoded}"


def _save_calibration_image(source_path: str | None) -> str | None:
    source = _as_path(source_path)
    if source is None or not os.path.isfile(source):
        return None
    image = cv2.imread(source)
    if image is None:
        return None
    _ensure_project_dirs()
    cv2.imwrite(_CALIBRATION_IMAGE_FILE, image)
    return _CALIBRATION_IMAGE_FILE


def _load_calibration_state() -> dict:
    default_state = {
        "image_path": None,
        "pixel_points": [],
        "meter_points": _DEFAULT_METER_PTS.to_dict(orient="records"),
        "grid_spacing_m": 1.0,
    }
    if not os.path.isfile(_CALIBRATION_STATE_FILE):
        return default_state
    try:
        with open(_CALIBRATION_STATE_FILE, "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except Exception:
        return default_state
    if not isinstance(state, dict):
        return default_state
    for key, value in default_state.items():
        state.setdefault(key, value)
    return state


def _persist_calibration_state(
    image_path: str | None,
    pixel_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> None:
    _ensure_project_dirs()
    payload = {
        "image_path": image_path,
        "pixel_points": pixel_df[["px", "py"]].to_dict(orient="records"),
        "meter_points": meter_df[["x_m", "y_m"]].to_dict(orient="records"),
        "grid_spacing_m": float(grid_spacing),
        "updated_at": _timestamp(),
    }
    with open(_CALIBRATION_STATE_FILE, "w", encoding="utf-8") as state_file:
        json.dump(payload, state_file, ensure_ascii=False, indent=2)


def _build_calibration_preview_html(image_path: str | None, points: list[list[float]] | None = None) -> str:
    image_url = _image_to_data_url(image_path)
    points_json = json.dumps(points or [], ensure_ascii=False)
    if image_url:
        image_block = (
            '<canvas id="calib-preview-canvas" style="width:100%;max-width:100%;border:1px solid #d0d7de;'
            'border-radius:12px;display:block;background:#111;"></canvas>'
        )
        status_text = "Klicke nacheinander auf 4 Referenzpunkte."
    else:
        image_block = (
            '<div style="height:420px;display:flex;align-items:center;justify-content:center;'
            'border:1px dashed #999;border-radius:12px;color:#666;background:#fafafa;">'
            'Noch kein Bild geladen.</div>'
        )
        status_text = "Lade ein Bild in die Vorschau."
    return (
        f'<div id="calib-preview-root" data-image-url="{image_url}" data-points="{points_json}">'
        f'<div style="margin-bottom:0.5rem;font-weight:600;">Kalibrierungs-Vorschau</div>'
        f'<div style="margin-bottom:0.75rem;color:#666;font-size:0.92rem;">{status_text}</div>'
        f'{image_block}'
        f'</div>'
    )


def _load_calibration_ui_state() -> tuple[str, pd.DataFrame, pd.DataFrame, str, str, str]:
    state = _load_calibration_state()
    image_path = state.get("image_path")
    pixel_points = state.get("pixel_points", [])
    meter_points = state.get("meter_points", _DEFAULT_METER_PTS.to_dict(orient="records"))
    pixel_df = _points_to_dataframe(pixel_points) if pixel_points else _empty_pixel_points_df()
    meter_df = pd.DataFrame(meter_points) if meter_points else _DEFAULT_METER_PTS.copy()
    if list(meter_df.columns) != ["x_m", "y_m"]:
        meter_df = _DEFAULT_METER_PTS.copy()

    points_json = json.dumps(pixel_points, ensure_ascii=False)
    preview_html = _build_calibration_preview_html(image_path, pixel_points)
    status = f"Gespeichert: {len(pixel_points)}/4 Referenzpunkte" if pixel_points else "Noch keine Referenzpunkte gespeichert."
    return preview_html, pixel_df, meter_df, points_json, str(image_path or ""), status


def _open_calibration_preview(
    image_file: str | None,
    current_meter_df: pd.DataFrame,
    grid_spacing: float,
) -> tuple[str, pd.DataFrame, str, str, str]:
    if image_file is None:
        raise gr.Error("Bitte zuerst ein Bild auswählen.")

    image_path = _as_path(image_file)
    if image_path is None or not os.path.isfile(image_path):
        raise gr.Error("Das ausgewählte Bild konnte nicht gelesen werden.")

    saved_image = _save_calibration_image(image_path)
    if saved_image is None:
        raise gr.Error("Das Bild konnte nicht in data/calibration gespeichert werden.")

    empty_points = _empty_pixel_points_df()
    _persist_calibration_state(saved_image, empty_points, current_meter_df, grid_spacing)

    preview_html = _build_calibration_preview_html(saved_image, [])
    return (
        preview_html,
        empty_points,
        json.dumps([], ensure_ascii=False),
        saved_image,
        "Bild geladen. Jetzt die 4 Punkte im Bild anklicken.",
    )


def _reset_calibration_points(
    image_path: str | None,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> tuple[str, pd.DataFrame, str, str]:
    empty_points = _empty_pixel_points_df()
    _persist_calibration_state(image_path or None, empty_points, meter_df, grid_spacing)
    preview_html = _build_calibration_preview_html(image_path, [])
    return preview_html, empty_points, json.dumps([], ensure_ascii=False), "Punkte zurückgesetzt."


def _save_current_calibration_state(
    image_path: str | None,
    pixel_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> str:
    if image_path:
        _save_calibration_image(image_path)
    _persist_calibration_state(image_path or None, pixel_df, meter_df, grid_spacing)
    return "Kalibrierungszustand gespeichert."


# ---------------------------------------------------------------------------
# Tab 1 helpers
# ---------------------------------------------------------------------------

def _run_calibration(
    image_path: str | None,
    pixel_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> tuple[np.ndarray | None, str | None]:
    """Compute homography and return (BEV image, calibration .npy path)."""
    if image_path is None:
        raise gr.Error("Please upload a video frame image.")

    frame_path = _as_path(image_path)
    if frame_path is None or not os.path.isfile(frame_path):
        raise gr.Error("Please upload a valid video frame image.")

    frame = cv2.imread(frame_path)
    if frame is None:
        raise gr.Error("Could not read the uploaded frame image.")

    try:
        src = pixel_df[["px", "py"]].values.astype(np.float32)
        dst = meter_df[["x_m", "y_m"]].values.astype(np.float32)
        H = compute_homography(src, dst)
    except Exception as exc:
        raise gr.Error(f"Calibration failed: {exc}") from exc

    bev = validate_calibration(frame, H, grid_spacing_m=grid_spacing)

    _ensure_project_dirs()
    saved_image = _save_calibration_image(frame_path)

    stamp = _timestamp()
    npy_path = os.path.join(_CALIBRATION_DIR, f"homography_{stamp}.npy")
    save_calibration(H, npy_path)

    params_path = os.path.join(_CALIBRATION_DIR, f"calibration_params_{stamp}.json")
    params_payload = {
        "pixel_points": pixel_df[["px", "py"]].to_dict(orient="records"),
        "meter_points": meter_df[["x_m", "y_m"]].to_dict(orient="records"),
        "grid_spacing_m": float(grid_spacing),
        "homography_file": os.path.basename(npy_path),
    }
    with open(params_path, "w", encoding="utf-8") as params_file:
        json.dump(params_payload, params_file, ensure_ascii=False, indent=2)

    if saved_image is not None:
        _persist_calibration_state(saved_image, pixel_df, meter_df, grid_spacing)

    return bev, npy_path


def _sync_clicked_points(
    image_path: str | None,
    points_json: str | None,
    meter_df: pd.DataFrame,
    grid_spacing: float,
) -> tuple[pd.DataFrame, str, str]:
    points = _parse_click_state(points_json)
    pixel_df = _points_to_dataframe(points) if points else _empty_pixel_points_df()
    _persist_calibration_state(image_path or None, pixel_df, meter_df, grid_spacing)
    preview_html = _build_calibration_preview_html(image_path, points)
    status = (
        f"Referenzpunkte gesetzt: {len(points)}/4"
        if points
        else "Klicke 4 Referenzpunkte der Reihe nach in das Bild."
    )
    return pixel_df, preview_html, status


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

    _ensure_project_dirs()
    saved_video_path = _copy_uploaded_file(video_file, _VIDEO_DIR, "video")

    try:
        H = load_calibration(calib_file)
    except Exception as exc:
        raise gr.Error(f"Cannot load calibration: {exc}") from exc

    progress(0, desc="Extracting trajectories…")

    def _prog_cb(current: int, total: int) -> None:
        progress(current / max(total, 1), desc=f"Tracking frame {current}/{total}")

    try:
        df = extract_trajectories_from_video(
            video_path=saved_video_path,
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

    csv_path = os.path.join(_TRAJECTORY_DIR, f"trajectories_{_timestamp()}.csv")
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

(
    INITIAL_CALIB_PREVIEW_HTML,
    INITIAL_PIXEL_POINTS_DF,
    INITIAL_METER_POINTS_DF,
    INITIAL_POINTS_JSON,
    INITIAL_IMAGE_PATH,
    INITIAL_POINT_STATUS,
) = _load_calibration_ui_state()

click_sync_js = r"""
(() => {
    const bindCalibrationClicks = () => {
        const root = document.getElementById('calib-frame');
        const state = document.querySelector('#calib-points-state textarea, #calib-points-state input');
        if (!root || !state) {
            return false;
        }

        const img = root.querySelector('img');
        if (!img || img.dataset.clickBound === '1') {
            return false;
        }

        img.dataset.clickBound = '1';
        img.style.cursor = 'crosshair';
        img.addEventListener('click', (event) => {
            const rect = img.getBoundingClientRect();
            if (!rect.width || !rect.height) {
                return;
            }

            let points = [];
            try {
                points = state.value ? JSON.parse(state.value) : [];
            } catch (error) {
                points = [];
            }

            if (!Array.isArray(points)) {
                points = [];
            }
            if (points.length >= 4) {
                return;
            }

            const scaleX = (img.naturalWidth || rect.width) / rect.width;
            const scaleY = (img.naturalHeight || rect.height) / rect.height;
            const x = (event.clientX - rect.left) * scaleX;
            const y = (event.clientY - rect.top) * scaleY;

            points.push([
                Math.round(x * 1000) / 1000,
                Math.round(y * 1000) / 1000,
            ]);

            state.value = JSON.stringify(points);
            state.dispatchEvent(new Event('input', { bubbles: true }));
        });

        return true;
    };

    const interval = setInterval(() => {
        if (bindCalibrationClicks()) {
            clearInterval(interval);
        }
    }, 250);

    bindCalibrationClicks();
})();
"""

with gr.Blocks(title="Pedestrian Crossing Analysis", theme=gr.themes.Default(), head=CALIBRATION_PREVIEW_HEAD) as demo:
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
                calib_frame = gr.Image(
                    label="Bild auswählen",
                    type="filepath",
                    sources=["upload"],
                    elem_id="calib-upload",
                )
                preview_image_path = gr.Textbox(
                    value=INITIAL_IMAGE_PATH,
                    visible=False,
                    elem_id="calib-image-path",
                )
                calib_points_state = gr.Textbox(
                    value=INITIAL_POINTS_JSON,
                    visible=False,
                    elem_id="calib-points-state",
                )
                pixel_pts_df = gr.Dataframe(
                    value=INITIAL_PIXEL_POINTS_DF,
                    label="4 Pixel Points (px, py)",
                    col_count=(2, "fixed"),
                    interactive=True,
                )
                point_status = gr.Textbox(
                    value=INITIAL_POINT_STATUS,
                    label="Punkt-Status",
                    interactive=False,
                )
                meter_pts_df = gr.Dataframe(
                    value=INITIAL_METER_POINTS_DF,
                    label="4 Meter Points (x_m, y_m)",
                    col_count=(2, "fixed"),
                    interactive=True,
                )
                grid_slider = gr.Slider(
                    minimum=0.5, maximum=5.0, step=0.5, value=1.0,
                    label="Grid spacing (m)",
                )
                with gr.Row():
                    open_btn = gr.Button("Bild in Vorschau öffnen", variant="primary")
                    reset_btn = gr.Button("Punkte zurücksetzen")
                    save_btn = gr.Button("Zustand speichern")
                calib_btn = gr.Button("Kalibrierung berechnen", variant="primary")
            with gr.Column():
                preview_html = gr.HTML(value=INITIAL_CALIB_PREVIEW_HTML)
                bev_output = gr.Image(label="BEV Validation Grid")
                calib_file_out = gr.File(label="Download Homography (.npy)")

        open_btn.click(
            fn=_open_calibration_preview,
            inputs=[calib_frame, meter_pts_df, grid_slider],
            outputs=[preview_html, pixel_pts_df, calib_points_state, preview_image_path, point_status],
        )

        calib_points_state.change(
            fn=_sync_clicked_points,
            inputs=[preview_image_path, calib_points_state, meter_pts_df, grid_slider],
            outputs=[pixel_pts_df, preview_html, point_status],
        )

        reset_btn.click(
            fn=_reset_calibration_points,
            inputs=[preview_image_path, meter_pts_df, grid_slider],
            outputs=[preview_html, pixel_pts_df, calib_points_state, point_status],
        )

        save_btn.click(
            fn=_save_current_calibration_state,
            inputs=[preview_image_path, pixel_pts_df, meter_pts_df, grid_slider],
            outputs=point_status,
        )

        calib_btn.click(
            fn=_run_calibration,
            inputs=[preview_image_path, pixel_pts_df, meter_pts_df, grid_slider],
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

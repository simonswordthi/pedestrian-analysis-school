#!/usr/bin/env python3
"""
Bootstrap script for the Pedestrian Crossing Analysis project.

Run once from the repository root:
    python init_project.py

This creates the full ``pedestrian_analysis/`` package with all source files.
"""

import os
import textwrap

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# File contents  (key = path relative to BASE)
# ---------------------------------------------------------------------------

FILES: dict[str, str] = {}

# ── config.py ───────────────────────────────────────────────────────────────
FILES["pedestrian_analysis/config.py"] = '''\
"""Global configuration parameters for the pedestrian crossing analysis pipeline."""

# ---------------------------------------------------------------------------
# Behavior-labeling thresholds
# ---------------------------------------------------------------------------

SPEED_THRESHOLD_MS: float = 0.3
"""Speed (m/s) below which a pedestrian is classified as \'waiting\'."""

WAITING_MIN_FRAMES: int = 5
"""Hysteresis: a new state must persist for this many frames before committing."""

# ---------------------------------------------------------------------------
# Group-detection parameters
# ---------------------------------------------------------------------------

PROXIMITY_THRESHOLD_M: float = 1.5
"""DBSCAN eps (m): pedestrians closer than this are considered one group."""

MIN_GROUP_FRAMES: int = 10
"""Minimum number of frames before a group is considered stable."""

TEMPORAL_SMOOTH_WINDOW: int = 10
"""Sliding window size (frames) for temporal smoothing of group assignments."""

# ---------------------------------------------------------------------------
# Leader-follower analysis
# ---------------------------------------------------------------------------

LEADER_FOLLOWER_MAX_DELAY_S: float = 5.0
"""Maximum follower delay (s) to be attributed to the same crossing event."""

# ---------------------------------------------------------------------------
# Paper-export / plot settings
# ---------------------------------------------------------------------------

PLOT_WIDTH_PX: int = 1200
PLOT_HEIGHT_PX: int = 800
PLOT_FONT_SIZE: int = 14
PLOT_FONT_FAMILY: str = "Arial"
PLOT_TEMPLATE: str = "plotly_white"
DPI: int = 300
'''

# ── pipeline/__init__.py ────────────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/__init__.py"] = '''\
"""Pipeline subpackage: calibration, tracking, analysis, labeling."""
'''

# ── pipeline/calibration.py ─────────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/calibration.py"] = '''\
"""Homography calibration: pixel → ground-plane (meter) transformation."""

from __future__ import annotations

import cv2
import numpy as np


def compute_homography(
    src_pixel_points: np.ndarray,
    dst_meter_points: np.ndarray,
) -> np.ndarray:
    """Compute 3×3 homography matrix H mapping image pixels to ground meters.

    Args:
        src_pixel_points: shape (4, 2) – pixel coordinates of the 4 markers.
        dst_meter_points: shape (4, 2) – corresponding real-world meter coords.

    Returns:
        H: 3×3 homography matrix (np.ndarray, float64).

    Raises:
        ValueError: if input arrays do not have the expected shape.
    """
    src = np.asarray(src_pixel_points, dtype=np.float32)
    dst = np.asarray(dst_meter_points, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(
            f"Expected shape (4, 2) for both point arrays, "
            f"got {src.shape} and {dst.shape}."
        )
    H, status = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography returned None – check input points.")
    return H


def pixel_to_meter(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """Transform a single foot-point from pixel to meter coordinates.

    Uses cv2.perspectiveTransform for numerical stability.

    Args:
        px, py: pixel coordinates (bottom-centre of bounding box).
        H:      3×3 homography matrix.

    Returns:
        (x_m, y_m): ground-plane coordinates in metres.
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    x_m, y_m = result[0, 0]
    return float(x_m), float(y_m)


def validate_calibration(
    frame: np.ndarray,
    H: np.ndarray,
    grid_spacing_m: float = 1.0,
    output_size: tuple[int, int] = (800, 800),
) -> np.ndarray:
    """Warp frame to BEV and overlay a metric grid for visual validation.

    Args:
        frame:          BGR image (original camera view).
        H:              3×3 homography matrix.
        grid_spacing_m: distance between grid lines in metres.
        output_size:    (width, height) of the output BEV image.

    Returns:
        Annotated BEV image (np.ndarray, BGR).
    """
    bev = cv2.warpPerspective(frame, H, output_size)
    h, w = bev.shape[:2]
    # Determine metric extent by back-projecting corners
    corners_px = np.array(
        [[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32
    )
    corners_m = cv2.perspectiveTransform(corners_px, H)[:, 0, :]
    x_min, y_min = corners_m.min(axis=0)
    x_max, y_max = corners_m.max(axis=0)

    H_inv = np.linalg.inv(H)
    color = (0, 200, 0)  # green
    font = cv2.FONT_HERSHEY_SIMPLEX

    x = x_min
    while x <= x_max:
        pt_top = cv2.perspectiveTransform(
            np.array([[[x, y_min]]], dtype=np.float32), H_inv
        )[0, 0]
        pt_bot = cv2.perspectiveTransform(
            np.array([[[x, y_max]]], dtype=np.float32), H_inv
        )[0, 0]
        cv2.line(bev, tuple(pt_top.astype(int)), tuple(pt_bot.astype(int)), color, 1)
        cv2.putText(
            bev, f"{x:.1f}m", tuple(pt_top.astype(int)),
            font, 0.4, color, 1, cv2.LINE_AA,
        )
        x += grid_spacing_m

    y = y_min
    while y <= y_max:
        pt_left = cv2.perspectiveTransform(
            np.array([[[x_min, y]]], dtype=np.float32), H_inv
        )[0, 0]
        pt_right = cv2.perspectiveTransform(
            np.array([[[x_max, y]]], dtype=np.float32), H_inv
        )[0, 0]
        cv2.line(bev, tuple(pt_left.astype(int)), tuple(pt_right.astype(int)), color, 1)
        cv2.putText(
            bev, f"{y:.1f}m", tuple(pt_left.astype(int)),
            font, 0.4, color, 1, cv2.LINE_AA,
        )
        y += grid_spacing_m

    return bev


def save_calibration(H: np.ndarray, path: str) -> None:
    """Save homography matrix to a .npy file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, H)


def load_calibration(path: str) -> np.ndarray:
    """Load homography matrix from a .npy file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    return np.load(path)


import os  # noqa: E402 – needed by save_calibration
'''

# ── pipeline/tracker.py ─────────────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/tracker.py"] = '''\
"""Video → trajectory extraction using YOLOv8 + ByteTrack (supervision)."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .calibration import pixel_to_meter


def extract_trajectories_from_video(
    video_path: str,
    H: np.ndarray,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
    class_filter: list[int] | None = None,
    frame_skip: int = 1,
    progress_callback=None,
) -> pd.DataFrame:
    """Extract per-frame pedestrian trajectories from a video.

    Args:
        video_path:           Path to the input video file.
        H:                    3×3 homography matrix (pixel → metre).
        model_name:           YOLOv8 model weights file or name.
        confidence_threshold: Minimum detection confidence.
        class_filter:         COCO class IDs to keep (default: [0] = person).
        frame_skip:           Process every Nth frame (1 = every frame).
        progress_callback:    Optional callable(current, total) for progress.

    Returns:
        DataFrame with columns: frame, id, x, y, px, py.

    Raises:
        FileNotFoundError: if video_path does not exist.
        RuntimeError:      if the video cannot be opened.
    """
    import os
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if class_filter is None:
        class_filter = [0]

    try:
        from ultralytics import YOLO
        import supervision as sv
    except ImportError as exc:
        raise ImportError(
            "ultralytics and supervision are required for tracking. "
            "Install them with: pip install ultralytics supervision"
        ) from exc

    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(
                frame,
                conf=confidence_threshold,
                classes=class_filter,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                track_id = int(detections.tracker_id[i])
                # foot point = bottom-centre of bounding box
                foot_px = (x1 + x2) / 2.0
                foot_py = float(y2)
                x_m, y_m = pixel_to_meter(foot_px, foot_py, H)
                records.append(
                    {
                        "frame": frame_idx,
                        "id": track_id,
                        "x": x_m,
                        "y": y_m,
                        "px": foot_px,
                        "py": foot_py,
                    }
                )

            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()

    if not records:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "px", "py"])

    df = pd.DataFrame(records)
    df = df.sort_values(["id", "frame"]).reset_index(drop=True)
    return df
'''

# ── pipeline/trajectory_io.py ───────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/trajectory_io.py"] = '''\
"""Load and save trajectory DataFrames in PedPy-compatible CSV format."""

from __future__ import annotations

import os

import pandas as pd

REQUIRED_COLUMNS = {"id", "frame", "x", "y"}


def save_trajectories(df: pd.DataFrame, path: str) -> None:
    """Save trajectory DataFrame to CSV.

    The CSV will always contain at least: id, frame, x, y.
    Additional columns (px, py, speed_ms, heading_deg, behavior, …) are
    preserved if present.

    Args:
        df:   Trajectory DataFrame.
        path: Destination file path (.csv).

    Raises:
        ValueError: if required columns are missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)


def load_trajectories(path: str) -> pd.DataFrame:
    """Load trajectory DataFrame from CSV.

    Args:
        path: Path to a .csv file.

    Returns:
        DataFrame with at least columns: id, frame, x, y.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError:        if required columns are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Trajectory CSV is missing required columns: {missing}"
        )
    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df
'''

# ── pipeline/pedpy_analysis.py ──────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/pedpy_analysis.py"] = '''\
"""PedPy wrapper: compute speed and heading for each tracked pedestrian."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kinematics(
    df: pd.DataFrame,
    fps: float,
    frame_step: int = 5,
) -> pd.DataFrame:
    """Compute per-frame speed (m/s) and heading angle (degrees) with PedPy.

    Args:
        df:         Trajectory DataFrame (must have: id, frame, x, y).
        fps:        Video frame rate in frames per second.
        frame_step: Frame interval used by PedPy for finite-difference speed.

    Returns:
        DataFrame with additional columns: \'speed_ms\', \'heading_deg\'.

    Raises:
        ImportError: if pedpy is not installed.
        ValueError:  if required columns are missing.
    """
    required = {"id", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    try:
        import pedpy
    except ImportError as exc:
        raise ImportError(
            "pedpy is required. Install it with: pip install pedpy"
        ) from exc

    df = df.copy()

    # PedPy expects columns: id, frame, x, y  (may also have z)
    traj_data = pedpy.TrajectoryData(
        data=df[["id", "frame", "x", "y"]].copy(),
        frame_rate=fps,
    )

    speed_df = pedpy.compute_individual_speed(
        traj_data=traj_data,
        frame_step=frame_step,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDE,
    )
    # speed_df has columns: id, frame, speed
    speed_df = speed_df.rename(columns={"speed": "speed_ms"})

    df = df.merge(speed_df[["id", "frame", "speed_ms"]], on=["id", "frame"], how="left")

    # Heading from consecutive (x, y) differences
    df = df.sort_values(["id", "frame"]).reset_index(drop=True)
    dx = df.groupby("id")["x"].diff()
    dy = df.groupby("id")["y"].diff()
    df["heading_deg"] = np.degrees(np.arctan2(dy, dx))

    return df
'''

# ── pipeline/behavior_labeling.py ───────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/behavior_labeling.py"] = '''\
"""Finite-state-machine behavior labeling: waiting / approaching / crossing / crossed."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ("waiting", "approaching", "crossing", "crossed")


def _classify_raw(row: pd.Series, street_start: float, street_end: float,
                  speed_thr: float) -> str:
    """Single-row raw state classification (no hysteresis)."""
    x = row["x"]
    speed = row.get("speed_ms", 0.0)
    if pd.isna(speed):
        speed = 0.0
    if x > street_end:
        return "crossed"
    if x >= street_start:
        return "crossing"
    if speed >= speed_thr:
        return "approaching"
    return "waiting"


def label_behaviors(
    df: pd.DataFrame,
    street_start_m: float,
    street_end_m: float,
    speed_threshold_ms: float = 0.3,
    waiting_min_frames: int = 5,
) -> pd.DataFrame:
    """Add behavior labels and timing metadata to the trajectory DataFrame.

    Applies hysteresis: a new state must be sustained for *waiting_min_frames*
    consecutive frames before it is committed (prevents flickering).

    Args:
        df:                 Trajectory DataFrame with columns id, frame, x, y,
                            speed_ms (call compute_kinematics first).
        street_start_m:     X coordinate where the street begins.
        street_end_m:       X coordinate where the street ends.
        speed_threshold_ms: Speed below which a pedestrian is considered waiting.
        waiting_min_frames: Hysteresis window size in frames.

    Returns:
        df with added columns: behavior, waiting_duration_s,
        waiting_start_frame, crossing_start_frame.
    """
    required = {"id", "frame", "x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if "speed_ms" not in df.columns:
        df = df.copy()
        df["speed_ms"] = 0.0

    df = df.copy().sort_values(["id", "frame"]).reset_index(drop=True)

    behavior_col: list[str] = ["waiting"] * len(df)
    waiting_start_frame_col: list[float] = [np.nan] * len(df)
    crossing_start_frame_col: list[float] = [np.nan] * len(df)

    for pid, group in df.groupby("id"):
        idx = group.index.tolist()
        raw_states = [
            _classify_raw(group.loc[i], street_start_m, street_end_m,
                          speed_threshold_ms)
            for i in idx
        ]

        # Apply hysteresis
        committed = raw_states[0]
        candidate = raw_states[0]
        candidate_count = 1
        smoothed: list[str] = [committed]

        for s in raw_states[1:]:
            if s == candidate:
                candidate_count += 1
            else:
                candidate = s
                candidate_count = 1

            if candidate_count >= waiting_min_frames:
                committed = candidate
            smoothed.append(committed)

        # Extract timing
        frames = group["frame"].tolist()
        waiting_start: float | None = None
        crossing_start: float | None = None

        for k, state in enumerate(smoothed):
            if state == "waiting" and waiting_start is None:
                waiting_start = float(frames[k])
            if state == "crossing" and crossing_start is None:
                crossing_start = float(frames[k])

        for k, i in enumerate(idx):
            behavior_col[i] = smoothed[k]
            waiting_start_frame_col[i] = waiting_start if waiting_start is not None else np.nan
            crossing_start_frame_col[i] = crossing_start if crossing_start is not None else np.nan

    df["behavior"] = behavior_col
    df["waiting_start_frame"] = waiting_start_frame_col
    df["crossing_start_frame"] = crossing_start_frame_col

    # waiting_duration_s requires fps; store in frames for now
    df["waiting_duration_frames"] = (
        df["crossing_start_frame"] - df["waiting_start_frame"]
    ).clip(lower=0)

    return df


def compute_crossing_events(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Extract one crossing-event row per individual.

    Args:
        df:  Labeled trajectory DataFrame (output of label_behaviors).
        fps: Video frame rate.

    Returns:
        DataFrame with columns: id, group_id, crossing_start_frame,
        crossing_end_frame, crossing_duration_s, waiting_duration_s.
    """
    required = {"id", "frame", "behavior"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    records: list[dict] = []
    has_group = "group_id" in df.columns

    for pid, group in df.groupby("id"):
        crossing_rows = group[group["behavior"] == "crossing"]
        if crossing_rows.empty:
            continue

        cross_start = int(crossing_rows["frame"].min())
        cross_end = int(crossing_rows["frame"].max())
        cross_dur = (cross_end - cross_start) / fps

        wait_start_frames = group["waiting_start_frame"].dropna()
        if len(wait_start_frames) > 0 and not pd.isna(group["crossing_start_frame"].iloc[0]):
            wait_frames = group["waiting_duration_frames"].iloc[0]
            wait_dur = float(wait_frames) / fps
        else:
            wait_dur = 0.0

        group_id = int(group["group_id"].iloc[0]) if has_group else -1

        records.append(
            {
                "id": pid,
                "group_id": group_id,
                "crossing_start_frame": cross_start,
                "crossing_end_frame": cross_end,
                "crossing_duration_s": cross_dur,
                "waiting_duration_s": wait_dur,
            }
        )

    return pd.DataFrame(records)
'''

# ── pipeline/group_analysis.py ──────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/group_analysis.py"] = '''\
"""DBSCAN-based group detection with temporal smoothing."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def detect_groups_per_frame(
    df: pd.DataFrame,
    proximity_threshold_m: float = 1.5,
    min_group_size: int = 2,
    temporal_smooth_window: int = 10,
    temporal_smooth_threshold: float = 0.7,
) -> pd.DataFrame:
    """Assign a stable group_id to every (frame, id) row.

    Steps
    -----
    1. Per frame: DBSCAN on (x, y) with eps=proximity_threshold_m.
    2. Temporal smoothing: each individual keeps a rolling history of
       group_id_frame values; the majority assignment across the last
       *temporal_smooth_window* frames becomes the committed group_id.

    Args:
        df:                       Trajectory DataFrame.
        proximity_threshold_m:    DBSCAN eps.
        min_group_size:           Clusters smaller than this are treated as
                                  singletons (group_id = -1).
        temporal_smooth_window:   Rolling window for smoothing.
        temporal_smooth_threshold: Fraction of window a label must hold.

    Returns:
        df with added column \'group_id\'.
    """
    required = {"id", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy().sort_values(["frame", "id"]).reset_index(drop=True)
    group_id_frame_col = pd.Series(-1, index=df.index, dtype=int)

    for frame_val, frame_df in df.groupby("frame"):
        if len(frame_df) < 2:
            continue
        coords = frame_df[["x", "y"]].values
        labels = DBSCAN(eps=proximity_threshold_m, min_samples=1).fit_predict(coords)
        # Remove clusters smaller than min_group_size
        counts = Counter(labels)
        for i, lab in zip(frame_df.index, labels):
            if lab == -1 or counts[lab] < min_group_size:
                group_id_frame_col[i] = -1
            else:
                group_id_frame_col[i] = int(lab)

    df["group_id_frame"] = group_id_frame_col

    # Temporal smoothing per individual
    history: dict[int, list[int]] = defaultdict(list)
    group_id_col = pd.Series(-1, index=df.index, dtype=int)

    for frame_val in sorted(df["frame"].unique()):
        frame_idx = df[df["frame"] == frame_val].index
        for i in frame_idx:
            pid = int(df.at[i, "id"])
            raw_gid = int(df.at[i, "group_id_frame"])
            history[pid].append(raw_gid)
            if len(history[pid]) > temporal_smooth_window:
                history[pid].pop(0)
            window = history[pid]
            counts = Counter(window)
            majority_gid, majority_count = counts.most_common(1)[0]
            if majority_count / len(window) >= temporal_smooth_threshold:
                group_id_col[i] = majority_gid
            else:
                group_id_col[i] = raw_gid

    df["group_id"] = group_id_col
    df = df.drop(columns=["group_id_frame"])
    return df


def compute_group_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for all detected groups.

    Returns a dict with keys:
    - group_count: int
    - size_histogram: dict {group_size: count}
    - split_events: list of dicts {frame, original_group_id, new_groups}
    - median_waiting_time_s: float (requires \'waiting_duration_s\' column)
    - partial_crossing_rate: float (requires \'behavior\' column)
    """
    required = {"id", "frame", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    real_groups = df[df["group_id"] >= 0]
    unique_groups = real_groups["group_id"].unique()
    group_count = len(unique_groups)

    # Group-size histogram: max size observed per group
    size_per_group = (
        real_groups.groupby(["frame", "group_id"])["id"]
        .nunique()
        .groupby("group_id")
        .max()
    )
    size_histogram: dict[int, int] = Counter(size_per_group.tolist())

    # Split events: group present in frame t but members diverge in t+k
    split_events: list[dict] = []
    frames = sorted(df["frame"].unique())
    for fi, frame_val in enumerate(frames[:-1]):
        next_frame = frames[fi + 1]
        curr = df[df["frame"] == frame_val][["id", "group_id"]]
        nxt = df[df["frame"] == next_frame][["id", "group_id"]]
        for gid in curr[curr["group_id"] >= 0]["group_id"].unique():
            members_now = set(curr[curr["group_id"] == gid]["id"])
            members_next_gids = nxt[nxt["id"].isin(members_now)]["group_id"].unique()
            if len(set(members_next_gids) - {-1}) > 1:
                split_events.append(
                    {
                        "frame": frame_val,
                        "original_group_id": gid,
                        "new_groups": members_next_gids.tolist(),
                    }
                )

    # Median waiting time
    median_waiting: float = float("nan")
    if "waiting_duration_s" in df.columns:
        wait_vals = df.drop_duplicates("id")["waiting_duration_s"].dropna()
        median_waiting = float(wait_vals.median()) if len(wait_vals) > 0 else float("nan")

    # Partial crossing rate
    partial_rate: float = float("nan")
    if "behavior" in df.columns and group_count > 0:
        partial_count = 0
        for gid in unique_groups:
            group_members = df[df["group_id"] == gid]
            behaviors = group_members.groupby("id")["behavior"].apply(
                lambda s: "crossing" in s.values or "crossed" in s.values
            )
            if behaviors.any() and not behaviors.all():
                partial_count += 1
        partial_rate = partial_count / group_count if group_count > 0 else 0.0

    return {
        "group_count": group_count,
        "size_histogram": dict(size_histogram),
        "split_events": split_events,
        "median_waiting_time_s": median_waiting,
        "partial_crossing_rate": partial_rate,
    }
'''

# ── pipeline/swarm_analysis.py ──────────────────────────────────────────────
FILES["pedestrian_analysis/pipeline/swarm_analysis.py"] = '''\
"""Reynolds flocking metrics: cohesion, alignment, separation, leader-follower."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import circstd


def compute_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    """Add \'cohesion_m\' column: distance from each individual to group centroid.

    Args:
        df: DataFrame with columns id, frame, x, y, group_id.

    Returns:
        df with added column \'cohesion_m\'.
    """
    required = {"id", "frame", "x", "y", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    cohesion_vals = pd.Series(np.nan, index=df.index, dtype=float)

    for (frame_val, gid), grp in df[df["group_id"] >= 0].groupby(["frame", "group_id"]):
        if len(grp) < 2:
            continue
        cx = grp["x"].mean()
        cy = grp["y"].mean()
        dists = np.sqrt((grp["x"] - cx) ** 2 + (grp["y"] - cy) ** 2)
        cohesion_vals[grp.index] = dists.values

    df["cohesion_m"] = cohesion_vals
    return df


def compute_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Add \'alignment_deg\' column: circular std of headings within each group.

    A small value means all members move in the same direction.

    Args:
        df: DataFrame with columns id, frame, heading_deg, group_id.

    Returns:
        df with added column \'alignment_deg\'.
    """
    required = {"id", "frame", "heading_deg", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    alignment_vals = pd.Series(np.nan, index=df.index, dtype=float)

    for (frame_val, gid), grp in df[df["group_id"] >= 0].groupby(["frame", "group_id"]):
        headings = grp["heading_deg"].dropna().values
        if len(headings) < 2:
            continue
        # circular std (degrees)
        circ_std = float(
            circstd(headings, high=180.0, low=-180.0) * 180.0 / np.pi
            if len(headings) >= 2 else np.nan
        )
        alignment_vals[grp.index] = circ_std

    df["alignment_deg"] = alignment_vals
    return df


def compute_separation(df: pd.DataFrame) -> pd.DataFrame:
    """Add \'min_separation_m\' column: min pairwise distance within each group.

    Args:
        df: DataFrame with columns id, frame, x, y, group_id.

    Returns:
        df with added column \'min_separation_m\'.
    """
    required = {"id", "frame", "x", "y", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    sep_vals = pd.Series(np.nan, index=df.index, dtype=float)

    for (frame_val, gid), grp in df[df["group_id"] >= 0].groupby(["frame", "group_id"]):
        if len(grp) < 2:
            continue
        positions = grp[["x", "y"]].values
        dists = pdist(positions)
        min_dist = float(dists.min())
        sep_vals[grp.index] = min_dist

    df["min_separation_m"] = sep_vals
    return df


def compute_leader_follower(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate leader-follower dynamics per group.

    The leader is the individual with the earliest *crossing_start_frame*.
    Follower delays are relative to the leader\'s crossing start.

    Args:
        df: DataFrame with columns id, group_id, crossing_start_frame.
            Typically the output of behavior_labeling merged with group_analysis.

    Returns:
        DataFrame with columns: group_id, leader_id, follower_ids,
        follower_delays_s (requires fps – here stored as frames; divide by fps
        externally).
    """
    required = {"id", "group_id", "crossing_start_frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    per_id = (
        df[df["group_id"] >= 0]
        .drop_duplicates("id")[["id", "group_id", "crossing_start_frame"]]
        .dropna(subset=["crossing_start_frame"])
    )

    records: list[dict] = []
    for gid, grp in per_id.groupby("group_id"):
        grp_sorted = grp.sort_values("crossing_start_frame")
        leader_row = grp_sorted.iloc[0]
        leader_id = int(leader_row["id"])
        leader_frame = float(leader_row["crossing_start_frame"])

        followers = grp_sorted.iloc[1:]
        follower_ids = followers["id"].tolist()
        follower_delays = (followers["crossing_start_frame"] - leader_frame).tolist()

        records.append(
            {
                "group_id": gid,
                "leader_id": leader_id,
                "follower_ids": follower_ids,
                "follower_delays_frames": follower_delays,
            }
        )

    return pd.DataFrame(records)
'''

# ── visualization/__init__.py ───────────────────────────────────────────────
FILES["pedestrian_analysis/visualization/__init__.py"] = '''\
"""Visualization subpackage: Plotly paper-quality figures."""
'''

# ── visualization/plot_trajectories.py ─────────────────────────────────────
FILES["pedestrian_analysis/visualization/plot_trajectories.py"] = '''\
"""Trajectory plot: colored lines per individual with street overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pedestrian_analysis import config


def plot_trajectories(
    df: pd.DataFrame,
    street_start_m: float,
    street_end_m: float,
) -> go.Figure:
    """Plot all trajectories as coloured lines on a 2D ground-plane.

    Args:
        df:              Trajectory DataFrame (id, x, y required; behavior optional).
        street_start_m:  X coordinate where street begins.
        street_end_m:    X coordinate where street ends.

    Returns:
        Plotly Figure ready for paper export.
    """
    fig = go.Figure()

    # Street area as shaded rectangle
    y_min = df["y"].min() - 1
    y_max = df["y"].max() + 1
    fig.add_shape(
        type="rect",
        x0=street_start_m, x1=street_end_m,
        y0=y_min, y1=y_max,
        fillcolor="lightgray",
        opacity=0.4,
        line_width=0,
        layer="below",
    )

    unique_ids = df["id"].unique()
    colors = [
        f"hsl({int(360 * i / len(unique_ids))},70%,50%)"
        for i in range(len(unique_ids))
    ]

    for pid, color in zip(unique_ids, colors):
        sub = df[df["id"] == pid].sort_values("frame")
        hover = sub.get("behavior", pd.Series([""] * len(sub))).tolist()

        # Trajectory line
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="lines",
                name=f"ID {pid}",
                line=dict(color=color, width=1.5),
                hovertemplate=(
                    "ID=%{customdata}<br>x=%{x:.2f} m<br>y=%{y:.2f} m"
                    "<br>behavior=%{text}<extra></extra>"
                ),
                text=hover,
                customdata=[pid] * len(sub),
                showlegend=True,
            )
        )

        # Start marker (triangle-up)
        fig.add_trace(
            go.Scatter(
                x=[sub["x"].iloc[0]],
                y=[sub["y"].iloc[0]],
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color=color),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # End marker (square)
        fig.add_trace(
            go.Scatter(
                x=[sub["x"].iloc[-1]],
                y=[sub["y"].iloc[-1]],
                mode="markers",
                marker=dict(symbol="square", size=8, color=color),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Group centroids as dashed lines
    if "group_id" in df.columns:
        for gid, grp in df[df["group_id"] >= 0].groupby("group_id"):
            centroid = grp.groupby("frame")[["x", "y"]].mean().reset_index()
            centroid = centroid.sort_values("frame")
            fig.add_trace(
                go.Scatter(
                    x=centroid["x"],
                    y=centroid["y"],
                    mode="lines",
                    name=f"Group {gid} centroid",
                    line=dict(dash="dash", width=1, color="black"),
                    opacity=0.5,
                )
            )

    fig.update_layout(
        template=config.PLOT_TEMPLATE,
        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE),
        width=config.PLOT_WIDTH_PX,
        height=config.PLOT_HEIGHT_PX,
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        yaxis=dict(showgrid=False),
        legend=dict(tracegroupgap=0),
    )
    return fig
'''

# ── visualization/plot_heatmap.py ───────────────────────────────────────────
FILES["pedestrian_analysis/visualization/plot_heatmap.py"] = '''\
"""Density heatmap of pedestrian positions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pedestrian_analysis import config


def plot_density_heatmap(
    df: pd.DataFrame,
    resolution_m: float = 0.5,
) -> go.Figure:
    """2-D histogram density heatmap of all recorded positions.

    Args:
        df:            Trajectory DataFrame (x, y required).
        resolution_m:  Bin size in metres.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=df["x"],
            y=df["y"],
            colorscale="YlOrRd",
            reversescale=False,
            xaxis="x",
            yaxis="y",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0.1)", size=3),
            showlegend=False,
        )
    )
    fig.update_layout(
        template=config.PLOT_TEMPLATE,
        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE),
        width=config.PLOT_WIDTH_PX,
        height=config.PLOT_HEIGHT_PX,
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        yaxis=dict(showgrid=False),
    )
    return fig
'''

# ── visualization/plot_behavior.py ──────────────────────────────────────────
FILES["pedestrian_analysis/visualization/plot_behavior.py"] = '''\
"""Gantt-style behavior timeline per individual."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pedestrian_analysis import config

BEHAVIOR_COLORS = {
    "waiting":    "steelblue",
    "approaching": "orange",
    "crossing":   "green",
    "crossed":    "gray",
}


def plot_behavior_timeline(df: pd.DataFrame, fps: float) -> go.Figure:
    """Gantt-like plot showing behavior labels over time for each individual.

    Args:
        df:  Labeled trajectory DataFrame (id, frame, behavior required).
        fps: Frames per second (used to convert frame numbers to seconds).

    Returns:
        Plotly Figure.
    """
    required = {"id", "frame", "behavior"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    fig = go.Figure()
    unique_ids = sorted(df["id"].unique())

    legend_added: set[str] = set()

    for pid in unique_ids:
        sub = df[df["id"] == pid].sort_values("frame").reset_index(drop=True)
        # Run-length encode behaviors
        prev_state = sub["behavior"].iloc[0]
        t_start = sub["frame"].iloc[0] / fps
        for k in range(1, len(sub)):
            state = sub["behavior"].iloc[k]
            if state != prev_state or k == len(sub) - 1:
                t_end = sub["frame"].iloc[k] / fps
                color = BEHAVIOR_COLORS.get(prev_state, "purple")
                show_leg = prev_state not in legend_added
                fig.add_trace(
                    go.Bar(
                        x=[t_end - t_start],
                        y=[str(pid)],
                        base=[t_start],
                        orientation="h",
                        name=prev_state,
                        marker_color=color,
                        showlegend=show_leg,
                        legendgroup=prev_state,
                        hovertemplate=(
                            f"ID {pid} – {prev_state}<br>"
                            f"t={t_start:.1f}s – {t_end:.1f}s<extra></extra>"
                        ),
                    )
                )
                legend_added.add(prev_state)
                prev_state = state
                t_start = sub["frame"].iloc[k] / fps

    # Vertical dashed lines at crossing events of group leaders
    if "crossing_start_frame" in df.columns and "group_id" in df.columns:
        leaders = (
            df[df["group_id"] >= 0]
            .dropna(subset=["crossing_start_frame"])
            .drop_duplicates("group_id")
        )
        for _, row in leaders.iterrows():
            t_cross = row["crossing_start_frame"] / fps
            fig.add_vline(
                x=t_cross,
                line=dict(color="red", dash="dash", width=1),
            )

    fig.update_layout(
        barmode="overlay",
        template=config.PLOT_TEMPLATE,
        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE),
        width=config.PLOT_WIDTH_PX,
        height=max(400, len(unique_ids) * 30 + 100),
        xaxis_title="Time (s)",
        yaxis_title="Pedestrian ID",
        yaxis=dict(showgrid=False),
    )
    return fig
'''

# ── visualization/plot_swarm.py ─────────────────────────────────────────────
FILES["pedestrian_analysis/visualization/plot_swarm.py"] = '''\
"""Swarm-metric time-series plots: cohesion, alignment, separation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pedestrian_analysis import config


def plot_swarm_metrics(df: pd.DataFrame, fps: float) -> go.Figure:
    """Three-panel subplot: cohesion, alignment, and separation over time.

    Args:
        df:  DataFrame with columns frame, cohesion_m, alignment_deg,
             min_separation_m (computed by swarm_analysis module).
        fps: Frames per second.

    Returns:
        Plotly Figure with three vertically stacked subplots.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Cohesion", "Alignment", "Separation"],
        vertical_spacing=0.08,
    )

    time_s = df["frame"] / fps

    if "cohesion_m" in df.columns:
        mean_coh = df.groupby("frame")["cohesion_m"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=mean_coh["frame"] / fps,
                y=mean_coh["cohesion_m"],
                mode="lines",
                name="Cohesion",
                line=dict(color="royalblue"),
            ),
            row=1, col=1,
        )

    if "alignment_deg" in df.columns:
        mean_aln = df.groupby("frame")["alignment_deg"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=mean_aln["frame"] / fps,
                y=mean_aln["alignment_deg"],
                mode="lines",
                name="Alignment",
                line=dict(color="darkorange"),
            ),
            row=2, col=1,
        )

    if "min_separation_m" in df.columns:
        mean_sep = df.groupby("frame")["min_separation_m"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=mean_sep["frame"] / fps,
                y=mean_sep["min_separation_m"],
                mode="lines",
                name="Separation",
                line=dict(color="seagreen"),
            ),
            row=3, col=1,
        )

    fig.update_layout(
        template=config.PLOT_TEMPLATE,
        font=dict(family=config.PLOT_FONT_FAMILY, size=config.PLOT_FONT_SIZE),
        width=config.PLOT_WIDTH_PX,
        height=config.PLOT_HEIGHT_PX,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Cohesion (m)", row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="Alignment (°)", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="Min. Separation (m)", row=3, col=1, showgrid=False)

    return fig
'''

# ── visualization/export.py ─────────────────────────────────────────────────
FILES["pedestrian_analysis/visualization/export.py"] = '''\
"""Export Plotly figures to PNG/PDF for paper inclusion."""

from __future__ import annotations

import os
import zipfile
from typing import Dict

import plotly.graph_objects as go

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pedestrian_analysis import config


def export_figure_png(
    fig: go.Figure,
    path: str,
    width: int = config.PLOT_WIDTH_PX,
    height: int = config.PLOT_HEIGHT_PX,
    scale: float = config.DPI / 96.0,
) -> None:
    """Export a Plotly figure to a PNG file.

    Args:
        fig:    Plotly Figure object.
        path:   Destination file path (will be created with parent dirs).
        width:  Image width in pixels.
        height: Image height in pixels.
        scale:  Scale factor (DPI / 96 gives roughly correct DPI for kaleido).

    Raises:
        ImportError: if kaleido is not installed.
    """
    try:
        import kaleido  # noqa: F401 – just check it is importable
    except ImportError as exc:
        raise ImportError(
            "kaleido is required for PNG export. Install: pip install kaleido"
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_image(path, format="png", width=width, height=height, scale=scale)


def export_figure_pdf(fig: go.Figure, path: str) -> None:
    """Export a Plotly figure to a PDF file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_image(path, format="pdf")


def export_all_figures(
    figures: Dict[str, go.Figure],
    output_dir: str,
) -> str:
    """Export all figures to PNGs and bundle them in a ZIP archive.

    Args:
        figures:    Mapping of file-stem → Plotly Figure.
        output_dir: Directory where PNG files and the ZIP are written.

    Returns:
        Path to the created ZIP file.
    """
    os.makedirs(output_dir, exist_ok=True)
    png_paths: list[str] = []
    for name, fig in figures.items():
        png_path = os.path.join(output_dir, f"{name}.png")
        export_figure_png(fig, png_path)
        png_paths.append(png_path)

    zip_path = os.path.join(output_dir, "figures.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in png_paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path
'''

# ── app.py ──────────────────────────────────────────────────────────────────
FILES["pedestrian_analysis/app.py"] = '''\
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
        raise gr.Error("CSV does not contain \'speed_ms\'. Run analysis first.")
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
        raise gr.Error("CSV does not contain \'behavior\'. Run analysis first.")
    return plot_behavior_timeline(df, fps)


def _load_and_plot_groups(csv_file: str | None):
    if csv_file is None:
        raise gr.Error("Please upload a trajectories CSV.")
    df = load_trajectories(csv_file)
    if "group_id" not in df.columns:
        raise gr.Error("CSV does not contain \'group_id\'. Run analysis first.")
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
            raise gr.Error(f"CSV does not contain \'{col}\'. Run analysis first.")
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
'''

# ── tests/test_pipeline.py ───────────────────────────────────────────────────
FILES["pedestrian_analysis/tests/__init__.py"] = ""

FILES["pedestrian_analysis/tests/test_pipeline.py"] = '''\
"""Unit tests for the pedestrian analysis pipeline."""

import math

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pedestrian_analysis.pipeline.calibration import compute_homography, pixel_to_meter
from pedestrian_analysis.pipeline.behavior_labeling import label_behaviors
from pedestrian_analysis.pipeline.group_analysis import detect_groups_per_frame
from pedestrian_analysis.pipeline.swarm_analysis import compute_cohesion


# ---------------------------------------------------------------------------
# calibration.py tests
# ---------------------------------------------------------------------------

class TestComputeHomography:
    """compute_homography should correctly transform known points."""

    def _make_simple_data(self):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
        return src, dst

    def test_known_corner_transform(self):
        src, dst = self._make_simple_data()
        H = compute_homography(src, dst)
        x_m, y_m = pixel_to_meter(100.0, 0.0, H)
        assert math.isclose(x_m, 5.0, abs_tol=1e-3)
        assert math.isclose(y_m, 0.0, abs_tol=1e-3)

    def test_centre_transform(self):
        src, dst = self._make_simple_data()
        H = compute_homography(src, dst)
        x_m, y_m = pixel_to_meter(50.0, 50.0, H)
        assert math.isclose(x_m, 2.5, abs_tol=0.01)
        assert math.isclose(y_m, 2.5, abs_tol=0.01)

    def test_wrong_shape_raises(self):
        src = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        with pytest.raises(ValueError):
            compute_homography(src, dst)


# ---------------------------------------------------------------------------
# behavior_labeling.py tests
# ---------------------------------------------------------------------------

def _make_behavior_df(xs: list[float], speeds: list[float]) -> pd.DataFrame:
    """Helper: one pedestrian (id=1), sequential frames."""
    n = len(xs)
    return pd.DataFrame(
        {
            "id": [1] * n,
            "frame": list(range(n)),
            "x": xs,
            "y": [0.0] * n,
            "speed_ms": speeds,
        }
    )


class TestLabelBehaviors:
    """label_behaviors should assign correct states."""

    STREET_START = 5.0
    STREET_END = 10.0
    SPEED_THR = 0.3

    def _label(self, xs, speeds):
        df = _make_behavior_df(xs, speeds)
        return label_behaviors(
            df,
            street_start_m=self.STREET_START,
            street_end_m=self.STREET_END,
            speed_threshold_ms=self.SPEED_THR,
            waiting_min_frames=1,  # no hysteresis for test clarity
        )

    def test_waiting_state(self):
        # Slow, before street → waiting
        df = self._label([1.0] * 10, [0.1] * 10)
        assert (df["behavior"] == "waiting").all()

    def test_approaching_state(self):
        # Fast, before street → approaching
        df = self._label([1.0] * 10, [0.5] * 10)
        assert (df["behavior"] == "approaching").all()

    def test_crossing_state(self):
        # Inside street → crossing
        df = self._label([7.0] * 10, [1.0] * 10)
        assert (df["behavior"] == "crossing").all()

    def test_crossed_state(self):
        # Past street → crossed
        df = self._label([12.0] * 10, [1.0] * 10)
        assert (df["behavior"] == "crossed").all()

    def test_transition_sequence(self):
        # waiting → approaching → crossing → crossed
        xs = [1.0] * 5 + [1.0] * 5 + [7.0] * 5 + [12.0] * 5
        speeds = [0.1] * 5 + [0.5] * 5 + [1.0] * 5 + [1.0] * 5
        df = self._label(xs, speeds)
        states = df["behavior"].tolist()
        assert states[:5] == ["waiting"] * 5
        assert states[5:10] == ["approaching"] * 5
        assert states[10:15] == ["crossing"] * 5
        assert states[15:20] == ["crossed"] * 5


# ---------------------------------------------------------------------------
# group_analysis.py tests
# ---------------------------------------------------------------------------

class TestDetectGroupsPerFrame:
    """detect_groups_per_frame should identify two distinct clusters."""

    def _two_group_df(self):
        # Two pedestrians 1 m apart (same cluster), one outlier 10 m away
        rows = []
        for frame in range(5):
            rows += [
                {"id": 1, "frame": frame, "x": 0.0, "y": 0.0},
                {"id": 2, "frame": frame, "x": 0.5, "y": 0.0},   # same cluster as 1
                {"id": 3, "frame": frame, "x": 10.0, "y": 0.0},  # separate cluster
            ]
        return pd.DataFrame(rows)

    def test_two_groups_detected(self):
        df = self._two_group_df()
        result = detect_groups_per_frame(df, proximity_threshold_m=1.5, min_group_size=2)
        # IDs 1 and 2 should share a group_id; ID 3 should be in a different group or -1
        last_frame = result[result["frame"] == 4]
        gid_1 = last_frame.loc[last_frame["id"] == 1, "group_id"].iloc[0]
        gid_2 = last_frame.loc[last_frame["id"] == 2, "group_id"].iloc[0]
        gid_3 = last_frame.loc[last_frame["id"] == 3, "group_id"].iloc[0]
        assert gid_1 == gid_2
        assert gid_3 != gid_1 or gid_3 == -1


# ---------------------------------------------------------------------------
# swarm_analysis.py tests
# ---------------------------------------------------------------------------

class TestComputeCohesion:
    """compute_cohesion should match analytically computed centroid distances."""

    def test_cohesion_equilateral_triangle(self):
        # Three points at vertices of an equilateral triangle with side 2 m
        # Centroid distance = 2 / sqrt(3) ≈ 1.1547 m
        s = 2.0
        h = s * math.sqrt(3) / 2.0
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "frame": [0, 0, 0],
                "x": [0.0, s, s / 2],
                "y": [0.0, 0.0, h],
                "group_id": [0, 0, 0],
            }
        )
        result = compute_cohesion(df)
        expected = s / math.sqrt(3)
        for _, row in result.iterrows():
            assert math.isclose(row["cohesion_m"], expected, rel_tol=1e-4)

    def test_cohesion_no_group(self):
        # Singletons (group_id = -1) should have NaN cohesion
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "frame": [0, 0],
                "x": [0.0, 10.0],
                "y": [0.0, 0.0],
                "group_id": [-1, -1],
            }
        )
        result = compute_cohesion(df)
        assert result["cohesion_m"].isna().all()
'''

# ── data and output placeholder .gitkeep files ──────────────────────────────
for placeholder in [
    "pedestrian_analysis/data/videos/.gitkeep",
    "pedestrian_analysis/data/trajectories/.gitkeep",
    "pedestrian_analysis/data/calibration/.gitkeep",
    "pedestrian_analysis/outputs/figures/.gitkeep",
]:
    FILES[placeholder] = ""

# ── .gitignore additions ─────────────────────────────────────────────────────
FILES["pedestrian_analysis/.gitignore"] = """\
# Ignore raw video files and large binary outputs
data/videos/*.mp4
data/videos/*.mov
data/videos/*.avi
data/calibration/*.npy
outputs/
__pycache__/
*.pyc
.DS_Store
"""

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write(rel_path: str, content: str) -> None:
    abs_path = os.path.join(BASE, rel_path)
    dir_path = os.path.dirname(abs_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  created  {rel_path}")


def main() -> None:
    print(f"Creating pedestrian_analysis project in: {BASE}")
    for rel_path, content in FILES.items():
        _write(rel_path, content)
    print(f"\n✅  Done – {len(FILES)} files written.")
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. cd pedestrian_analysis && python app.py")
    print("  3. Open http://localhost:7860 in your browser")


if __name__ == "__main__":
    main()

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

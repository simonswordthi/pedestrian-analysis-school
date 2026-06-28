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
        df with added column 'group_id'.
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
    - median_waiting_time_s: float (requires 'waiting_duration_s' column)
    - partial_crossing_rate: float (requires 'behavior' column)
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

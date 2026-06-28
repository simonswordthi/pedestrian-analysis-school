"""Reynolds flocking metrics: cohesion, alignment, separation, leader-follower."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import circstd


def compute_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'cohesion_m' column: distance from each individual to group centroid.

    Args:
        df: DataFrame with columns id, frame, x, y, group_id.

    Returns:
        df with added column 'cohesion_m'.
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
    """Add 'alignment_deg' column: circular std of headings within each group.

    A small value means all members move in the same direction.

    Args:
        df: DataFrame with columns id, frame, heading_deg, group_id.

    Returns:
        df with added column 'alignment_deg'.
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
    """Add 'min_separation_m' column: min pairwise distance within each group.

    Args:
        df: DataFrame with columns id, frame, x, y, group_id.

    Returns:
        df with added column 'min_separation_m'.
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
    Follower delays are relative to the leader's crossing start.

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

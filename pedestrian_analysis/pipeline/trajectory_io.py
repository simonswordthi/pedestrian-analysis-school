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

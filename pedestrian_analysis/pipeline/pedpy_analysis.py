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
        DataFrame with additional columns: 'speed_ms', 'heading_deg'.

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
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
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

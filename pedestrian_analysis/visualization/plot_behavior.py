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

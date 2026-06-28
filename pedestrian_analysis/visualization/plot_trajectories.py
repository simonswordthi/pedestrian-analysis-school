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

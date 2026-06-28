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

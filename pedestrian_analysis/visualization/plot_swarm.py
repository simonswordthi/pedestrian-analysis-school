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

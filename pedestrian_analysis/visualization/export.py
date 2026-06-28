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

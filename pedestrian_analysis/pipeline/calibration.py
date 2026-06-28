"""Homography calibration: pixel → ground-plane (meter) transformation."""

from __future__ import annotations

import cv2
import numpy as np


def compute_homography(
    src_pixel_points: np.ndarray,
    dst_meter_points: np.ndarray,
) -> np.ndarray:
    """Compute 3×3 homography matrix H mapping image pixels to ground meters.

    Args:
        src_pixel_points: shape (4, 2) – pixel coordinates of the 4 markers.
        dst_meter_points: shape (4, 2) – corresponding real-world meter coords.

    Returns:
        H: 3×3 homography matrix (np.ndarray, float64).

    Raises:
        ValueError: if input arrays do not have the expected shape.
    """
    src = np.asarray(src_pixel_points, dtype=np.float32)
    dst = np.asarray(dst_meter_points, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(
            f"Expected shape (4, 2) for both point arrays, "
            f"got {src.shape} and {dst.shape}."
        )
    H, status = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography returned None – check input points.")
    return H


def pixel_to_meter(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """Transform a single foot-point from pixel to meter coordinates.

    Uses cv2.perspectiveTransform for numerical stability.

    Args:
        px, py: pixel coordinates (bottom-centre of bounding box).
        H:      3×3 homography matrix.

    Returns:
        (x_m, y_m): ground-plane coordinates in metres.
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    x_m, y_m = result[0, 0]
    return float(x_m), float(y_m)


def validate_calibration(
    frame: np.ndarray,
    H: np.ndarray,
    grid_spacing_m: float = 1.0,
    output_size: tuple[int, int] = (800, 800),
) -> np.ndarray:
    """Warp frame to BEV and overlay a metric grid for visual validation.

    Args:
        frame:          BGR image (original camera view).
        H:              3×3 homography matrix.
        grid_spacing_m: distance between grid lines in metres.
        output_size:    (width, height) of the output BEV image.

    Returns:
        Annotated BEV image (np.ndarray, BGR).
    """
    bev = cv2.warpPerspective(frame, H, output_size)
    h, w = bev.shape[:2]
    # Determine metric extent by back-projecting corners
    corners_px = np.array(
        [[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32
    )
    corners_m = cv2.perspectiveTransform(corners_px, H)[:, 0, :]
    x_min, y_min = corners_m.min(axis=0)
    x_max, y_max = corners_m.max(axis=0)

    H_inv = np.linalg.inv(H)
    color = (0, 200, 0)  # green
    font = cv2.FONT_HERSHEY_SIMPLEX

    x = x_min
    while x <= x_max:
        pt_top = cv2.perspectiveTransform(
            np.array([[[x, y_min]]], dtype=np.float32), H_inv
        )[0, 0]
        pt_bot = cv2.perspectiveTransform(
            np.array([[[x, y_max]]], dtype=np.float32), H_inv
        )[0, 0]
        cv2.line(bev, tuple(pt_top.astype(int)), tuple(pt_bot.astype(int)), color, 1)
        cv2.putText(
            bev, f"{x:.1f}m", tuple(pt_top.astype(int)),
            font, 0.4, color, 1, cv2.LINE_AA,
        )
        x += grid_spacing_m

    y = y_min
    while y <= y_max:
        pt_left = cv2.perspectiveTransform(
            np.array([[[x_min, y]]], dtype=np.float32), H_inv
        )[0, 0]
        pt_right = cv2.perspectiveTransform(
            np.array([[[x_max, y]]], dtype=np.float32), H_inv
        )[0, 0]
        cv2.line(bev, tuple(pt_left.astype(int)), tuple(pt_right.astype(int)), color, 1)
        cv2.putText(
            bev, f"{y:.1f}m", tuple(pt_left.astype(int)),
            font, 0.4, color, 1, cv2.LINE_AA,
        )
        y += grid_spacing_m

    return bev


def save_calibration(H: np.ndarray, path: str) -> None:
    """Save homography matrix to a .npy file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, H)


def load_calibration(path: str) -> np.ndarray:
    """Load homography matrix from a .npy file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    return np.load(path)


import os  # noqa: E402 – needed by save_calibration

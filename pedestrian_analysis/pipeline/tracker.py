"""Video → trajectory extraction using YOLOv8 + ByteTrack (supervision)."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .calibration import pixel_to_meter


def extract_trajectories_from_video(
    video_path: str,
    H: np.ndarray,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
    class_filter: list[int] | None = None,
    frame_skip: int = 1,
    progress_callback=None,
) -> pd.DataFrame:
    """Extract per-frame pedestrian trajectories from a video.

    Args:
        video_path:           Path to the input video file.
        H:                    3×3 homography matrix (pixel → metre).
        model_name:           YOLOv8 model weights file or name.
        confidence_threshold: Minimum detection confidence.
        class_filter:         COCO class IDs to keep (default: [0] = person).
        frame_skip:           Process every Nth frame (1 = every frame).
        progress_callback:    Optional callable(current, total) for progress.

    Returns:
        DataFrame with columns: frame, id, x, y, px, py.

    Raises:
        FileNotFoundError: if video_path does not exist.
        RuntimeError:      if the video cannot be opened.
    """
    import os
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if class_filter is None:
        class_filter = [0]

    try:
        from ultralytics import YOLO
        import supervision as sv
    except ImportError as exc:
        raise ImportError(
            "ultralytics and supervision are required for tracking. "
            "Install them with: pip install ultralytics supervision"
        ) from exc

    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(
                frame,
                conf=confidence_threshold,
                classes=class_filter,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                track_id = int(detections.tracker_id[i])
                # foot point = bottom-centre of bounding box
                foot_px = (x1 + x2) / 2.0
                foot_py = float(y2)
                x_m, y_m = pixel_to_meter(foot_px, foot_py, H)
                records.append(
                    {
                        "frame": frame_idx,
                        "id": track_id,
                        "x": x_m,
                        "y": y_m,
                        "px": foot_px,
                        "py": foot_py,
                    }
                )

            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()

    if not records:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "px", "py"])

    df = pd.DataFrame(records)
    df = df.sort_values(["id", "frame"]).reset_index(drop=True)
    return df

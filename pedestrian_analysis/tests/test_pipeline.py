"""Unit tests for the pedestrian analysis pipeline."""

import math

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pedestrian_analysis.pipeline.calibration import compute_homography, pixel_to_meter
from pedestrian_analysis.pipeline.behavior_labeling import label_behaviors
from pedestrian_analysis.pipeline.group_analysis import detect_groups_per_frame
from pedestrian_analysis.pipeline.swarm_analysis import compute_cohesion


# ---------------------------------------------------------------------------
# calibration.py tests
# ---------------------------------------------------------------------------

class TestComputeHomography:
    """compute_homography should correctly transform known points."""

    def _make_simple_data(self):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
        return src, dst

    def test_known_corner_transform(self):
        src, dst = self._make_simple_data()
        H = compute_homography(src, dst)
        x_m, y_m = pixel_to_meter(100.0, 0.0, H)
        assert math.isclose(x_m, 5.0, abs_tol=1e-3)
        assert math.isclose(y_m, 0.0, abs_tol=1e-3)

    def test_centre_transform(self):
        src, dst = self._make_simple_data()
        H = compute_homography(src, dst)
        x_m, y_m = pixel_to_meter(50.0, 50.0, H)
        assert math.isclose(x_m, 2.5, abs_tol=0.01)
        assert math.isclose(y_m, 2.5, abs_tol=0.01)

    def test_wrong_shape_raises(self):
        src = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        with pytest.raises(ValueError):
            compute_homography(src, dst)


# ---------------------------------------------------------------------------
# behavior_labeling.py tests
# ---------------------------------------------------------------------------

def _make_behavior_df(xs: list[float], speeds: list[float]) -> pd.DataFrame:
    """Helper: one pedestrian (id=1), sequential frames."""
    n = len(xs)
    return pd.DataFrame(
        {
            "id": [1] * n,
            "frame": list(range(n)),
            "x": xs,
            "y": [0.0] * n,
            "speed_ms": speeds,
        }
    )


class TestLabelBehaviors:
    """label_behaviors should assign correct states."""

    STREET_START = 5.0
    STREET_END = 10.0
    SPEED_THR = 0.3

    def _label(self, xs, speeds):
        df = _make_behavior_df(xs, speeds)
        return label_behaviors(
            df,
            street_start_m=self.STREET_START,
            street_end_m=self.STREET_END,
            speed_threshold_ms=self.SPEED_THR,
            waiting_min_frames=1,  # no hysteresis for test clarity
        )

    def test_waiting_state(self):
        # Slow, before street → waiting
        df = self._label([1.0] * 10, [0.1] * 10)
        assert (df["behavior"] == "waiting").all()

    def test_approaching_state(self):
        # Fast, before street → approaching
        df = self._label([1.0] * 10, [0.5] * 10)
        assert (df["behavior"] == "approaching").all()

    def test_crossing_state(self):
        # Inside street → crossing
        df = self._label([7.0] * 10, [1.0] * 10)
        assert (df["behavior"] == "crossing").all()

    def test_crossed_state(self):
        # Past street → crossed
        df = self._label([12.0] * 10, [1.0] * 10)
        assert (df["behavior"] == "crossed").all()

    def test_transition_sequence(self):
        # waiting → approaching → crossing → crossed
        xs = [1.0] * 5 + [1.0] * 5 + [7.0] * 5 + [12.0] * 5
        speeds = [0.1] * 5 + [0.5] * 5 + [1.0] * 5 + [1.0] * 5
        df = self._label(xs, speeds)
        states = df["behavior"].tolist()
        assert states[:5] == ["waiting"] * 5
        assert states[5:10] == ["approaching"] * 5
        assert states[10:15] == ["crossing"] * 5
        assert states[15:20] == ["crossed"] * 5


# ---------------------------------------------------------------------------
# group_analysis.py tests
# ---------------------------------------------------------------------------

class TestDetectGroupsPerFrame:
    """detect_groups_per_frame should identify two distinct clusters."""

    def _two_group_df(self):
        # Two pedestrians 1 m apart (same cluster), one outlier 10 m away
        rows = []
        for frame in range(5):
            rows += [
                {"id": 1, "frame": frame, "x": 0.0, "y": 0.0},
                {"id": 2, "frame": frame, "x": 0.5, "y": 0.0},   # same cluster as 1
                {"id": 3, "frame": frame, "x": 10.0, "y": 0.0},  # separate cluster
            ]
        return pd.DataFrame(rows)

    def test_two_groups_detected(self):
        df = self._two_group_df()
        result = detect_groups_per_frame(df, proximity_threshold_m=1.5, min_group_size=2)
        # IDs 1 and 2 should share a group_id; ID 3 should be in a different group or -1
        last_frame = result[result["frame"] == 4]
        gid_1 = last_frame.loc[last_frame["id"] == 1, "group_id"].iloc[0]
        gid_2 = last_frame.loc[last_frame["id"] == 2, "group_id"].iloc[0]
        gid_3 = last_frame.loc[last_frame["id"] == 3, "group_id"].iloc[0]
        assert gid_1 == gid_2
        assert gid_3 != gid_1 or gid_3 == -1


# ---------------------------------------------------------------------------
# swarm_analysis.py tests
# ---------------------------------------------------------------------------

class TestComputeCohesion:
    """compute_cohesion should match analytically computed centroid distances."""

    def test_cohesion_equilateral_triangle(self):
        # Three points at vertices of an equilateral triangle with side 2 m
        # Centroid distance = 2 / sqrt(3) ≈ 1.1547 m
        s = 2.0
        h = s * math.sqrt(3) / 2.0
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "frame": [0, 0, 0],
                "x": [0.0, s, s / 2],
                "y": [0.0, 0.0, h],
                "group_id": [0, 0, 0],
            }
        )
        result = compute_cohesion(df)
        expected = s / math.sqrt(3)
        for _, row in result.iterrows():
            assert math.isclose(row["cohesion_m"], expected, rel_tol=1e-4)

    def test_cohesion_no_group(self):
        # Singletons (group_id = -1) should have NaN cohesion
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "frame": [0, 0],
                "x": [0.0, 10.0],
                "y": [0.0, 0.0],
                "group_id": [-1, -1],
            }
        )
        result = compute_cohesion(df)
        assert result["cohesion_m"].isna().all()

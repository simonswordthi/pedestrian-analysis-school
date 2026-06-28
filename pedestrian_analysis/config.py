"""Global configuration parameters for the pedestrian crossing analysis pipeline."""

# ---------------------------------------------------------------------------
# Behavior-labeling thresholds
# ---------------------------------------------------------------------------

SPEED_THRESHOLD_MS: float = 0.3
"""Speed (m/s) below which a pedestrian is classified as 'waiting'."""

WAITING_MIN_FRAMES: int = 5
"""Hysteresis: a new state must persist for this many frames before committing."""

# ---------------------------------------------------------------------------
# Group-detection parameters
# ---------------------------------------------------------------------------

PROXIMITY_THRESHOLD_M: float = 1.5
"""DBSCAN eps (m): pedestrians closer than this are considered one group."""

MIN_GROUP_FRAMES: int = 10
"""Minimum number of frames before a group is considered stable."""

TEMPORAL_SMOOTH_WINDOW: int = 10
"""Sliding window size (frames) for temporal smoothing of group assignments."""

# ---------------------------------------------------------------------------
# Leader-follower analysis
# ---------------------------------------------------------------------------

LEADER_FOLLOWER_MAX_DELAY_S: float = 5.0
"""Maximum follower delay (s) to be attributed to the same crossing event."""

# ---------------------------------------------------------------------------
# Paper-export / plot settings
# ---------------------------------------------------------------------------

PLOT_WIDTH_PX: int = 1200
PLOT_HEIGHT_PX: int = 800
PLOT_FONT_SIZE: int = 14
PLOT_FONT_FAMILY: str = "Arial"
PLOT_TEMPLATE: str = "plotly_white"
DPI: int = 300

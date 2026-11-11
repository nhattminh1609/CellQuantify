"""Central configuration for the Cell Counter application.

All tunable constants (paths, model thresholds, grid-detection parameters and
physical calibration) live here so they can be adjusted in one place.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODE1_MODEL_FILE = "mode1_model.onnx"
MODE2_MODEL_FILE = "mode2_multi_label.onnx"

# Accepted (fluorescence, bright-field) sub-folder pairs, tried in order.
CHANNEL_FOLDER_PAIRS = [
    ("green_background", "yellow_background"),
    ("bright_background", "dark_background"),
]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# QSettings identifiers
SETTINGS_ORG = "BioLab"
SETTINGS_APP = "CellCounter"

# ---------------------------------------------------------------------------
# Segmentation / inference
# ---------------------------------------------------------------------------
DEFAULT_MODEL_INPUT_SIZE = 640          # used when the ONNX input shape is dynamic

MODE1_CONF_THRESHOLD = 0.5              # lowered from 0.8 to reliably detect cells in dimmer images
MODE1_MIN_COMPONENT_AREA = 5            # px (model input space)

MODE2_CONF_THRESHOLD = 0.25             # default threshold for Mode 2
MODE2_UNDEFINE_CONF_THRESHOLD = 0.40    # elevated threshold to prevent spurious undefined false positives
MODE2_UNDEFINE_MIN_AREA = 20            # px (filter out background speckle noise)

MODE2_CLASS_CONF_THRESHOLDS = {
    "a549": 0.25,                       # epithelial cancer cells
    "nk": 0.20,                         # red fluorescent NK cells
    "undefine": MODE2_UNDEFINE_CONF_THRESHOLD  # selective threshold for genuine ambiguous/contact cells
}

MODE2_MIN_COMPONENT_AREA = 8            # px (default min area)
MODE2_CLASS_MIN_AREAS = {
    "a549": 15,                         # larger epithelial carcinoma cells (20-30 µm)
    "nk": 6,                            # small lymphocyte NK cells (7-10 µm)
    "undefine": MODE2_UNDEFINE_MIN_AREA # minimum component area for undefined cells
}
MODE2_CLASS_NAMES = ["a549", "nk", "undefine"]

# Border artifact suppression (only applied to boundary noise lines, not internal cells)
BORDER_MARGIN = 2                       # px margin from image boundary
MAX_COMPONENT_DIMENSION = 120           # px (only used for border-touching artifact rejection)
MAX_COMPONENT_AREA = 5000               # px

SINGLE_CELL_AREA_THRESHOLD = 100        # components smaller than this count as 1 cell

# Cluster splitting (local-maximum peak search inside large components)
CLUSTER_PAD_LEFT_TOP = 2
CLUSTER_PAD_RIGHT = 10
CLUSTER_PAD_BOTTOM = 5
CLUSTER_CLAHE_CLIP = 4.5
CLUSTER_CLAHE_TILE = (4, 4)
CLUSTER_INTENSITY_OFFSET = -10
CLUSTER_MIN_PEAK_INTENSITY = 2

# ---------------------------------------------------------------------------
# Microwell grid detection (bright-field channel)
# ---------------------------------------------------------------------------
GRID_SIZE = 10                          # 10 x 10 wells
CROP_OUTPUT_SIZE = 640                  # cropped grid is resized to this square
DEFAULT_WELL_RADIUS_PX = 18

HOUGH_DP = 1.2
HOUGH_MIN_DIST = 35
HOUGH_PARAM1 = 55
HOUGH_PARAM2 = 15
HOUGH_MIN_RADIUS = 15
HOUGH_MAX_RADIUS = 20

# Region of the raw frame where wells are searched (raw image pixels)
ROI_MIN_X, ROI_MAX_X = 250, 1050
ROI_MIN_Y, ROI_MAX_Y = 150, 900

NOISE_MIN_DISTANCE = 45                 # min distance between two accepted circles
SKEW_ROW_TOLERANCE = 20
GROUP_TOLERANCE = 10
GROUP_MIN_POINTS = 3
GROUP_MAX_GAP = 65
GRID_MIN_STEP = 40
CROP_PADDING_EXTRA = 15

# ---------------------------------------------------------------------------
# Physical calibration
# ---------------------------------------------------------------------------
BASE_PX_PER_UM = 0.775                  # px/µm of the 640 px crop at nominal span
WELL_RADIUS_UM = 20.0
WELL_PITCH_UM = 70.0
GRID_SPAN_UM = WELL_PITCH_UM * (GRID_SIZE - 1)   # 630 µm centre-to-centre
SCALE_BAR_UM = 200

# Mode-2 well/cell association: cell box centre must be within this fraction of radius
MODE2_CENTER_DISTANCE_FACTOR = 1.0

# Mode-1 comparison: max distance (px) for matching wells between two samples
COMPARE_WELL_MATCH_DIST = 15

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
GRID_COLUMNS = 4

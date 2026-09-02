"""Central configuration for the Face Recognition Attendance System."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
FACES_DIR: Path = DATA_DIR / "faces"
TRAINER_DIR: Path = DATA_DIR / "trainer"
EXPORTS_DIR: Path = PROJECT_ROOT / "exports"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

DB_PATH: Path = DATA_DIR / "attendance.db"
UNKNOWNS_DIR: Path = DATA_DIR / "unknowns"
MODEL_PATH: Path = TRAINER_DIR / "lbph_model.yml"
LABEL_MAP_PATH: Path = TRAINER_DIR / "label_map.json"
# Written when students change so the UI/CLI can prompt a retrain.
MODEL_STALE_FLAG: Path = TRAINER_DIR / ".model_stale"
# Shipped with the app so OpenCV 5 / slim wheels still detect faces.
HAAR_CASCADE_NAME: str = "haarcascade_frontalface_default.xml"
BUNDLED_HAAR_CASCADE: Path = ASSETS_DIR / HAAR_CASCADE_NAME

# Make sure required directories exist on first run
for _dir in (DATA_DIR, FACES_DIR, TRAINER_DIR, EXPORTS_DIR, ASSETS_DIR, UNKNOWNS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Face recognition tuning
# ---------------------------------------------------------------------------
# How many face samples to capture per student during registration.
SAMPLES_PER_STUDENT: int = 60

# Detected faces are resized to this square shape before training/prediction.
FACE_IMAGE_SIZE: tuple[int, int] = (200, 200)

# LBPH confidence threshold. LOWER distance = better match in OpenCV's LBPH.
# Anything above this value is treated as "Unknown".
CONFIDENCE_THRESHOLD: float = 70.0
# Backward-compatible alias used by 1.x callers.
RECOGNITION_THRESHOLD: float = CONFIDENCE_THRESHOLD
# Distances in (threshold - margin, threshold] are too close to call → Unknown.
AMBIGUOUS_MARGIN: float = 8.0

# Minimum seconds between consecutive attendance marks for the same student.
# Prevents duplicate entries while a student stands in front of the camera.
ATTENDANCE_COOLDOWN_SECONDS: int = 30

# Live-camera liveness: require the face-box center to move this many pixels
# (max-min on x or y) across the last N samples before the first mark.
LIVENESS_HISTORY: int = 12
LIVENESS_MIN_MOTION_PX: int = 16

# Minimum seconds between unknown-face log rows (live camera is ~25 fps).
UNKNOWN_LOG_COOLDOWN_SECONDS: int = 5

# Image suffixes accepted by folder enrollment.
ENROLL_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)
# Folder enrollment skips samples smaller than this, or nearly-black frames.
ENROLL_MIN_SIZE: tuple[int, int] = (40, 40)
ENROLL_MIN_MEAN_PIXEL: float = 8.0
# Webcam enrollment skips frames whose Laplacian variance is below this.
ENROLL_MIN_LAPLACIAN: float = 40.0

# ---------------------------------------------------------------------------
# Periods + lateness
# ---------------------------------------------------------------------------
DEFAULT_PERIOD_NAME: str = "Morning"
DEFAULT_PERIOD_START: str = "09:00"  # HH:MM, 24-hour
LATE_GRACE_MINUTES: int = 10
# Seeded into the `periods` table on a fresh database.
PERIODS: list[dict[str, str]] = [
    {"name": "Morning", "start_hhmm": "09:00"},
    {"name": "Afternoon", "start_hhmm": "13:00"},
]

# Default span for weekly_summary / `stats --week`.
WEEK_DAYS_DEFAULT: int = 7

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
APP_NAME: str = "Face Recognition Attendance System"
APP_VERSION: str = "3.0.0"
WINDOW_SIZE: str = "1200x720"

FONT_FAMILY: str = "Segoe UI"

# Palettes. Light is the historical default so existing screenshots stay valid.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "COLOR_PRIMARY": "#2563eb",
        "COLOR_PRIMARY_DARK": "#1d4ed8",
        "COLOR_ACCENT": "#10b981",
        "COLOR_ACCENT_DARK": "#0f9b75",
        "COLOR_DANGER": "#ef4444",
        "COLOR_WARNING": "#f59e0b",
        "COLOR_BG": "#f8fafc",
        "COLOR_SURFACE": "#ffffff",
        "COLOR_TEXT": "#0f172a",
        "COLOR_MUTED": "#64748b",
        "COLOR_BORDER": "#e2e8f0",
        "COLOR_SIDEBAR": "#0f172a",
        "COLOR_SIDEBAR_TEXT": "#f8fafc",
        "COLOR_PREVIEW": "#0f172a",
    },
    "dark": {
        "COLOR_PRIMARY": "#3b82f6",
        "COLOR_PRIMARY_DARK": "#2563eb",
        "COLOR_ACCENT": "#34d399",
        "COLOR_ACCENT_DARK": "#059669",
        "COLOR_DANGER": "#f87171",
        "COLOR_WARNING": "#fbbf24",
        "COLOR_BG": "#0b1220",
        "COLOR_SURFACE": "#1e293b",
        "COLOR_TEXT": "#e2e8f0",
        "COLOR_MUTED": "#94a3b8",
        "COLOR_BORDER": "#334155",
        "COLOR_SIDEBAR": "#020617",
        "COLOR_SIDEBAR_TEXT": "#e2e8f0",
        "COLOR_PREVIEW": "#020617",
    },
}

ACTIVE_THEME: str = "light"

# Color names are installed from THEMES[ACTIVE_THEME] below.
COLOR_PRIMARY: str
COLOR_PRIMARY_DARK: str
COLOR_ACCENT: str
COLOR_ACCENT_DARK: str
COLOR_DANGER: str
COLOR_WARNING: str
COLOR_BG: str
COLOR_SURFACE: str
COLOR_TEXT: str
COLOR_MUTED: str
COLOR_BORDER: str
COLOR_SIDEBAR: str
COLOR_SIDEBAR_TEXT: str
COLOR_PREVIEW: str

_runtime_threshold: float | None = None


def _install_palette(palette: dict[str, str]) -> None:
    globals().update(palette)


def apply_theme(name: str) -> str:
    """Activate a named palette and copy its colors onto this module.

    Unknown names fall back to light. Returns the theme that was applied.
    """
    global ACTIVE_THEME
    key = (name or "light").strip().lower()
    if key not in THEMES:
        key = "light"
    ACTIVE_THEME = key
    _install_palette(THEMES[key])
    return key


def get_theme() -> str:
    return ACTIVE_THEME


def set_confidence_threshold(value: float) -> float:
    """Override the live LBPH distance threshold (lower = stricter)."""
    global CONFIDENCE_THRESHOLD, RECOGNITION_THRESHOLD, _runtime_threshold
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("confidence threshold must be a positive number")
    _runtime_threshold = parsed
    CONFIDENCE_THRESHOLD = parsed
    RECOGNITION_THRESHOLD = parsed
    return parsed


def get_confidence_threshold() -> float:
    if _runtime_threshold is not None:
        return float(_runtime_threshold)
    return float(CONFIDENCE_THRESHOLD)


def reset_runtime_overrides() -> None:
    """Restore theme + threshold defaults. Used by tests."""
    global CONFIDENCE_THRESHOLD, RECOGNITION_THRESHOLD, _runtime_threshold
    _runtime_threshold = None
    CONFIDENCE_THRESHOLD = 70.0
    RECOGNITION_THRESHOLD = 70.0
    apply_theme("light")


# Install the default (light) palette at import time.
apply_theme("light")

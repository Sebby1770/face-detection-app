"""OpenCV LBPH face detection, training, and recognition."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(image: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(image).tobytes())


def sample_hash(image: np.ndarray) -> str:
    """Public alias for the sample sha256 used to skip duplicate captures."""
    return _sha256_array(image)


def preprocess_face(gray: np.ndarray) -> np.ndarray:
    """CLAHE (clip 2.0, 8×8 tiles), then equalizeHist when the mean is very low."""
    if gray is None or getattr(gray, "size", 0) == 0:
        return gray
    image = gray
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(image)
    if float(np.mean(image)) < 40.0:
        out = cv2.equalizeHist(out)
    return out


def is_low_quality_sample(gray: np.ndarray) -> bool:
    """True when a grayscale sample is smaller than 40×40 or nearly black."""
    if gray is None or getattr(gray, "size", 0) == 0:
        return True
    height, width = gray.shape[:2]
    min_h, min_w = config.ENROLL_MIN_SIZE
    if height < min_h or width < min_w:
        return True
    return float(np.mean(gray)) < float(config.ENROLL_MIN_MEAN_PIXEL)


class FaceEngineError(RuntimeError):
    """Raised when the face engine cannot detect, train, or predict."""


def _cascade_candidates() -> list[Path]:
    """Prefer the vendored XML, then whatever this OpenCV wheel shipped."""
    paths: list[Path] = [config.BUNDLED_HAAR_CASCADE]
    haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if haar_dir:
        base = Path(haar_dir)
        paths.append(base / config.HAAR_CASCADE_NAME)
        if base.is_file():
            paths.append(base)
    cv2_root = Path(cv2.__file__).resolve().parent
    paths.extend(
        [
            cv2_root / "data" / config.HAAR_CASCADE_NAME,
            cv2_root / "data" / "haarcascades" / config.HAAR_CASCADE_NAME,
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def load_face_detector() -> tuple[Optional["cv2.CascadeClassifier"], Optional[Path]]:
    """Load a Haar cascade. Returns (classifier, path) or (None, None)."""
    for path in _cascade_candidates():
        if not path.is_file():
            continue
        detector = cv2.CascadeClassifier(str(path))
        if not detector.empty():
            return detector, path
    return None, None


class FaceEngine:
    """Wraps OpenCV's Haar-cascade face detector and LBPH face recognizer."""

    def __init__(self) -> None:
        self.detector, self.cascade_path = load_face_detector()
        # Detection is optional: folder enrollment of pre-cropped faces and
        # LBPH train/predict still work if Haar XML is missing.

        # opencv-contrib-python ships face.LBPHFaceRecognizer_create
        if not hasattr(cv2, "face"):
            raise FaceEngineError(
                "cv2.face is unavailable. Install opencv-contrib-python "
                "(NOT just opencv-python):\n"
                "  pip uninstall -y opencv-python opencv-contrib-python\n"
                "  pip install opencv-contrib-python"
            )
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception as exc:  # noqa: BLE001
            raise FaceEngineError(
                "Could not create LBPHFaceRecognizer. Ensure opencv-contrib-python "
                f"is installed correctly. Underlying error: {exc}"
            ) from exc
        self.label_map: dict[int, int] = {}  # opencv label -> student_id
        self._loaded: bool = False
        self.last_quality_skipped: int = 0

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_faces(
        self,
        gray_image: np.ndarray,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (60, 60),
    ) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) for every detected face."""
        if self.detector is None or self.detector.empty():
            return []
        if gray_image is None or getattr(gray_image, "size", 0) == 0:
            return []
        faces = self.detector.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        return [tuple(map(int, f)) for f in faces]

    @staticmethod
    def crop_and_resize(gray_image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        face = gray_image[y : y + h, x : x + w]
        return cv2.resize(face, config.FACE_IMAGE_SIZE)

    @staticmethod
    def normalize_face(gray_image: np.ndarray) -> np.ndarray:
        """Resize a grayscale crop to the training size."""
        if gray_image is None or getattr(gray_image, "size", 0) == 0:
            raise FaceEngineError("Cannot normalize an empty image.")
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        if gray_image.shape[:2] != config.FACE_IMAGE_SIZE:
            return cv2.resize(gray_image, config.FACE_IMAGE_SIZE)
        return gray_image

    def extract_face(
        self,
        gray_image: np.ndarray,
        detect: bool = True,
        require_face: bool = False,
    ) -> Optional[np.ndarray]:
        """Return a 200×200 face crop, or the whole image if already cropped.

        When ``detect`` is True, Haar is tried first (including a looser pass).
        If nothing is found and ``require_face`` is True, return ``None``
        (webcam capture must not treat the full frame as a face). Otherwise
        the full frame is treated as a pre-cropped sample so folder enrollment
        works without a webcam or on synthetic PNGs.
        """
        if gray_image is None or getattr(gray_image, "size", 0) == 0:
            return None
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        if detect:
            faces = self.detect_faces(gray_image)
            if not faces:
                faces = self.detect_faces(
                    gray_image,
                    scale_factor=1.1,
                    min_neighbors=3,
                    min_size=(30, 30),
                )
            if faces:
                faces.sort(key=lambda b: b[2] * b[3], reverse=True)
                return self.crop_and_resize(gray_image, faces[0])
            if require_face:
                return None

        return self.normalize_face(gray_image)

    def enroll_from_folder(
        self,
        source_dir: Path,
        student_id: int,
        dest_root: Optional[Path] = None,
        max_samples: Optional[int] = None,
        detect: bool = True,
    ) -> int:
        """Load images from ``source_dir``, extract faces, write sample PNGs.

        Images with no detectable face are treated as already-cropped samples.
        Exact duplicates (sha256 of source bytes or of the extracted sample)
        are skipped. Frames smaller than 40×40 or nearly-black (mean pixel < 8)
        are skipped as low quality. Returns the number of samples written.
        Marks the model stale. ``last_quality_skipped`` is the quality skip count.
        """
        source = Path(source_dir)
        self.last_quality_skipped = 0
        if not source.is_dir():
            raise FaceEngineError(f"Photo folder does not exist: {source}")

        dest_root = Path(dest_root) if dest_root else config.FACES_DIR
        dest = dest_root / str(int(student_id))
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FaceEngineError(f"Could not create sample folder {dest}: {exc}") from exc

        if max_samples is None:
            cap: Optional[int] = int(config.SAMPLES_PER_STUDENT)
        elif int(max_samples) <= 0:
            cap = None
        else:
            cap = int(max_samples)

        existing_nums: list[int] = []
        for png in dest.glob("*.png"):
            try:
                existing_nums.append(int(png.stem))
            except ValueError:
                continue
        next_idx = (max(existing_nums) + 1) if existing_nums else 1

        seen_hashes: set[str] = set()
        for existing in dest.glob("*.png"):
            try:
                seen_hashes.add(_sha256_bytes(existing.read_bytes()))
            except OSError:
                continue
            already = cv2.imread(str(existing), cv2.IMREAD_GRAYSCALE)
            if already is not None:
                seen_hashes.add(_sha256_array(already))

        written = 0
        quality_skipped = 0
        paths = sorted(
            p
            for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in config.ENROLL_IMAGE_SUFFIXES
        )
        for img_path in paths:
            try:
                raw = img_path.read_bytes()
            except OSError:
                continue
            src_hash = _sha256_bytes(raw)
            if src_hash in seen_hashes:
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            if is_low_quality_sample(gray):
                quality_skipped += 1
                continue
            face = self.extract_face(gray, detect=detect, require_face=False)
            if face is None:
                continue
            face_hash = _sha256_array(face)
            if face_hash in seen_hashes:
                continue
            out = dest / f"{next_idx:03d}.png"
            if not cv2.imwrite(str(out), face):
                continue
            try:
                seen_hashes.add(_sha256_bytes(out.read_bytes()))
            except OSError:
                pass
            seen_hashes.add(src_hash)
            seen_hashes.add(face_hash)
            next_idx += 1
            written += 1
            if cap is not None and written >= cap:
                break

        self.last_quality_skipped = quality_skipped
        if written == 0:
            extra = (
                f" Skipped {quality_skipped} low-quality image(s)."
                if quality_skipped
                else ""
            )
            raise FaceEngineError(
                f"No usable images found in {source}. "
                "Add PNG/JPG photos (full portraits or already-cropped faces)."
                + extra
            )
        mark_model_stale(f"Folder enrollment wrote {written} samples for id={student_id}")
        return written

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_from_dataset(self, faces_root: Optional[Path] = None) -> int:
        """Walk ``data/faces/<student_id>/*.png`` and (re)train the LBPH model.

        Returns the number of training images used.
        """
        faces_root = Path(faces_root) if faces_root else config.FACES_DIR
        if not faces_root.exists():
            raise FaceEngineError(
                f"Faces directory does not exist: {faces_root}. "
                "Register at least one student first."
            )

        images: list[np.ndarray] = []
        labels: list[int] = []
        label_map: dict[int, int] = {}
        opencv_label = 0
        student_dirs = 0

        for student_dir in sorted(faces_root.iterdir()):
            if not student_dir.is_dir():
                continue
            try:
                student_id = int(student_dir.name)
            except ValueError:
                continue

            sample_count = 0
            sample_paths = sorted(
                p
                for p in student_dir.iterdir()
                if p.is_file() and p.suffix.lower() in config.ENROLL_IMAGE_SUFFIXES
            )
            for img_path in sample_paths:
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                if gray.shape[:2] != config.FACE_IMAGE_SIZE:
                    gray = cv2.resize(gray, config.FACE_IMAGE_SIZE)
                images.append(preprocess_face(gray))
                labels.append(opencv_label)
                sample_count += 1

            if sample_count > 0:
                label_map[opencv_label] = student_id
                opencv_label += 1
                student_dirs += 1

        if not images:
            raise FaceEngineError(
                "No training images found under "
                f"{faces_root}. Register at least one student and capture "
                "face samples first."
            )

        try:
            self.recognizer.train(images, np.array(labels))
        except cv2.error as exc:
            raise FaceEngineError(
                f"LBPH training failed ({student_dirs} students, "
                f"{len(images)} images): {exc}"
            ) from exc

        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.recognizer.write(str(config.MODEL_PATH))
            with open(config.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
        except OSError as exc:
            raise FaceEngineError(
                f"Could not write model files to {config.TRAINER_DIR}: {exc}"
            ) from exc

        self.label_map = label_map
        self._loaded = True
        clear_model_stale_flag()
        return len(images)

    # ------------------------------------------------------------------
    # Loading & prediction
    # ------------------------------------------------------------------
    def load(self) -> bool:
        if not config.MODEL_PATH.exists() or not config.LABEL_MAP_PATH.exists():
            return False
        try:
            self.recognizer.read(str(config.MODEL_PATH))
            with open(config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.label_map = {int(k): int(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError, cv2.error, ValueError) as exc:
            raise FaceEngineError(
                f"Failed to load trained model from {config.MODEL_PATH}: {exc}. "
                "Retrain with: python main.py train"
            ) from exc
        self._loaded = True
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self,
        face_image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[Optional[int], float]:
        """Return (student_id, confidence). student_id is None if unknown.

        OpenCV LBPH confidence: lower = more confident. We treat anything
        above ``CONFIDENCE_THRESHOLD`` as unknown. Distances in the
        ``AMBIGUOUS_MARGIN`` band just below the limit are also unknown.

        ``threshold`` overrides the live setting for this call only.
        """
        if not self._loaded:
            raise FaceEngineError(
                "Face engine has no model loaded. Train first with "
                "'python main.py train' or the Train Model page."
            )
        if face_image is None or getattr(face_image, "size", 0) == 0:
            raise FaceEngineError("Cannot predict on an empty face image.")
        if face_image.shape[:2] != config.FACE_IMAGE_SIZE:
            face_image = cv2.resize(face_image, config.FACE_IMAGE_SIZE)
        face_image = preprocess_face(face_image)

        try:
            opencv_label, confidence = self.recognizer.predict(face_image)
        except cv2.error as exc:
            raise FaceEngineError(
                f"Recognition failed: {exc}. The model may be corrupt — retrain."
            ) from exc

        limit = (
            float(threshold)
            if threshold is not None
            else config.get_confidence_threshold()
        )
        margin = float(getattr(config, "AMBIGUOUS_MARGIN", 8.0) or 0.0)
        if confidence > (limit - margin):
            return None, float(confidence)
        student_id = self.label_map.get(int(opencv_label))
        return student_id, float(confidence)

    # ------------------------------------------------------------------
    # Staleness helpers
    # ------------------------------------------------------------------
    def model_is_stale(self) -> bool:
        """True if students changed after the last train (flag file present)."""
        return model_is_stale()

    def invalidate_model_if_needed(
        self,
        student_ids_on_disk: Optional[set[int]] = None,
    ) -> bool:
        """Mark the model stale when label map no longer matches face folders.

        Returns True if the model was (or already is) considered stale.
        """
        return invalidate_model_if_needed(student_ids_on_disk=student_ids_on_disk)


# ---------------------------------------------------------------------------
# Module-level helpers (usable from CLI without instantiating FaceEngine)
# ---------------------------------------------------------------------------
def mark_model_stale(reason: str = "") -> None:
    """Create/update the stale flag so callers know a retrain is needed."""
    config.TRAINER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stale": True,
        "reason": reason,
        "marked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.MODEL_STALE_FLAG.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_model_stale_flag() -> None:
    if config.MODEL_STALE_FLAG.exists():
        try:
            config.MODEL_STALE_FLAG.unlink()
        except OSError:
            pass


def model_is_stale() -> bool:
    return config.MODEL_STALE_FLAG.exists()


def _student_ids_from_faces(faces_root: Optional[Path] = None) -> set[int]:
    root = Path(faces_root) if faces_root else config.FACES_DIR
    ids: set[int] = set()
    if not root.exists():
        return ids
    for p in root.iterdir():
        if p.is_dir():
            try:
                ids.add(int(p.name))
            except ValueError:
                continue
    return ids


def _student_ids_from_label_map() -> set[int]:
    if not config.LABEL_MAP_PATH.exists():
        return set()
    try:
        with open(config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(v) for v in raw.values()}
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return set()


def invalidate_model_if_needed(
    student_ids_on_disk: Optional[set[int]] = None,
    faces_root: Optional[Path] = None,
) -> bool:
    """Detect a mismatch between trained labels and current face folders.

    Also treats a missing model with existing face data, or a flag file, as
    stale. Writes ``MODEL_STALE_FLAG`` when stale. Returns True if stale.
    """
    if model_is_stale():
        return True

    disk_ids = (
        student_ids_on_disk
        if student_ids_on_disk is not None
        else _student_ids_from_faces(faces_root)
    )
    label_ids = _student_ids_from_label_map()
    model_exists = config.MODEL_PATH.exists() and config.LABEL_MAP_PATH.exists()

    if not model_exists:
        if disk_ids:
            mark_model_stale("Face samples exist but no trained model is present.")
            return True
        return False

    if disk_ids != label_ids:
        missing = disk_ids - label_ids
        extra = label_ids - disk_ids
        parts = []
        if missing:
            parts.append(f"new/untrained student folders: {sorted(missing)}")
        if extra:
            parts.append(f"deleted students still in model: {sorted(extra)}")
        mark_model_stale("; ".join(parts) or "label map mismatch")
        return True

    # mtime heuristic: any face sample newer than the model file
    try:
        model_mtime = config.MODEL_PATH.stat().st_mtime
        root = Path(faces_root) if faces_root else config.FACES_DIR
        for student_dir in root.iterdir() if root.exists() else []:
            if not student_dir.is_dir():
                continue
            for img in student_dir.iterdir():
                if not img.is_file() or img.suffix.lower() not in config.ENROLL_IMAGE_SUFFIXES:
                    continue
                if img.stat().st_mtime > model_mtime:
                    mark_model_stale(
                        f"Face sample newer than model: {img.name}"
                    )
                    return True
    except OSError:
        pass

    return False

"""Session analytics for face-detection runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class SessionAnalytics:
    """Accumulate per-session face-detection statistics."""

    started_at: float = field(default_factory=time.time)
    total_frames: int = 0
    faces_per_frame: List[int] = field(default_factory=list)
    unique_track_ids: Set[int] = field(default_factory=set)
    peak_concurrent_faces: int = 0
    _score_sum: float = 0.0
    _score_count: int = 0
    ended_at: Optional[float] = None

    def update(self, face_count: int, scores: Sequence[float], track_ids: Sequence[int]) -> None:
        """Record one processed frame."""
        self.total_frames += 1
        self.faces_per_frame.append(face_count)
        if face_count > self.peak_concurrent_faces:
            self.peak_concurrent_faces = face_count
        for score in scores:
            self._score_sum += float(score)
            self._score_count += 1
        for tid in track_ids:
            self.unique_track_ids.add(int(tid))

    def finish(self) -> None:
        """Mark the session end time (idempotent)."""
        if self.ended_at is None:
            self.ended_at = time.time()

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def average_faces_per_frame(self) -> float:
        if not self.faces_per_frame:
            return 0.0
        return sum(self.faces_per_frame) / len(self.faces_per_frame)

    @property
    def average_confidence(self) -> float:
        if self._score_count == 0:
            return 0.0
        return self._score_sum / self._score_count

    @property
    def unique_id_count(self) -> int:
        return len(self.unique_track_ids)

    def summary_dict(self) -> Dict[str, Any]:
        """Serializable summary suitable for JSON export."""
        self.finish()
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_frames": self.total_frames,
            "peak_concurrent_faces": self.peak_concurrent_faces,
            "unique_track_ids": sorted(self.unique_track_ids),
            "unique_id_count": self.unique_id_count,
            "average_faces_per_frame": round(self.average_faces_per_frame, 4),
            "average_confidence": round(self.average_confidence, 4),
            "faces_per_frame": self.faces_per_frame,
        }

    def export_json(self, path: Path | str) -> Path:
        """Write session stats to a JSON file and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary_dict()
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out

    def hud_lines(self) -> List[str]:
        """Short lines for the on-screen analytics HUD."""
        return [
            f"Peak faces: {self.peak_concurrent_faces}",
            f"Unique IDs: {self.unique_id_count}",
        ]


def load_stats(path: Path | str) -> Dict[str, Any]:
    """Load previously exported stats JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

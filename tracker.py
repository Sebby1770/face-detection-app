"""Greedy IoU tracker with EMA smoothing for stable face IDs and jitter-free boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from detectors import Bbox, Detection


@dataclass
class Track:
    id: int
    bbox: Bbox
    score: float
    landmarks: List[Tuple[float, float]] = field(default_factory=list)
    age: int = 0
    misses: int = 0

    @property
    def bbox_int(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return int(x), int(y), int(w), int(h)

    @classmethod
    def from_detection(cls, det: Detection, track_id: int = 0) -> "Track":
        return cls(
            id=track_id,
            bbox=det.bbox,
            score=det.score,
            landmarks=[(float(lx), float(ly)) for lx, ly in det.landmarks],
        )


def _iou(a: Bbox, b: Bbox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    if not inter:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 8,
        ema: float = 0.5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.ema = ema
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        pairs: List[Tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = _iou(track.bbox, det.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._merge(self.tracks[ti], detections[di])

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            self.tracks.append(Track.from_detection(det, track_id=self._next_id))
            self._next_id += 1

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.misses += 1
                track.age += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def _merge(self, track: Track, det: Detection) -> None:
        a = self.ema
        ox, oy, ow, oh = track.bbox
        track.bbox = (
            (1 - a) * ox + a * det.x,
            (1 - a) * oy + a * det.y,
            (1 - a) * ow + a * det.w,
            (1 - a) * oh + a * det.h,
        )
        track.score = (1 - a) * track.score + a * det.score
        if det.landmarks:
            if track.landmarks and len(track.landmarks) == len(det.landmarks):
                track.landmarks = [
                    ((1 - a) * tx + a * dx, (1 - a) * ty + a * dy)
                    for (tx, ty), (dx, dy) in zip(track.landmarks, det.landmarks)
                ]
            else:
                track.landmarks = [(float(x), float(y)) for x, y in det.landmarks]
        track.age += 1
        track.misses = 0

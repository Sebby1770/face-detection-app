"""Unit tests for session analytics (no camera required)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from analytics import SessionAnalytics, load_stats


def test_empty_session_defaults() -> None:
    stats = SessionAnalytics()
    assert stats.total_frames == 0
    assert stats.peak_concurrent_faces == 0
    assert stats.unique_id_count == 0
    assert stats.average_faces_per_frame == 0.0
    assert stats.average_confidence == 0.0
    assert stats.duration_seconds >= 0.0


def test_update_accumulates_frames_and_faces() -> None:
    stats = SessionAnalytics()
    stats.update(face_count=2, scores=[0.9, 0.8], track_ids=[1, 2])
    stats.update(face_count=1, scores=[0.7], track_ids=[1])
    stats.update(face_count=3, scores=[0.6, 0.5, 0.4], track_ids=[1, 3, 4])

    assert stats.total_frames == 3
    assert stats.faces_per_frame == [2, 1, 3]
    assert stats.peak_concurrent_faces == 3
    assert stats.unique_track_ids == {1, 2, 3, 4}
    assert stats.unique_id_count == 4
    assert abs(stats.average_faces_per_frame - 2.0) < 1e-9
    # (0.9+0.8+0.7+0.6+0.5+0.4) / 6 = 0.65
    assert abs(stats.average_confidence - 0.65) < 1e-9


def test_peak_is_max_not_last() -> None:
    stats = SessionAnalytics()
    stats.update(5, [0.9] * 5, list(range(5)))
    stats.update(1, [0.5], [0])
    assert stats.peak_concurrent_faces == 5


def test_finish_sets_ended_at() -> None:
    stats = SessionAnalytics()
    assert stats.ended_at is None
    stats.finish()
    assert stats.ended_at is not None
    ended = stats.ended_at
    stats.finish()  # idempotent
    assert stats.ended_at == ended


def test_duration_advances() -> None:
    stats = SessionAnalytics(started_at=time.time() - 1.5)
    stats.finish()
    assert stats.duration_seconds >= 1.0


def test_export_json(tmp_path: Path) -> None:
    stats = SessionAnalytics()
    stats.update(2, [0.9, 0.8], [10, 11])
    out = tmp_path / "session.json"
    written = stats.export_json(out)
    assert written == out
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_frames"] == 1
    assert data["peak_concurrent_faces"] == 2
    assert data["unique_id_count"] == 2
    assert data["unique_track_ids"] == [10, 11]
    assert "duration_seconds" in data
    assert data["faces_per_frame"] == [2]
    assert abs(data["average_confidence"] - 0.85) < 1e-6


def test_load_stats(tmp_path: Path) -> None:
    stats = SessionAnalytics()
    stats.update(1, [0.99], [42])
    path = stats.export_json(tmp_path / "out.json")
    loaded = load_stats(path)
    assert loaded["unique_track_ids"] == [42]


def test_hud_lines() -> None:
    stats = SessionAnalytics()
    stats.update(3, [0.5, 0.5, 0.5], [1, 2, 3])
    lines = stats.hud_lines()
    assert any("Peak faces: 3" in line for line in lines)
    assert any("Unique IDs: 3" in line for line in lines)


def test_summary_dict_rounds() -> None:
    stats = SessionAnalytics()
    stats.update(1, [1 / 3], [1])
    summary = stats.summary_dict()
    assert isinstance(summary["average_confidence"], float)
    assert summary["ended_at"] is not None

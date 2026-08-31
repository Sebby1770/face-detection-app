from pathlib import Path

import cv2
import numpy as np

from batch import (
    coverage_entry,
    coverage_totals,
    print_coverage_summary,
    process_tree,
    write_coverage,
)
from detectors import Detection
from face_detection import coverage_from_stats
from media import MediaStats
from pipeline import FrameProcessor, RedactionConfig


class StaticDetector:
    name = "synthetic"

    def detect(self, _frame):
        return [Detection(8, 8, 8, 8, 1.0)]


def factory():
    return FrameProcessor(
        StaticDetector(),
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
    )


class EmptyDetector:
    name = "synthetic"

    def detect(self, _frame):
        return []


def empty_factory():
    return FrameProcessor(
        EmptyDetector(),
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
    )


def test_process_tree_two_synthetic_images(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    nested = input_dir / "sub"
    nested.mkdir(parents=True)
    assert cv2.imwrite(str(input_dir / "a.png"), np.full((32, 32, 3), 200, dtype=np.uint8))
    assert cv2.imwrite(str(nested / "b.png"), np.full((32, 32, 3), 180, dtype=np.uint8))

    coverage = process_tree(input_dir, output_dir, factory)

    written_a = cv2.imread(str(output_dir / "a.png"))
    written_b = cv2.imread(str(output_dir / "sub" / "b.png"))
    assert written_a is not None and written_b is not None
    assert np.all(written_a[8:16, 8:16] == 0)
    assert np.all(written_a[:4] == 200)
    assert np.all(written_b[8:16, 8:16] == 0)
    assert coverage["totals"]["files"] == 2
    assert coverage["totals"]["faces_seen"] == 2
    assert coverage["totals"]["unique_ids"] == 2
    assert coverage["totals"]["frames"] == 2
    assert coverage["totals"]["files_with_no_faces"] == 0
    assert coverage["totals"]["skipped"] == 0
    assert coverage["skipped"] == []
    paths = {entry["path"] for entry in coverage["files"]}
    assert paths == {"a.png", "sub/b.png"}
    assert all(entry["kind"] == "image" for entry in coverage["files"])
    assert all(entry["frames"] == 1 for entry in coverage["files"])
    assert all(entry["miss_frame_rate"] == 0.0 for entry in coverage["files"])
    assert all(entry["files_with_no_faces"] == 0 for entry in coverage["files"])
    assert "warning" not in coverage["totals"]


def test_process_tree_skips_existing_unless_overwrite(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "a.png"), np.full((32, 32, 3), 200, dtype=np.uint8))
    existing = np.full((32, 32, 3), 15, dtype=np.uint8)
    assert cv2.imwrite(str(output_dir / "a.png"), existing)

    skipped = process_tree(input_dir, output_dir, factory, overwrite=False)
    assert skipped["totals"]["files"] == 0
    assert skipped["skipped"] == ["a.png"]
    assert skipped["totals"]["skipped"] == 1
    assert np.array_equal(cv2.imread(str(output_dir / "a.png")), existing)

    rewritten = process_tree(input_dir, output_dir, factory, overwrite=True)
    assert rewritten["totals"]["files"] == 1
    assert rewritten["skipped"] == []
    assert rewritten["totals"]["skipped"] == 0
    assert np.all(cv2.imread(str(output_dir / "a.png"))[8:16, 8:16] == 0)


def test_batch_existing_output_without_overwrite_records_skipped(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "a.png"), np.full((32, 32, 3), 200, dtype=np.uint8))
    assert cv2.imwrite(str(input_dir / "b.png"), np.full((32, 32, 3), 180, dtype=np.uint8))
    existing = np.full((32, 32, 3), 15, dtype=np.uint8)
    assert cv2.imwrite(str(output_dir / "a.png"), existing)

    coverage = process_tree(input_dir, output_dir, factory, overwrite=False)

    assert coverage["skipped"] == ["a.png"]
    assert coverage["totals"]["skipped"] == 1
    assert coverage["totals"]["files"] == 1
    assert {entry["path"] for entry in coverage["files"]} == {"b.png"}
    assert np.array_equal(cv2.imread(str(output_dir / "a.png")), existing)
    written_b = cv2.imread(str(output_dir / "b.png"))
    assert written_b is not None
    assert np.all(written_b[8:16, 8:16] == 0)


def test_coverage_entry_flags_empty_files_and_miss_rate():
    stats = MediaStats()
    stats.frames_processed = 10
    stats.face_observations = 0
    stats.empty_frames = 4
    entry = coverage_entry("empty.mp4", "video", stats)

    assert entry["files_with_no_faces"] == 1
    assert entry["miss_frame_rate"] == 0.4
    totals = coverage_totals([entry], skipped=["kept.png"])
    assert totals["files_with_no_faces"] == 1
    assert totals["unique_ids"] == 0
    assert totals["frames"] == 10
    assert totals["skipped"] == 1
    assert totals["miss_frame_rate"] == 0.4
    assert totals["warning"] == "high empty-frame rate; faces may have been missed"

    coverage = coverage_from_stats("blank.png", "image", stats)
    assert coverage["skipped"] == []
    assert coverage["totals"]["skipped"] == 0
    assert coverage["totals"]["files_with_no_faces"] == 1
    assert coverage["totals"]["frames"] == 10
    assert coverage["files"][0]["miss_frame_rate"] == 0.4


def test_process_tree_records_files_with_no_faces(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "blank.png"), np.full((32, 32, 3), 200, dtype=np.uint8))

    coverage = process_tree(input_dir, output_dir, empty_factory)

    assert coverage["totals"]["files"] == 1
    assert coverage["totals"]["faces_seen"] == 0
    assert coverage["totals"]["unique_ids"] == 0
    assert coverage["totals"]["files_with_no_faces"] == 1
    assert coverage["totals"]["empty_frames"] == 1
    assert coverage["files"][0]["files_with_no_faces"] == 1
    assert coverage["files"][0]["miss_frame_rate"] == 1.0
    assert coverage["totals"]["warning"] == "high empty-frame rate; faces may have been missed"


def test_coverage_totals_warns_at_quarter_miss_rate():
    warned = coverage_totals(
        [{"frames": 4, "empty_frames": 1, "faces": 3, "files_with_no_faces": 0}]
    )
    assert warned["miss_frame_rate"] == 0.25
    assert warned["warning"] == "high empty-frame rate; faces may have been missed"

    ok = coverage_totals(
        [{"frames": 5, "empty_frames": 1, "faces": 4, "files_with_no_faces": 0}]
    )
    assert ok["miss_frame_rate"] == 0.2
    assert "warning" not in ok


def test_process_tree_prints_coverage_and_warning(tmp_path: Path, capsys):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "blank.png"), np.full((32, 32, 3), 200, dtype=np.uint8))

    coverage = process_tree(input_dir, output_dir, empty_factory)
    captured = capsys.readouterr()

    assert "Coverage:" in captured.out
    assert "miss rate" in captured.out
    assert captured.err.strip() == coverage["totals"]["warning"]
    assert "faces may have been missed" in captured.err


def test_print_coverage_summary_writes_warning_to_stderr(capsys):
    coverage = {
        "totals": {
            "files": 1,
            "faces_seen": 0,
            "unique_ids": 0,
            "empty_frames": 4,
            "files_with_no_faces": 1,
            "miss_frame_rate": 0.4,
            "warning": "high empty-frame rate; faces may have been missed",
        }
    }
    print_coverage_summary(coverage)
    captured = capsys.readouterr()
    assert "Coverage:" in captured.out
    assert "0 id observation(s)" in captured.out
    assert "1 file(s) with no faces" in captured.out
    assert "miss rate 40%" in captured.out
    assert "high empty-frame rate; faces may have been missed" in captured.err


def test_coverage_totals_sums_unique_ids():
    totals = coverage_totals(
        [
            {"frames": 2, "empty_frames": 0, "faces": 3, "unique_ids": 2},
            {"frames": 1, "empty_frames": 0, "faces": 1, "unique_ids": 1},
        ]
    )
    assert totals["unique_ids"] == 3
    assert totals["files_with_no_faces"] == 0


def test_print_coverage_summary_includes_id_observations_and_empty_files(capsys):
    coverage = {
        "totals": {
            "files": 3,
            "faces_seen": 4,
            "unique_ids": 5,
            "empty_frames": 2,
            "files_with_no_faces": 1,
            "miss_frame_rate": 0.1,
        }
    }
    print_coverage_summary(coverage)
    out = capsys.readouterr().out
    assert "5 id observation(s)" in out
    assert "1 file(s) with no faces" in out


def test_write_coverage_round_trip(tmp_path: Path):
    path = tmp_path / "reports" / "coverage.json"
    payload = {"files": [], "totals": {"files": 0, "faces_seen": 0, "empty_frames": 0}}
    written = write_coverage(path, payload)
    assert written == path
    assert path.exists()

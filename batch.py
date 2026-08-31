"""Recurse an input tree of images/videos and write a mirrored redacted tree."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from media import (
    VIDEO_OUTPUT_SUFFIXES,
    MediaError,
    MediaPipeline,
    MediaStats,
    media_kind,
)
from pipeline import FrameProcessor

ProcessorFactory = Callable[[], FrameProcessor]


def iter_media_files(root: Path) -> list[Path]:
    """Return media files under ``root`` in deterministic relative-path order."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            media_kind(path)
        except MediaError:
            continue
        files.append(path)
    return files


def coverage_entry(relative_path: str, kind: str, stats: MediaStats) -> dict:
    frames = int(stats.frames_processed)
    empty_frames = int(stats.empty_frames)
    faces = int(stats.face_observations)
    return {
        "path": relative_path,
        "kind": kind,
        "frames": frames,
        "faces": faces,
        "unique_ids": stats.unique_id_count,
        "empty_frames": empty_frames,
        "files_with_no_faces": int(faces == 0),
        "miss_frame_rate": empty_frames / max(frames, 1),
    }


def coverage_totals(
    files: list[dict],
    skipped: Optional[Sequence[str]] = None,
) -> dict:
    skipped_paths = list(skipped) if skipped is not None else []
    frames = sum(int(entry.get("frames", 0)) for entry in files)
    empty_frames = sum(int(entry.get("empty_frames", 0)) for entry in files)
    miss_frame_rate = empty_frames / max(frames, 1)
    totals = {
        "files": len(files),
        "faces_seen": sum(int(entry.get("faces", 0)) for entry in files),
        "unique_ids": sum(int(entry.get("unique_ids", 0)) for entry in files),
        "empty_frames": empty_frames,
        "frames": frames,
        "skipped": len(skipped_paths),
        "files_with_no_faces": sum(1 for entry in files if int(entry.get("faces", 0)) == 0),
        "miss_frame_rate": miss_frame_rate,
    }
    if miss_frame_rate >= 0.25:
        totals["warning"] = "high empty-frame rate; faces may have been missed"
    return totals


def print_coverage_summary(coverage: dict, output: Optional[Path] = None) -> None:
    """Print one coverage line to stdout; emit a miss-rate warning to stderr."""
    totals = coverage.get("totals", coverage)
    destination = f" -> {output}" if output is not None else ""
    miss = float(totals.get("miss_frame_rate", 0.0))
    print(
        f"Coverage: {int(totals.get('files', 0))} file(s), "
        f"{int(totals.get('faces_seen', 0))} face observation(s), "
        f"{int(totals.get('unique_ids', 0))} id observation(s), "
        f"{int(totals.get('empty_frames', 0))} empty frame(s), "
        f"{int(totals.get('files_with_no_faces', 0))} file(s) with no faces, "
        f"miss rate {miss:.0%}{destination}"
    )
    warning = totals.get("warning")
    if warning:
        print(warning, file=sys.stderr)


def write_coverage(path: Path | str, coverage: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    return out


def _destination_for(src: Path, dest: Path, kind: str) -> Path:
    if kind == "video" and dest.suffix.lower() not in VIDEO_OUTPUT_SUFFIXES:
        return dest.with_suffix(".mp4")
    return dest


def process_tree(
    input_dir: Path | str,
    output_dir: Path | str,
    processor_factory: ProcessorFactory,
    overwrite: bool = False,
    include_overlays: bool = False,
    frame_transform=None,
    max_frames: Optional[int] = None,
) -> dict:
    """Redact every image and video under ``input_dir``, mirroring relative paths.

    Existing outputs are skipped unless ``overwrite`` is true; skipped relative
    paths are listed in the returned coverage. Each file gets a fresh processor
    from ``processor_factory`` so track IDs restart per clip.
    """
    root = Path(input_dir)
    destination_root = Path(output_dir)
    if not root.is_dir():
        raise MediaError(f"input is not a directory: {root}")
    if destination_root.exists() and not destination_root.is_dir():
        raise MediaError(f"output is not a directory: {destination_root}")
    if destination_root.exists() and root.resolve() == destination_root.resolve():
        raise MediaError("input and output directories must be different")

    destination_root.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    skipped: list[str] = []

    for src in iter_media_files(root):
        relative = src.relative_to(root)
        kind = media_kind(src)
        dest = _destination_for(src, destination_root / relative, kind)
        if dest.exists() and not overwrite:
            skipped.append(relative.as_posix())
            continue

        processor = processor_factory()
        pipeline = MediaPipeline(processor)
        if kind == "image":
            _result, stats = pipeline.process_image(
                src,
                output_path=dest,
                include_overlays=include_overlays,
                frame_transform=frame_transform,
            )
        else:
            stats = pipeline.process_capture(
                str(src),
                output_path=dest,
                include_overlays=include_overlays,
                frame_transform=frame_transform,
                max_frames=max_frames,
            )
        files.append(coverage_entry(relative.as_posix(), kind, stats))

    coverage = {
        "files": files,
        "skipped": skipped,
        "totals": coverage_totals(files, skipped=skipped),
    }
    print_coverage_summary(coverage, destination_root)
    return coverage

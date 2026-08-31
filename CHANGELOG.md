# Changelog

## 2.3.1 — 2026-08-31

### Added
- After a run, print the track IDs that were seen so `--keep-ids` / `--redact-ids` is guesswork-free.

## 2.3.0 — 2026-08-31

### Added

- Review HUD lists current track IDs (`IDs: 1,2,5`) so `--keep-ids` / `--redact-ids` can be chosen from the preview window
- Held tracks with detector misses show `hold #N m/H` using `--hold-frames` (for example `hold #3 1/2`)
- Preview key `f` toggles feather between 0 and 8; still images reprocess from the untouched source
- `--min-size N` drops detections whose width or height is below N pixels before tracking (default 0)
- Coverage summary and totals include unique ID observations (`id observations`) and files with no faces

### Privacy

- `--min-size` can hide small true faces as well as noise. Review output before sharing, especially when IDs are used to keep a host visible.

## 2.2.0 — 2026-08-31

### Added

- Landmark-aware ellipse masks: with YuNet's 5 points (eyes, nose, mouth corners), the ellipse expands ~0.55× bbox height above the eyes and ~0.35× below the mouth, plus 0.25× bbox width on the landmark x-range
- Preview keys: `e` toggles box/ellipse, `p` toggles pixelate (same remember-previous-mode behavior as `b` / blur)
- Empty-frame HUD line `NO FACE this frame` when a preview frame has no tracks
- Coverage totals include `warning` when `miss_frame_rate >= 0.25`; batch and CLI print a coverage summary and emit that warning to stderr

### Privacy

- Landmark ellipses still depend on a correct detection. Review output; a high empty-frame rate is a miss signal, not a guarantee that the media had no faces.

## 2.1.0 — 2026-08-31

### Fixed

- Still-image preview no longer double-counts the first frame in session analytics (`--export-stats`)
- `--identify` now errors instead of implying face-crop export is available

### Changed

- Ellipse masks stretch 1.20× vertically and 1.08× horizontally, clamped to the frame
- Coverage JSON records skipped existing outputs, files with no faces, miss-frame rate, and total frames
- Batch prints how many files were skipped and mentions `--overwrite`

### Privacy

- This release does not export face crops. Use `--review` then `--keep-ids` / `--redact-ids`.

## 2.0.0 — 2026-08-31

Privacy-first Face Privacy Redactor 2.0. Defaults stay a solid black box, 25% padding, two-frame hold, and no diagnostic overlays on saved media.

### Added

- Ellipse masks (`--mask ellipse`) and optional feathered edges (`--feather N`)
- Directory batch redaction that mirrors relative paths, plus `--coverage` JSON
- Selective `--redact-ids` / `--keep-ids` (keep wins when an ID is in both lists)
- `--review` window HUD, `--export-stats` session JSON, and optional pose yaw indicators
- Positional source path as an alias of `--input`
- `--pixelate` shortcut alongside `--blur`
- Packaging metadata, `face-detection` console script, and CI on Python 3.9 and 3.12

### Changed

- Interactive HUD can show peak/unique counts; recordings and snapshots remain clean unless `--overlays` is set

### Privacy

- Face crops are not exported. `--identify` is reserved and does not enable a gallery in this release.
- Detection can still miss faces. Review output before sharing.

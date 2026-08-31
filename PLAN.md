# Face Privacy Redactor 2.2 — Implementation Plan

**From:** 2.1.0  
**To:** 2.2.0 — landmark-aware masks, preview keys for ellipse/pixelate, honest empty-frame HUD, batch miss warnings

No web UI, no new models.

## Why

Boxes and even padded ellipses still miss forehead and jaw when YuNet returns 5 landmarks. Reviewers also cannot switch mask/mode without restarting. Batch runs stay quiet when most frames had no face.

## Work packages

1. **Landmark hull** — if a track has 5 landmarks (eyes, nose, mouth corners), expand the redaction region to include forehead (~0.55× bbox height above the eyes) and chin (~0.35× below the mouth), then ellipse-fill that region. If landmarks are missing, keep current ellipse/box path. Box + feather 0 tests must still pass (landmark path only when `shape=="ellipse"` and landmarks exist, or when a new `landmark_aware=True` default on ellipse).
2. **Preview keys** — `e` cycles box/ellipse; `p` toggles pixelate like `b` toggles blur. Update HELP_LINES.
3. **Empty-frame HUD** — when `face_count==0`, show `NO FACE this frame` so reviewers notice dropouts.
4. **Batch honesty** — always print coverage totals; if `miss_frame_rate >= 0.25`, print a warning to stderr that faces may have been missed. Include `warning` on coverage JSON when true.
5. Tests for landmark expansion vs far pixels, `e`/`p` key handlers if easy, coverage warning flag.
6. CHANGELOG / README / version 2.2.0.

---

# Face Detection App 2.0 — Implementation Plan

**From:** privacy-redaction pipeline at `602f479` plus unused `analytics.py` / `pose.py`  
**To:** one privacy-first product with optional review overlays, batch redaction, safer masks, and a coverage report

The GitHub `main` branch is a live-detector with gallery crops. The local rewrite is a redactor. This plan **keeps redaction as the default product** and folds analytics/pose in as opt-in review tools. Gallery crops stay **off** unless `--identify` is passed (explicit, dangerous, documented).

---

## Why this upgrade

People do not open this app to draw boxes on their face. They open it to **share a video without sharing faces**. 2.0 makes that path complete: one file, a folder of files, safer than a rectangle, a JSON audit, and a review HUD you can turn on without burning it into the saved file.

---

## Decision log

| Decision | Choice | Why |
| --- | --- | --- |
| Default product | Solid padded redaction, no overlays on disk | Matches the privacy rewrite |
| Review HUD | Live preview only unless `--overlays` | HUD-on-recordings was a privacy hole |
| Mask shape | `box` (default, tested) plus `ellipse` | Ellipse covers more of the head; box stays the tested default so existing tests still mean something |
| Feather | Optional `--feather N` pixels on ellipse/box | Soft edge without a new model |
| Batch | `python face_detection.py --input DIR --output DIR` | Same binary, no extra command to discover |
| Selective IDs | `--redact-ids` / `--keep-ids` | Interview “host stays visible” |
| Analytics | Session JSON via `--export-stats` | Module already exists |
| Pose | Drawn only with `--overlays` | Never on saved redacted media unless requested |
| Gallery | `--identify --gallery` only | Exporting face crops is identifying |
| Deps | OpenCV + stdlib; pytest in *dev* extra | pytest is not a runtime dependency |

---

## Work packages

### WP1 — Unify the CLI

Keep HEAD flags (`--input`, `--output`, `--headless`, `--redaction`, `--padding`, `--hold-frames`, `--overlays`).

Add:

- `--review` — show boxes/landmarks/IDs/pose in the **window only**
- `--export-stats PATH` — write `analytics.py` JSON
- `--pose` / `--no-pose` (pose requires `--review` or `--overlays`)
- `--mask box|ellipse` (default `box`)
- `--feather N` (default 0)
- `--redact-ids 1,3` / `--keep-ids 2`
- `--identify` — unlock gallery; refuse without it
- Directory `--input` / `--output` for batch
- `--coverage PATH` — JSON sidecar of per-file stats

Positional source (`photo.jpg`) is accepted as an alias of `--input` so GitHub-main muscle memory works.

### WP2 — Safer masks (`pipeline.py`)

`RedactionConfig.shape`: `box` | `ellipse`  
`RedactionConfig.feather`: int ≥ 0

Ellipse: draw a filled ellipse into a mask covering the padded bbox (slightly taller: 1.15× height to include forehead/chin), then apply solid/blur/pixelate through the mask.

Feather: Gaussian-blur the mask, then `output = frame*(1-a) + redacted*a`.

Existing solid-box tests stay on `shape="box", feather=0`.

`FrameProcessor.process` accepts `redact_ids` / `keep_ids` collections.

### WP3 — Batch + coverage

`media.py` / new `batch.py`:

- Recurse images and videos
- Mirror relative paths into `--output`
- Skip existing unless `--overwrite`
- Coverage JSON:

```json
{
  "files": [{"path": "...", "kind": "image", "frames": 1, "faces": 1, "unique_ids": 1, "empty_frames": 0}],
  "totals": {"files": 12, "faces_seen": 40, "empty_frames": 3}
}
```

### WP4 — Preview session

Wire `analytics.SessionAnalytics` into the live loop. HUD can show peak/unique. Recordings and snapshots still use the redacted frame **without** HUD unless `--overlays`.

Pose yaw drawn in review mode via `pose.py`.

### WP5 — Packaging / CI

- `pyproject.toml`: package metadata, `face-detection` script, `[dev]` extra, Python 3.9–3.13
- `requirements.txt`: opencv-python only
- `requirements-dev.txt`: pytest, numpy, opencv-python-headless
- CI: 3.9 and 3.12, headless OpenCV, `pytest -q`, `python face_detection.py --help`

### WP6 — Tests

Keep existing pipeline/media/cli tests. Add:

- Ellipse changes ROI, leaves far pixels
- Feather does not crash at 0 and 8
- `--keep-ids` leaves that track unredacted
- Batch on a temp dir of two synthetic images
- Analytics JSON round-trip (already in `test_analytics.py`)
- `test_tracker.py` remains the HEAD version (first-frame / reset)

### Out of scope

Web UI, ffmpeg audio mux, ReID, SHA256 model pin (nice-to-have if cheap: document checksum in README), painting missed faces by mouse.

---

## Files

- `face_detection.py`, `pipeline.py`, `media.py`, `batch.py` (new)
- `analytics.py`, `pose.py`, `tracker.py`, `detectors.py`
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`
- `tests/*`, `.github/workflows/ci.yml`
- `README.md`

## Acceptance

```
python -m pytest -q
python face_detection.py --help
```

Headless image redaction still writes a file with a solid box by default. Ellipse is opt-in. Batch on a folder produces coverage JSON.

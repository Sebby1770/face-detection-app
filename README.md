# Faces

**Live site:** [https://sebby1770.github.io/face-detection-app/](https://sebby1770.github.io/face-detection-app/)

Detect and **redact** faces, or **recognize** them and mark attendance. One local Python app. Nothing is uploaded.

This repo used to be the privacy redactor only. It now also contains the attendance system that lived in `face-recognition-attendance`.

## Two desks

| Mode | What it does |
| --- | --- |
| **Redact** | YuNet / Haar detection, solid / blur / pixelate masks, webcam, images, video, folders |
| **Attendance** | Enroll students, LBPH recognition, SQLite present/late/excused, kiosk PIN, calendar, local API |

The GitHub Pages demo redacts in the tab (Chrome/Edge face detector) and hosts the attendance desk UI. Point the desk at a local API with `faces attendance serve`.

## Install

```bash
git clone https://github.com/Sebby1770/face-detection-app.git
cd face-detection-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

YuNet downloads into `./models/` on first redact run (`python download_model.py` to pre-fetch).

## CLI

```bash
faces --help

# Privacy redaction
faces redact portrait.jpg --output out.png --headless
faces redact --input interview.mp4 --output out.mp4 --headless --redaction pixelate

# Attendance
faces attendance                         # Tkinter GUI
faces attendance students add --roll 1 --name Ada
faces attendance mark --roll 1
faces attendance report digest --date today
faces attendance serve --port 8768
```

Original entry points still work: `python face_detection.py`, `python -m attendance.cli`.

### Redact flags

See `faces redact --help`. Conservative defaults: solid black masks, 25% padding, two-frame dropout hold. `--keep-ids` leaves a host visible.

### Attendance commands

Same surface as before: `train`, `register-folder`, `import-students`, `mark`, `report`, `calendar`, `streaks`, `holidays`, `alerts`, `backup`, `restore-db`, `doctor`, `export`, PIN kiosk, and the loopback JSON API.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

No camera required. CI uses `opencv-contrib-python-headless`.

## Layout

```
face_detection.py   redact CLI
detectors.py        YuNet + Haar
pipeline.py         redaction + overlay
attendance/         recognition, SQLite, GUI, API
faces.py            unified dispatcher
web/                Pages studio (redact + attendance)
```

## License

MIT

"""2.6 unique period marks, holiday stats, scrypt PIN, doctor, preprocess."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from attendance.database import (
    Database,
    PIN_MAX_FAILS,
    hash_student_pin,
    verify_stored_pin,
)


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    from attendance import config

    data = tmp_path / "data"
    faces = data / "faces"
    trainer = data / "trainer"
    exports = tmp_path / "exports"
    for folder in (faces, trainer, exports):
        folder.mkdir(parents=True, exist_ok=True)
    db_path = data / "api.db"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "FACES_DIR", faces)
    monkeypatch.setattr(config, "TRAINER_DIR", trainer)
    monkeypatch.setattr(config, "EXPORTS_DIR", exports)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "MODEL_PATH", trainer / "lbph_model.yml")
    monkeypatch.setattr(config, "LABEL_MAP_PATH", trainer / "label_map.json")
    monkeypatch.setattr(config, "MODEL_STALE_FLAG", trainer / ".model_stale")
    return Database(db_path=db_path)


def test_unique_per_period(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    morning = datetime(2026, 8, 18, 9, 0, 0)
    assert db.mark_attendance(sid, at=morning, period="Morning", status="Present")
    assert (
        db.mark_attendance(
            sid,
            at=morning + timedelta(seconds=40),
            period="Morning",
            status="Late",
        )
        is False
    )
    assert db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 13, 0, 0),
        period="Afternoon",
        status="Present",
    )
    df = db.get_attendance(date="2026-08-18")
    assert len(df) == 2
    assert set(df["period"]) == {"Morning", "Afternoon"}


def test_holiday_absentees_empty_and_stats_rate(db: Database) -> None:
    db.add_student("R1", "Ada")
    db.add_student("R2", "Bob")
    db.add_holiday("2026-08-19", "Break")

    absentees = db.get_absentees("2026-08-19")
    assert absentees.empty

    digest = db.daily_digest("2026-08-19")
    assert digest["holiday"] is True
    assert digest["holiday_name"] == "Break"
    assert digest["absentee_count"] == 0
    assert digest["attendance_rate"] == pytest.approx(100.0)

    s = db.stats(date="2026-08-19")
    assert s["holiday"] is True
    assert s["holiday_name"] == "Break"
    assert s["present_today"] == 0
    assert s["attendance_rate_today"] == pytest.approx(100.0)


def test_stats_first_mark_wins(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 9, 0, 0),
        period="Morning",
        status="Present",
    )
    db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 13, 20, 0),
        period="Afternoon",
        status="Late",
    )
    s = db.stats(date="2026-08-18")
    assert s["on_time_today"] == 1
    assert s["late_today"] == 0
    assert s["present_today"] == 1


def test_excused_does_not_break_present_streak(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    db.mark_attendance(sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(sid, at=datetime(2026, 8, 18, 9, 0, 0), status="Excused")
    db.mark_attendance(sid, at=datetime(2026, 8, 19, 9, 0, 0), status="Late")
    # Excused is skipped (not a break). If it broke the run this would be 1.
    assert db.present_streak("R1", as_of="2026-08-19") == 2
    assert db.streaks_report(as_of="2026-08-19")[0]["streak"] == 2


def test_scrypt_pin_and_legacy_sha256_verify(db: Database) -> None:
    db.add_student("R1", "Ada")
    db.add_student("R2", "Bob")
    db.set_pin("R1", "1234")
    db.set_pin("R2", "1234")
    stored = str(db.get_student_by_roll("R1")["pin_hash"])
    assert stored.startswith("scrypt$")
    assert "1234" not in stored
    assert hash_student_pin("R1", "1234").startswith("scrypt$")
    assert db.get_student_by_roll("R1")["pin_hash"] != db.get_student_by_roll("R2")[
        "pin_hash"
    ]
    assert db.verify_pin("R1", "1234") is True

    legacy = hashlib.sha256(b"R3:5678").hexdigest()
    db.add_student("R3", "Cara")
    with db._connect() as conn:
        conn.execute(
            "UPDATE students SET pin_hash = ? WHERE roll_number = ?",
            (legacy, "R3"),
        )
    assert verify_stored_pin(legacy, "R3", "5678") is True
    assert db.verify_pin("R3", "5678") is True
    assert db.verify_pin("R3", "0000") is False


def test_pin_lockout_after_five_fails(db: Database) -> None:
    db.add_student("R1", "Ada")
    db.set_pin("R1", "1234")
    for _ in range(PIN_MAX_FAILS):
        assert db.verify_pin("R1", "0000") is False
    assert db.verify_pin("R1", "1234") is False
    with db._connect() as conn:
        conn.execute(
            "UPDATE pin_attempts SET locked_until = ? WHERE roll_number = ?",
            ("2000-01-01T00:00:00", "R1"),
        )
    assert db.verify_pin("R1", "1234") is True
    assert db.verify_pin("R1", "1234") is True


def test_preprocess_face_is_not_identity() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from attendance.face_engine import preprocess_face

    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, :32] = 40
    img[:, 32:] = 200
    img[10:20, 10:20] = 90
    out = preprocess_face(img)
    assert out is not None
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_doctor_cli(isolated_db: Database, capsys: pytest.CaptureFixture) -> None:
    from main import main

    isolated_db.add_student("R1", "Ada")
    isolated_db.add_holiday("2026-12-25", "Xmas")
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "OpenCV" in out
    assert "cv2.face" in out
    assert "Cascade" in out
    assert "Database" in out
    assert "Students" in out
    assert "Sample folders" in out
    assert "Model" in out
    assert "stale" in out.lower()
    assert "Holidays" in out
    assert "1" in out
    try:
        import cv2

        expect = 0 if hasattr(cv2, "face") else 1
    except ImportError:
        expect = 1
    assert rc == expect


def test_extract_face_require_face_blank() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from attendance.face_engine import FaceEngine, FaceEngineError

    try:
        engine = FaceEngine()
    except FaceEngineError as exc:
        pytest.skip(str(exc))
    blank = np.zeros((200, 200), dtype=np.uint8)
    assert engine.extract_face(blank, detect=True, require_face=True) is None
    fallback = engine.extract_face(blank, detect=True, require_face=False)
    assert fallback is not None
    assert fallback.shape[:2] == (200, 200)


def test_camera_open_no_dshow_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    import attendance.camera as camera_mod
    from attendance.camera import Camera

    calls: list[tuple] = []

    class FakeCap:
        def __init__(self, *args: object) -> None:
            calls.append(args)

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            return None

    monkeypatch.setattr(camera_mod.sys, "platform", "darwin")
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeCap)
    cam = Camera(0)
    assert cam.open() is True
    assert calls
    dshow = getattr(camera_mod.cv2, "CAP_DSHOW", object())
    for args in calls:
        assert dshow not in args


def test_api_stats_doctor_and_cors(isolated_db: Database) -> None:
    from attendance.server import make_server

    isolated_db.add_student("S1", "Api Kid")
    isolated_db.add_holiday("2026-08-19", "Break")
    isolated_db.mark_attendance(
        isolated_db.get_student_by_roll("S1")["id"],
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Present",
    )

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["total_students"] == 1
        assert payload["present_today"] in (0, 1)

        with urlopen(f"http://127.0.0.1:{port}/doctor", timeout=5) as resp:
            doctor = json.loads(resp.read().decode("utf-8"))
        assert "cv2_face" in doctor
        assert doctor["student_count"] == 1
        assert doctor["holiday_count"] == 1
        assert "ok" in doctor

        req = Request(
            f"http://127.0.0.1:{port}/health",
            method="OPTIONS",
        )
        with urlopen(req, timeout=5) as resp:
            assert resp.status in (200, 204)
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

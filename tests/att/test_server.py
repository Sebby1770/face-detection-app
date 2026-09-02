"""Local JSON API tests — thread + urllib, no camera."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from attendance.database import Database


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
    return Database(db_path=db_path)


def test_serve_health_students_attendance(isolated_db: Database) -> None:
    isolated_db.add_student("S1", "Api Kid", email="a@x.com", department="CS", section="A")
    isolated_db.mark_attendance(
        isolated_db.get_student_by_roll("S1")["id"],
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Present",
    )

    from attendance.server import make_server

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode("utf-8"))
        assert health["status"] == "ok"

        with urlopen(f"http://127.0.0.1:{port}/students", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["count"] == 1
        assert payload["students"][0]["roll_number"] == "S1"
        assert payload["students"][0]["name"] == "Api Kid"

        url = f"http://127.0.0.1:{port}/attendance?date=2026-08-18"
        with urlopen(url, timeout=5) as resp:
            att = json.loads(resp.read().decode("utf-8"))
        assert att["count"] == 1
        assert att["date"] == "2026-08-18"
        assert att["records"][0]["name"] == "Api Kid"
        assert att["records"][0]["roll_number"] == "S1"

        with pytest.raises(HTTPError) as missing:
            urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

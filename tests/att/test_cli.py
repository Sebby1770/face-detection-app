"""CLI smoke tests that avoid the camera and GUI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from attendance.database import Database


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    """Point config paths at a temp tree so CLI commands never touch real data/."""
    from attendance import config

    data = tmp_path / "data"
    faces = data / "faces"
    trainer = data / "trainer"
    exports = tmp_path / "exports"
    for d in (faces, trainer, exports):
        d.mkdir(parents=True, exist_ok=True)

    db_path = data / "cli.db"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "FACES_DIR", faces)
    monkeypatch.setattr(config, "TRAINER_DIR", trainer)
    monkeypatch.setattr(config, "EXPORTS_DIR", exports)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "MODEL_PATH", trainer / "lbph_model.yml")
    monkeypatch.setattr(config, "LABEL_MAP_PATH", trainer / "label_map.json")
    monkeypatch.setattr(config, "MODEL_STALE_FLAG", trainer / ".model_stale")
    return Database(db_path=db_path)


def test_cli_students_add_and_list(isolated_db: Database, capsys: pytest.CaptureFixture) -> None:
    from main import main

    rc = main(
        [
            "students",
            "add",
            "--roll",
            "C100",
            "--name",
            "CLI User",
            "--section",
            "X",
            "--department",
            "CS",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "C100" in out

    rc = main(["students", "list", "--section", "X"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CLI User" in out
    assert "C100" in out


def test_cli_stats_and_export(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    sid = isolated_db.add_student("S9", "Stats Kid", section="A")
    isolated_db.mark_attendance(sid, at=datetime.now())

    rc = main(["stats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Total students" in out

    out_csv = tmp_path / "out.csv"
    rc = main(["export", "--date", "today", "-o", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    assert "Stats Kid" in out_csv.read_text(encoding="utf-8")


def test_cli_absentee_and_digest(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    present = isolated_db.add_student("P1", "Here", section="A")
    isolated_db.add_student("A1", "Gone", section="A")
    isolated_db.mark_attendance(present, at=datetime.now())

    rc = main(["report", "absentee", "--date", "today"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Gone" in out
    assert "Here" not in out or "Gone" in out

    digest_path = tmp_path / "digest.md"
    abs_csv = tmp_path / "abs.csv"
    rc = main(
        [
            "report",
            "digest",
            "--date",
            "today",
            "-o",
            str(digest_path),
            "--csv",
            str(abs_csv),
        ]
    )
    assert rc == 0
    assert digest_path.exists()
    assert "Gone" in digest_path.read_text(encoding="utf-8")
    assert abs_csv.exists()


def test_cli_help_and_no_crash() -> None:
    from main import build_parser

    parser = build_parser()
    # Ensure subcommands exist
    args = parser.parse_args(["students", "list"])
    assert args.func is not None
    args = parser.parse_args(["report", "digest", "--date", "today"])
    assert args.func is not None
    args = parser.parse_args(["register-folder", "R1", "Name", "/tmp/photos"])
    assert args.func is not None
    args = parser.parse_args(["export", "--json", "--date", "today"])
    assert args.json is True
    args = parser.parse_args(["stats", "--week", "--days", "5"])
    assert args.week is True
    assert args.days == 5
    args = parser.parse_args(["settings", "--threshold", "61", "--theme", "dark"])
    assert args.threshold == 61
    assert args.theme == "dark"
    args = parser.parse_args(["mark", "--roll", "R1", "--period", "Morning"])
    assert args.period == "Morning"
    args = parser.parse_args(["mark", "--roll", "R1", "--status", "Excused"])
    assert args.status == "Excused"
    args = parser.parse_args(["import-students", "roster.csv"])
    assert args.func is not None
    assert args.update is False
    args = parser.parse_args(["import-students", "roster.csv", "--update"])
    assert args.update is True
    args = parser.parse_args(["students", "merge", "A1", "B1"])
    assert args.from_roll == "A1"
    assert args.to_roll == "B1"
    args = parser.parse_args(["holidays", "add", "2026-12-25", "--name", "Xmas"])
    assert args.date == "2026-12-25"
    assert args.name == "Xmas"
    args = parser.parse_args(["serve", "--port", "8768"])
    assert args.port == 8768
    args = parser.parse_args(["doctor"])
    assert args.func is not None
    args = parser.parse_args(["export-unknowns", "--date", "today"])
    assert args.date == "today"
    args = parser.parse_args(
        ["report", "--from", "2026-08-01", "--to", "2026-08-18"]
    )
    assert args.range_from == "2026-08-01"
    assert args.range_to == "2026-08-18"


def test_cli_export_json(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    sid = isolated_db.add_student("J9", "Json Kid", section="A")
    isolated_db.mark_attendance(sid, at=datetime.now(), status="Present")
    out_json = tmp_path / "out.json"
    rc = main(["export", "--date", "today", "--json", "-o", str(out_json)])
    assert rc == 0
    assert out_json.exists()
    text = out_json.read_text(encoding="utf-8")
    assert "Json Kid" in text
    assert '"format": "attendance"' in text
    capsys.readouterr()


def test_cli_stats_week(
    isolated_db: Database, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    sid = isolated_db.add_student("W9", "Week Kid", section="A")
    isolated_db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present"
    )
    rc = main(["stats", "--week", "--from", "2026-08-17", "--days", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Week Kid" in out
    assert "Weekly summary" in out


def test_cli_settings_threshold_and_theme(
    isolated_db: Database, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    rc = main(["settings", "--threshold", "55", "--theme", "dark", "--grace", "15"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "55" in out
    assert isolated_db.get_setting("confidence_threshold") == "55.0" or isolated_db.get_setting(
        "confidence_threshold"
    ) == "55"
    assert isolated_db.get_setting("theme") == "dark"
    assert isolated_db.get_setting("grace_minutes") == "15"


def test_cli_mark_period_late(
    isolated_db: Database, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("M1", "Marky")
    rc = main(
        [
            "mark",
            "--roll",
            "M1",
            "--period",
            "Morning",
            "--at",
            "2026-08-18T09:30:00",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Late" in out
    df = isolated_db.get_attendance()
    assert df.iloc[0]["status"] == "Late"
    assert df.iloc[0]["period"] == "Morning"


def test_cli_register_folder(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from main import main

    photos = tmp_path / "shots"
    photos.mkdir()
    for i in range(3):
        img = np.full((200, 200), 100 + i * 20, dtype=np.uint8)
        assert cv2.imwrite(str(photos / f"{i}.png"), img)

    rc = main(["register-folder", "F1", "Folder Kid", str(photos), "--section", "Z"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Folder Kid" in out
    assert "3" in out
    row = isolated_db.get_student_by_roll("F1")
    assert row is not None
    assert row["name"] == "Folder Kid"
    samples = list((tmp_path / "data" / "faces" / str(row["id"])).glob("*.png"))
    assert len(samples) == 3


def test_cli_register_folder_missing_dir(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    rc = main(["register-folder", "F2", "Ghost", str(tmp_path / "missing")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not a directory" in err


def test_cli_periods(isolated_db: Database, capsys: pytest.CaptureFixture) -> None:
    from main import main

    rc = main(["periods", "add", "Lab", "14:15"])
    assert rc == 0
    rc = main(["periods", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Lab" in out
    assert "14:15" in out


def test_cli_mark_excused(
    isolated_db: Database, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("E1", "Excused Kid")
    rc = main(
        [
            "mark",
            "--roll",
            "E1",
            "--status",
            "Excused",
            "--at",
            "2026-08-18T10:00:00",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Excused" in out
    df = isolated_db.get_attendance()
    assert df.iloc[0]["status"] == "Excused"


def test_cli_import_students(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "roll_number,name,email,department,section\n"
        "I1,Imported One,one@x.com,CS,A\n"
        "I2,Imported Two,two@x.com,EE,B\n",
        encoding="utf-8",
    )
    rc = main(["import-students", str(csv_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Imported 2" in out
    assert isolated_db.get_student_by_roll("I1") is not None
    rc = main(["import-students", str(csv_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped 2" in out.lower() or "Imported 0" in out


def test_cli_import_students_update(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("I1", "Old One", email="old@x.com", department="X", section="Z")
    csv_path = tmp_path / "up.csv"
    csv_path.write_text(
        "roll_number,name,email,department,section\n"
        "I1,Imported One,one@x.com,CS,A\n",
        encoding="utf-8",
    )
    rc = main(["import-students", str(csv_path), "--update"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "updated 1" in out.lower() or "updated" in out.lower()
    row = isolated_db.get_student_by_roll("I1")
    assert row is not None
    assert row["name"] == "Imported One"
    assert row["email"] == "one@x.com"
    assert row["department"] == "CS"
    assert row["section"] == "A"


def test_cli_students_merge(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    src = isolated_db.add_student("FROM1", "From Kid")
    isolated_db.add_student("TO1", "To Kid")
    isolated_db.mark_attendance(src, at=datetime(2026, 8, 18, 9, 0, 0), status="Present")
    face_dir = tmp_path / "data" / "faces" / str(src)
    face_dir.mkdir(parents=True, exist_ok=True)
    (face_dir / "001.png").write_bytes(b"x")

    rc = main(["students", "merge", "FROM1", "TO1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "FROM1" in out and "TO1" in out
    assert isolated_db.get_student_by_roll("FROM1") is None
    att = isolated_db.get_attendance()
    assert len(att) == 1
    assert att.iloc[0]["roll_number"] == "TO1"
    assert not face_dir.exists()


def test_cli_holidays(isolated_db: Database, capsys: pytest.CaptureFixture) -> None:
    from main import main

    rc = main(["holidays", "add", "2026-12-25", "--name", "Christmas"])
    assert rc == 0
    rc = main(["holidays", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2026-12-25" in out
    assert "Christmas" in out


def test_cli_range_report(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    sid = isolated_db.add_student("RR1", "Range Kid", section="A")
    isolated_db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present"
    )
    isolated_db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 9, 0, 0), status="Excused"
    )
    rc = main(["report", "--from", "2026-08-17", "--to", "2026-08-19"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Range Kid" in out
    assert "Range report" in out

    out_json = tmp_path / "range.json"
    rc = main(
        [
            "report",
            "--from",
            "2026-08-17",
            "--to",
            "2026-08-19",
            "--json",
            "-o",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    text = out_json.read_text(encoding="utf-8")
    assert "range_report" in text
    assert "Range Kid" in text
    capsys.readouterr()


def test_cli_export_unknowns(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.log_unknown_face(80.0, at=datetime(2026, 8, 18, 11, 0, 0))
    out_csv = tmp_path / "unk.csv"
    rc = main(["export-unknowns", "--date", "2026-08-18", "-o", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    assert "80" in out_csv.read_text(encoding="utf-8")
    out_json = tmp_path / "unk.json"
    rc = main(
        [
            "export-unknowns",
            "--date",
            "2026-08-18",
            "--json",
            "-o",
            str(out_json),
        ]
    )
    assert rc == 0
    assert '"format": "unknown_faces"' in out_json.read_text(encoding="utf-8")
    capsys.readouterr()


def test_cli_digest_prints_excused(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("P1", "Here", section="A")
    excused = isolated_db.add_student("E1", "Home", section="A")
    isolated_db.mark_attendance(
        excused, at=datetime(2026, 8, 18, 9, 0, 0), status="Excused"
    )
    digest_path = tmp_path / "digest.md"
    rc = main(
        ["report", "digest", "--date", "2026-08-18", "-o", str(digest_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Excused" in out
    assert "Excused" in digest_path.read_text(encoding="utf-8")


def test_cli_digest_prints_late(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    present = isolated_db.add_student("P1", "Here", section="A")
    isolated_db.add_student("A1", "Gone", section="A")
    isolated_db.mark_attendance(
        present, at=datetime(2026, 8, 18, 9, 40, 0), period="Morning"
    )
    digest_path = tmp_path / "digest.md"
    rc = main(
        ["report", "digest", "--date", "2026-08-18", "-o", str(digest_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Late" in out
    assert "Late" in digest_path.read_text(encoding="utf-8")

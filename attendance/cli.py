"""Entry point for the Face Recognition Attendance System.

Usage:
    python main.py                         # GUI (default)
    python main.py train
    python main.py register-folder ROLL NAME DIR
    python main.py import-students file.csv
    python main.py import-students file.csv --update
    python main.py students merge FROM_ROLL TO_ROLL
    python main.py holidays add YYYY-MM-DD --name X
    python main.py holidays list
    python main.py holidays import file.csv
    python main.py students archive ROLL
    python main.py alerts --from DATE --to DATE
    python main.py excuse-all --date YYYY-MM-DD
    python main.py students pin ROLL --pin 1234
    python main.py mark --roll ROLL --pin 1234
    python main.py calendar --from DATE --to DATE
    python main.py calendar --from DATE --to DATE --ics out.ics
    python main.py export-ics --from DATE --to DATE -o out.ics
    python main.py streaks [--section S] [--as-of DATE]
    python main.py mark --roll X --note "bus late"
    python main.py mark --roll X --out
    python main.py backup -o attendance.db
    python main.py restore-db FILE
    python main.py serve --port 8768
    python main.py doctor
    python main.py export --date YYYY-MM-DD -o out.csv
    python main.py export --json --date today -o out.json
    python main.py export-unknowns --date YYYY-MM-DD
    python main.py students list
    python main.py students export -o roster.csv
    python main.py students add --roll X --name Y
    python main.py unknowns list [--date YYYY-MM-DD]
    python main.py unknowns assign ROLL FILE
    python main.py mark --roll X --period Morning
    python main.py mark --roll X --status Excused
    python main.py report absentee --date YYYY-MM-DD
    python main.py report digest --date today
    python main.py report digest --html -o digest.html
    python main.py roster [--min N]
    python main.py report --from YYYY-MM-DD --to YYYY-MM-DD
    python main.py report --from YYYY-MM-DD --to YYYY-MM-DD --html -o out.html
    python main.py stats
    python main.py stats --week
    python main.py settings --threshold 65 --theme dark
    python main.py --threshold 70 stats
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from attendance import __version__, config
from attendance.database import (
    Database,
    calendar_cell_letter,
    doctor_report,
    parse_hhmm,
    week_start,
)


def _parse_date(value: str) -> str:
    """Accept YYYY-MM-DD or the literal 'today'."""
    value = (value or "").strip().lower()
    if value in ("", "today"):
        return datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD or 'today'."
        ) from exc
    return value


def _parse_datetime(value: str) -> datetime:
    raw = (value or "").strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            if fmt == "%Y-%m-%d":
                return parsed.replace(hour=datetime.now().hour, minute=datetime.now().minute, second=0)
            return parsed
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid datetime '{value}'. Use ISO 8601 or 'YYYY-MM-DD HH:MM[:SS]'."
    )


def _db() -> Database:
    return Database()


def _apply_runtime_settings(args: argparse.Namespace | None = None) -> Database:
    """Load persisted settings, then apply any CLI ``--threshold`` override."""
    db = _db()
    raw_theme = db.get_setting("theme")
    if raw_theme:
        config.apply_theme(raw_theme)
    raw_threshold = db.get_setting("confidence_threshold")
    if raw_threshold:
        try:
            config.set_confidence_threshold(float(raw_threshold))
        except ValueError:
            pass
    cli_threshold = getattr(args, "threshold", None) if args is not None else None
    if cli_threshold is not None:
        config.set_confidence_threshold(cli_threshold)
        db.set_setting("confidence_threshold", str(float(cli_threshold)))
    return db


def cmd_gui(_args: argparse.Namespace) -> int:
    from attendance.app import AttendanceApp

    AttendanceApp().run()
    return 0


def cmd_train(_args: argparse.Namespace) -> int:
    # Import lazily so headless DB-only commands don't need OpenCV-heavy paths
    # beyond face_engine itself (still needs opencv-contrib for training).
    from attendance.face_engine import FaceEngine, FaceEngineError, model_is_stale

    print(f"Training LBPH model from {config.FACES_DIR} ...")
    try:
        engine = FaceEngine()
        count = engine.train_from_dataset()
    except FaceEngineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: training failed: {exc}", file=sys.stderr)
        return 1

    print(f"Trained on {count} face samples.")
    print(f"Model written to {config.MODEL_PATH}")
    print(f"Label map written to {config.LABEL_MAP_PATH}")
    print(f"Threshold         : {config.get_confidence_threshold():.1f}")
    if model_is_stale():
        print("Warning: model still flagged as stale after train.", file=sys.stderr)
    else:
        print("Model is up to date.")
    return 0


def cmd_register_folder(args: argparse.Namespace) -> int:
    from attendance.face_engine import FaceEngine, FaceEngineError, mark_model_stale

    roll = (args.roll or "").strip()
    name = (args.name or "").strip()
    source = Path(args.dir)
    if not roll or not name:
        print("ERROR: ROLL and NAME are required.", file=sys.stderr)
        return 1
    if not source.is_dir():
        print(f"ERROR: not a directory: {source}", file=sys.stderr)
        return 1

    db = _db()
    if db.get_student_by_roll(roll):
        print(f"ERROR: student with roll '{roll}' already exists.", file=sys.stderr)
        return 1

    try:
        sid = db.add_student(
            roll_number=roll,
            name=name,
            email=(args.email or "").strip(),
            department=(args.department or "").strip(),
            section=(args.section or "").strip(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not add student: {exc}", file=sys.stderr)
        return 1

    try:
        engine = FaceEngine()
        count = engine.enroll_from_folder(
            source,
            sid,
            max_samples=args.max_samples,
        )
    except FaceEngineError as exc:
        db.delete_student(sid)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        db.delete_student(sid)
        print(f"ERROR: folder enrollment failed: {exc}", file=sys.stderr)
        return 1

    mark_model_stale(f"Registered {roll} from folder")
    dest = config.FACES_DIR / str(sid)
    print(f"Enrolled id={sid} roll={roll} name={name}")
    print(f"Wrote {count} face sample(s) to {dest}")
    skipped = int(getattr(engine, "last_quality_skipped", 0) or 0)
    if skipped:
        print(f"Skipped {skipped} low-quality image(s) (tiny or nearly-black).")
    print("Next: python main.py train")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    # When --date is omitted, export all; when provided, accept YYYY-MM-DD or 'today'.
    date = _parse_date(args.date) if args.date is not None else None
    as_json = bool(getattr(args, "json", False))

    out = Path(args.output) if args.output else None
    if out is None:
        stamp = date or "all"
        ext = "json" if as_json else "csv"
        out = config.EXPORTS_DIR / f"attendance_{stamp}.{ext}"

    db = _db()
    if as_json:
        path = db.export_attendance_json(
            out, date=date, section=args.section, period=args.period
        )
    else:
        path = db.export_attendance_csv(
            out, date=date, section=args.section, period=args.period
        )
    n = len(db.get_attendance(date=date, section=args.section, period=args.period))
    kind = "JSON" if as_json else "CSV"
    print(f"Exported {n} row(s) to {path} ({kind})")
    return 0


def cmd_students_list(args: argparse.Namespace) -> int:
    db = _db()
    rows = db.list_students(section=args.section)
    if not rows:
        print("No students registered.")
        return 0
    print(f"{'ID':>4}  {'ROLL':<12}  {'NAME':<24}  {'SECTION':<10}  {'DEPT':<16}  EMAIL")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['id']:>4}  {r['roll_number']:<12}  {r['name']:<24}  "
            f"{(r['section'] or '—'):<10}  {(r['department'] or '—'):<16}  "
            f"{r['email'] or ''}"
        )
    print(f"\n{len(rows)} student(s).")
    return 0


def cmd_roster(args: argparse.Namespace) -> int:
    min_samples = int(getattr(args, "min_samples", 8) or 8)
    db = _db()
    rows = db.enrollment_roster(min_samples=min_samples)
    as_json = bool(getattr(args, "json", False))
    out = Path(args.output) if getattr(args, "output", None) else None
    if as_json or (out is not None and out.suffix.lower() == ".json"):
        payload = {
            "count": len(rows),
            "min_samples": min_samples,
            "students": rows,
        }
        text = json.dumps(payload, indent=2)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"Wrote roster JSON ({len(rows)} student(s)) to {out}")
        else:
            print(text)
        return 0
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "roll_number",
                    "name",
                    "section",
                    "samples",
                    "ready",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote roster CSV ({len(rows)} student(s)) to {out}")
        return 0
    print(
        f"{'ID':>4}  {'ROLL':<12}  {'NAME':<24}  {'SECTION':<10}  "
        f"{'SAMPLES':>7}  READY"
    )
    print("-" * 80)
    for row in rows:
        ready = "yes" if row["ready"] else "no"
        print(
            f"{row['id']:>4}  {row['roll_number']:<12}  {row['name']:<24}  "
            f"{(row['section'] or '—'):<10}  {int(row['samples']):>7}  {ready}"
        )
    print(f"\n{len(rows)} student(s). Ready if samples ≥ {min_samples}.")
    return 0


def cmd_students_export(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else config.EXPORTS_DIR / "roster.csv"
    db = _db()
    samples_by_id = {
        int(row["id"]): int(row["samples"])
        for row in db.enrollment_roster(min_samples=0)
    }
    from attendance.database import count_enroll_samples

    faces_root = Path(config.FACES_DIR)
    students = db.list_students(include_inactive=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "roll_number",
                "name",
                "email",
                "department",
                "section",
                "active",
                "samples",
            ],
        )
        writer.writeheader()
        for student in students:
            sid = int(student["id"])
            samples = samples_by_id.get(sid)
            if samples is None:
                samples = count_enroll_samples(faces_root / str(sid))
            writer.writerow(
                {
                    "roll_number": student["roll_number"],
                    "name": student["name"],
                    "email": student["email"] or "",
                    "department": student["department"] or "",
                    "section": student["section"] or "",
                    "active": int(student["active"] if student["active"] is not None else 1),
                    "samples": int(samples),
                }
            )
    print(f"Exported {len(students)} student(s) to {out}")
    return 0


def _unknown_files_on_disk(date: str | None = None) -> list[Path]:
    root = Path(config.UNKNOWNS_DIR)
    if not root.is_dir():
        return []
    suffixes = {suffix.lower() for suffix in config.ENROLL_IMAGE_SUFFIXES}
    found: list[Path] = []
    if date:
        folder = root / date
        if not folder.is_dir():
            return []
        try:
            entries = folder.iterdir()
        except OSError:
            return []
        for path in sorted(entries, key=lambda item: item.name):
            try:
                if path.is_file() and path.suffix.lower() in suffixes:
                    found.append(path)
            except OSError:
                continue
        return found
    try:
        entries = root.rglob("*")
    except OSError:
        return []
    for path in sorted(entries, key=lambda item: str(item)):
        try:
            if path.is_file() and path.suffix.lower() in suffixes:
                found.append(path)
        except OSError:
            continue
    return found


def cmd_unknowns_list(args: argparse.Namespace) -> int:
    date = _parse_date(args.date) if args.date is not None else None
    db = _db()
    rows = db.list_unknown_crops(date=date)
    seen: set[str] = set()
    for row in rows:
        raw = row.get("path")
        if not raw:
            continue
        try:
            seen.add(str(Path(str(raw)).resolve()))
        except OSError:
            seen.add(str(raw))
    extras: list[Path] = []
    for path in _unknown_files_on_disk(date=date):
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            extras.append(path)

    if not rows and not extras:
        print("No unknown-face crops.")
        return 0

    if rows:
        print(
            f"{'ID':>4}  {'DATE':<12}  {'TIME':<8}  {'CONF':>6}  PATH"
        )
        print("-" * 72)
        for row in rows:
            conf = row.get("confidence")
            conf_s = f"{float(conf):.0f}" if conf is not None else "—"
            print(
                f"{int(row['id']):>4}  {str(row['date'] or ''):<12}  "
                f"{str(row['time'] or ''):<8}  {conf_s:>6}  "
                f"{row.get('path') or '—'}"
            )
        print(f"\n{len(rows)} logged crop(s).")
    if extras:
        print("On disk (no DB path):")
        for path in extras:
            print(f"  {path}")
        print(f"{len(extras)} file(s) under {config.UNKNOWNS_DIR}.")
    return 0


def cmd_unknowns_assign(args: argparse.Namespace) -> int:
    roll = (args.roll or "").strip()
    src = Path(args.file)
    try:
        result = _db().assign_unknown_crop(src, roll)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Assigned {src} → {result['dest']} "
        f"({result['samples_written']} sample(s) for {result['roll']})"
    )
    return 0


def cmd_students_add(args: argparse.Namespace) -> int:
    roll = args.roll.strip()
    name = args.name.strip()
    if not roll or not name:
        print("ERROR: --roll and --name are required.", file=sys.stderr)
        return 1

    db = _db()
    if db.get_student_by_roll(roll):
        print(f"ERROR: student with roll '{roll}' already exists.", file=sys.stderr)
        return 1

    try:
        sid = db.add_student(
            roll_number=roll,
            name=name,
            email=(args.email or "").strip(),
            department=(args.department or "").strip(),
            section=(args.section or "").strip(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not add student: {exc}", file=sys.stderr)
        return 1

    print(f"Added student id={sid} roll={roll} name={name}")
    print(
        "Note: capture samples in the GUI, or enroll from a folder:\n"
        "  python main.py register-folder ROLL NAME /path/to/photos\n"
        "then run: python main.py train"
    )
    return 0


def cmd_import_students(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.is_file():
        print(f"ERROR: CSV file not found: {path}", file=sys.stderr)
        return 1
    db = _db()
    update = bool(getattr(args, "update", False))
    try:
        result = db.import_students_csv(path, update=update)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    added = int(result["added"])
    skipped = int(result["skipped"])
    updated = int(result.get("updated") or 0)
    if update:
        print(
            f"Imported {added} student(s); updated {updated}; "
            f"skipped {skipped} invalid roll(s)."
        )
    else:
        print(
            f"Imported {added} student(s); skipped {skipped} existing/invalid roll(s)."
        )
    for roll in result["added_rolls"]:
        print(f"  + {roll}")
    for roll in result.get("updated_rolls") or []:
        print(f"  ~ updated {roll}")
    for roll in result["skipped_rolls"]:
        print(f"  - skipped {roll}")
    return 0


def cmd_students_merge(args: argparse.Namespace) -> int:
    from_roll = (args.from_roll or "").strip()
    to_roll = (args.to_roll or "").strip()
    if not from_roll or not to_roll:
        print("ERROR: FROM_ROLL and TO_ROLL are required.", file=sys.stderr)
        return 1
    db = _db()
    try:
        result = db.merge_students(from_roll, to_roll)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    from attendance.face_engine import mark_model_stale

    mark_model_stale(f"Merged {from_roll} into {to_roll}")
    print(
        f"Merged {result['from_roll']} → {result['to_roll']}; "
        f"moved {result['moved']} attendance row(s)."
    )
    return 0


def cmd_students_pin(args: argparse.Namespace) -> int:
    roll = (args.roll or "").strip()
    pin = args.pin
    try:
        _db().set_pin(roll, pin)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"PIN set for {roll}")
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    roll = (args.roll or "").strip()
    pin = (getattr(args, "pin", None) or "").strip() or None
    if pin and not roll:
        print("ERROR: --roll is required with --pin.", file=sys.stderr)
        return 1
    if not roll:
        print("ERROR: --roll is required.", file=sys.stderr)
        return 1
    db = _db()
    row = db.get_student_by_roll(roll)
    if row is None:
        print(f"ERROR: no student with roll '{roll}'.", file=sys.stderr)
        return 1
    when = args.at if getattr(args, "at", None) else None
    note = (getattr(args, "note", None) or "").strip() or None
    touch = bool(getattr(args, "out", False))
    if pin:
        inserted = db.mark_with_pin(
            roll,
            pin,
            confidence=args.confidence,
            status=args.status,
            at=when,
            period=args.period,
            source="pin",
            note=note,
            touch=touch,
        )
    else:
        inserted = db.mark_attendance(
            student_id=int(row["id"]),
            confidence=args.confidence,
            status=args.status,
            at=when,
            period=args.period,
            source="cli",
            note=note,
            touch=touch,
        )
    date_key = (when or datetime.now()).strftime("%Y-%m-%d")
    latest = db.get_attendance(
        date=date_key, student_id=int(row["id"]), period=args.period
    )
    if not inserted:
        if touch and not latest.empty:
            time_out = latest.iloc[0].get("time_out")
            duration = latest.iloc[0].get("duration_seconds")
            if time_out is not None and str(time_out).strip() not in {"", "nan", "None"}:
                extra = ""
                if duration is not None and str(duration) not in {"", "nan", "None"}:
                    extra = f" duration={int(duration)}s"
                print(
                    f"Updated time_out for {row['name']} ({roll}) "
                    f"to {time_out}{extra}"
                )
                return 0
        print(
            f"Skipped {roll}: already marked for this period "
            "or cooldown still active."
        )
        return 0
    status = ""
    period = args.period or ""
    if not latest.empty:
        status = str(latest.iloc[0]["status"])
        period = str(latest.iloc[0].get("period") or period)
    print(
        f"Marked {row['name']} ({roll}) "
        f"status={status or 'Present'} period={period or '—'}"
    )
    return 0


def cmd_report_absentee(args: argparse.Namespace) -> int:
    date = _parse_date(args.date)
    db = _db()
    df = db.get_absentees(date=date, section=args.section)
    if args.output:
        out = Path(args.output)
        if getattr(args, "json", False) or out.suffix.lower() == ".json":
            path = db.export_absentees_json(out, date=date, section=args.section)
        else:
            path = db.export_absentees_csv(out, date=date, section=args.section)
        print(f"Exported {len(df)} absentee(s) to {path}")
        return 0

    print(f"Absentees for {date}" + (f" [section={args.section}]" if args.section else ""))
    if df.empty:
        print("  (none — full attendance or no students)")
        return 0
    for _, row in df.iterrows():
        sec = row.get("section") or "—"
        print(f"  - {row['roll_number']}: {row['name']} [{sec}]")
    print(f"\n{len(df)} absentee(s).")
    return 0


def cmd_report_digest(args: argparse.Namespace) -> int:
    date = _parse_date(args.date)
    fmt = (args.format or "md").lower()
    if getattr(args, "html", False):
        fmt = "html"
    if fmt not in ("md", "txt", "markdown", "text", "html", "htm"):
        print("ERROR: --format must be md, txt, or html", file=sys.stderr)
        return 1
    if fmt in ("markdown",):
        fmt = "md"
    if fmt in ("text",):
        fmt = "txt"
    if fmt in ("htm",):
        fmt = "html"

    out = Path(args.output) if args.output else None
    if out is None:
        ext = {"html": "html", "txt": "txt"}.get(fmt, "md")
        out = config.EXPORTS_DIR / f"digest_{date}.{ext}"

    db = _db()
    path = db.write_daily_digest(
        out_path=out, date=date, section=args.section, fmt=fmt
    )
    digest = db.daily_digest(date=date, section=args.section)
    print(f"Daily digest for {date}")
    print(f"  Present : {digest['present_count']}/{digest['total_students']}")
    print(f"  On time : {digest['on_time_count']}")
    print(f"  Late    : {digest['late_count']}")
    print(f"  Excused : {digest.get('excused_count', 0)}")
    print(f"  Absent  : {digest['absentee_count']}")
    print(f"  Rate    : {digest['attendance_rate']:.1f}%")
    print(f"Wrote {path}")

    if args.csv:
        csv_path = Path(args.csv)
        db.export_absentees_csv(csv_path, date=date, section=args.section)
        print(f"Absentee CSV: {csv_path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    db = _db()
    s = db.stats()
    print(f"Face Recognition Attendance System v{__version__}")
    print(f"  Database         : {db.db_path}")
    print(f"  Total students   : {s['total_students']}")
    print(f"  Total records    : {s['total_records']}")
    print(f"  Present today    : {s['present_today']}")
    print(f"  On time today    : {s['on_time_today']}")
    print(f"  Late today       : {s['late_today']}")
    print(f"  Unknown today    : {s['unknown_today']}")
    print(f"  Attendance rate  : {s['attendance_rate_today']:.1f}%")
    print(f"  Week rate        : {s['week_rate']:.1f}% (from {s['week_start']})")
    print(f"  Threshold        : {config.get_confidence_threshold():.1f}")
    print(f"  Theme            : {config.get_theme()}")

    from attendance.face_engine import invalidate_model_if_needed, model_is_stale

    model_ok = config.MODEL_PATH.exists() and config.LABEL_MAP_PATH.exists()
    print(f"  Model on disk    : {'yes' if model_ok else 'no'}")
    stale = invalidate_model_if_needed() or model_is_stale()
    print(f"  Model stale      : {'yes — retrain recommended' if stale else 'no'}")

    if getattr(args, "week", False):
        start = (
            _parse_date(args.week_from)
            if getattr(args, "week_from", None)
            else week_start()
        )
        days = int(getattr(args, "days", None) or config.WEEK_DAYS_DEFAULT)
        section = getattr(args, "section", None)
        df = db.weekly_summary(start, days=days, section=section)
        rate = db.weekly_rate(start, days=days, section=section)
        print()
        print(f"Weekly summary starting {start} ({days} day(s))"
              + (f" [section={section}]" if section else ""))
        print(f"  Overall rate     : {rate:.1f}%")
        if df.empty:
            print("  (no students)")
            return 0
        print(f"  {'ROLL':<12}  {'NAME':<24}  {'P':>3}  {'L':>3}  {'E':>3}  {'A':>3}  RATE")
        print("  " + "-" * 68)
        for _, row in df.iterrows():
            print(
                f"  {row['roll_number']:<12}  {row['name']:<24}  "
                f"{int(row['present']):>3}  {int(row['late']):>3}  "
                f"{int(row.get('excused', 0)):>3}  "
                f"{int(row['absent']):>3}  {row['attendance_rate']:.1f}%"
            )
    return 0


def cmd_report_range(args: argparse.Namespace) -> int:
    start = getattr(args, "range_from", None)
    end = getattr(args, "range_to", None)
    if not start or not end:
        print("ERROR: --from and --to are required (YYYY-MM-DD).", file=sys.stderr)
        return 1
    start = _parse_date(start)
    end = _parse_date(end)
    if end < start:
        print("ERROR: --to must be on or after --from.", file=sys.stderr)
        return 1

    db = _db()
    try:
        df = db.range_report(start, end, section=args.section)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    as_html = bool(getattr(args, "html", False))
    as_json = bool(getattr(args, "json", False))
    out = Path(args.output) if getattr(args, "output", None) else None
    if as_html or (out is not None and out.suffix.lower() in {".html", ".htm"}):
        if out is None:
            out = config.EXPORTS_DIR / f"range_{start}_{end}.html"
        path = db.export_range_report_html(
            out, start, end, section=args.section
        )
        print(f"Exported range report ({len(df)} student(s)) to {path} (HTML)")
        return 0
    if out is not None:
        if as_json or out.suffix.lower() == ".json":
            path = db.export_range_report_json(
                out, start, end, section=args.section
            )
            kind = "JSON"
        else:
            path = db.export_range_report_csv(
                out, start, end, section=args.section
            )
            kind = "CSV"
        print(f"Exported range report ({len(df)} student(s)) to {path} ({kind})")
        return 0

    print(
        f"Range report {start} → {end}"
        + (f" [section={args.section}]" if args.section else "")
    )
    if df.empty:
        print("  (no students)")
        return 0
    print(f"  {'ROLL':<12}  {'NAME':<24}  {'P':>3}  {'L':>3}  {'E':>3}  {'A':>3}  RATE")
    print("  " + "-" * 68)
    for _, row in df.iterrows():
        print(
            f"  {row['roll_number']:<12}  {row['name']:<24}  "
            f"{int(row['present']):>3}  {int(row['late']):>3}  "
            f"{int(row.get('excused', 0)):>3}  "
            f"{int(row['absent']):>3}  {row['attendance_rate']:.1f}%"
        )
    print(f"\n{len(df)} student(s).")
    return 0


def cmd_report_auto(args: argparse.Namespace) -> int:
    """`report --from DATE --to DATE` with no subcommand."""
    if getattr(args, "range_from", None) and getattr(args, "range_to", None):
        return cmd_report_range(args)
    print(
        "ERROR: specify a report (absentee, digest, range) or pass --from and --to.",
        file=sys.stderr,
    )
    return 2


def cmd_export_unknowns(args: argparse.Namespace) -> int:
    date = _parse_date(args.date) if args.date is not None else None
    as_json = bool(getattr(args, "json", False))
    out = Path(args.output) if args.output else None
    if out is None:
        stamp = date or "all"
        ext = "json" if as_json else "csv"
        out = config.EXPORTS_DIR / f"unknowns_{stamp}.{ext}"

    db = _db()
    if as_json or out.suffix.lower() == ".json":
        path = db.export_unknown_faces_json(out, date=date)
        kind = "JSON"
    else:
        path = db.export_unknown_faces_csv(out, date=date)
        kind = "CSV"
    n = len(db.list_unknown_faces(date=date))
    print(f"Exported {n} unknown-face row(s) to {path} ({kind})")
    return 0


def cmd_settings(args: argparse.Namespace) -> int:
    db = _db()
    changed = False
    if args.threshold is not None:
        try:
            value = config.set_confidence_threshold(args.threshold)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        db.set_setting("confidence_threshold", str(value))
        changed = True
    if args.theme:
        applied = config.apply_theme(args.theme)
        db.set_setting("theme", applied)
        changed = True
    if args.grace is not None:
        if args.grace < 0:
            print("ERROR: --grace must be >= 0", file=sys.stderr)
            return 1
        db.set_setting("grace_minutes", str(int(args.grace)))
        changed = True
    if args.period:
        name = args.period.strip()
        if db.get_period(name) is None and name.lower() not in {
            p["name"].lower() for p in config.PERIODS
        }:
            print(
                f"Warning: period '{name}' is not in the periods table yet.",
                file=sys.stderr,
            )
        db.set_setting("default_period", name)
        changed = True

    settings = db.all_settings()
    print("Settings" + (" (updated)" if changed else ""))
    for key in (
        "theme",
        "confidence_threshold",
        "grace_minutes",
        "default_period",
        "week_days",
    ):
        print(f"  {key:<22}: {settings.get(key, '')}")
    print(f"  live_threshold        : {config.get_confidence_threshold():.1f}")
    print(f"  live_theme            : {config.get_theme()}")
    return 0


def cmd_periods_list(_args: argparse.Namespace) -> int:
    db = _db()
    rows = db.list_periods()
    if not rows:
        print("No periods configured.")
        return 0
    print(f"{'NAME':<16}  START")
    print("-" * 26)
    for row in rows:
        print(f"{row['name']:<16}  {row['start_hhmm']}")
    print(f"\nGrace: {db.get_grace_minutes()} minute(s)")
    return 0


def cmd_holidays_list(_args: argparse.Namespace) -> int:
    db = _db()
    rows = db.list_holidays()
    if not rows:
        print("No holidays configured.")
        return 0
    print(f"{'DATE':<12}  NAME")
    print("-" * 36)
    for row in rows:
        print(f"{row['date']:<12}  {row['name'] or ''}")
    print(f"\n{len(rows)} holiday(s).")
    return 0


def cmd_holidays_add(args: argparse.Namespace) -> int:
    try:
        date = _parse_date(args.date)
    except argparse.ArgumentTypeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    name = (args.name or "").strip() or "Holiday"
    try:
        _db().add_holiday(date, name)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved holiday {date} ({name})")
    return 0


def cmd_holidays_import(args: argparse.Namespace) -> int:
    path = Path(args.file)
    try:
        result = _db().import_holidays_csv(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {result['added']} holiday(s); skipped {result['skipped']}.")
    return 0


def cmd_students_archive(args: argparse.Namespace) -> int:
    roll = (args.roll or "").strip()
    restore = bool(getattr(args, "restore", False))
    try:
        if restore:
            _db().restore_student(roll)
            print(f"Restored {roll} to the active roster.")
        else:
            _db().archive_student(roll)
            print(f"Archived {roll}.")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_students_undo(args: argparse.Namespace) -> int:
    roll = (args.roll or "").strip()
    try:
        removed = _db().undo_last_mark(roll)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if removed is None:
        print(f"No attendance rows for {roll}.")
        return 0
    print(
        f"Removed last mark for {roll}: {removed['date']} {removed['time']} "
        f"{removed['status']}"
    )
    return 0


def cmd_alerts(args: argparse.Namespace) -> int:
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    threshold = float(args.threshold)
    db = _db()
    at_risk = db.at_risk_students(start, end, threshold=threshold, section=args.section)
    consecutive = db.consecutive_absences(
        as_of=end, min_days=args.min_days, section=args.section
    )
    print(f"At-risk (<{threshold:.0f}%) {start}..{end}: {len(at_risk)}")
    for row in at_risk:
        print(
            f"  {row['roll_number']}: {row['name']} "
            f"{row['attendance_rate']:.1f}% absent={row['absent']}"
        )
    print(f"Consecutive absences (≥{args.min_days}) as of {end}: {len(consecutive)}")
    for row in consecutive:
        print(
            f"  {row['roll_number']}: {row['name']} "
            f"{row['consecutive_absences']} day(s)"
        )
    return 0


def cmd_excuse_all(args: argparse.Namespace) -> int:
    date = _parse_date(args.date)
    try:
        marked = _db().bulk_excuse(date, section=args.section, note=args.note or "bulk excuse")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Excused {marked} unmarked student(s) on {date}.")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    dest = Path(args.output) if args.output else config.EXPORTS_DIR / "attendance-backup.db"
    try:
        path = _db().backup_database(dest)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    return 0


def cmd_restore_db(args: argparse.Namespace) -> int:
    src = Path(args.file)
    if not src.is_file():
        print(f"ERROR: backup not found: {src}", file=sys.stderr)
        return 1
    try:
        path = _db().restore_database(src)
    except (OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Restored database from {src} → {path}")
    return 0


def _write_attendance_ics(
    db: Database,
    start: str,
    end: str,
    section: str | None,
    out_path: Path | None,
) -> int:
    try:
        text = db.export_attendance_ics(start, end, section=section)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    dest = out_path or (config.EXPORTS_DIR / f"attendance_{start}_{end}.ics")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(text.encode("utf-8"))
    events = text.count("BEGIN:VEVENT")
    print(f"Wrote {events} event(s) to {dest}")
    return 0


def cmd_export_ics(args: argparse.Namespace) -> int:
    start = _parse_date(args.cal_from)
    end = _parse_date(args.cal_to)
    out = Path(args.output) if getattr(args, "output", None) else None
    return _write_attendance_ics(_db(), start, end, args.section, out)


def cmd_streaks(args: argparse.Namespace) -> int:
    as_of = _parse_date(getattr(args, "as_of", None) or "today")
    section = getattr(args, "section", None)
    try:
        rows = _db().streaks_report(as_of=as_of, section=section)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    section_bit = f" [section={section}]" if section else ""
    print(f"Present streaks as of {as_of}{section_bit}")
    if not rows:
        print("  (no students)")
        return 0
    print(f"{'ROLL':<12}  {'NAME':<24}  {'SECTION':<10}  STREAK")
    print("-" * 56)
    for row in rows:
        print(
            f"{row['roll_number']:<12}  {str(row['name'] or '')[:24]:<24}  "
            f"{(row['section'] or '—'):<10}  {int(row['streak'])}"
        )
    print(f"\n{len(rows)} student(s).")
    return 0


def cmd_calendar(args: argparse.Namespace) -> int:
    start = _parse_date(args.cal_from)
    end = _parse_date(args.cal_to)
    db = _db()
    ics_arg = getattr(args, "ics", None)
    if ics_arg is not None:
        out = Path(ics_arg) if ics_arg else None
        return _write_attendance_ics(db, start, end, args.section, out)
    try:
        grid = db.calendar_grid(start, end, section=args.section)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    as_json = bool(getattr(args, "json", False))
    out = Path(args.output) if getattr(args, "output", None) else None
    if as_json or (out is not None and out.suffix.lower() == ".json"):
        text = json.dumps(grid, indent=2)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"Wrote calendar JSON ({len(grid['students'])} student(s)) to {out}")
        else:
            print(text)
        return 0
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        holidays = grid["holidays"]
        weekends = set(grid.get("weekends") or [])
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["roll_number", "name", "section", *grid["dates"]])
            for student in grid["students"]:
                cells = []
                for day in grid["dates"]:
                    if day in holidays:
                        cells.append(holidays[day])
                    elif day in weekends:
                        cells.append("Weekend")
                    else:
                        cells.append(student["days"].get(day) or "")
                writer.writerow(
                    [
                        student["roll_number"],
                        student["name"],
                        student["section"],
                        *cells,
                    ]
                )
        print(f"Wrote calendar CSV ({len(grid['students'])} student(s)) to {out}")
        return 0

    holidays = grid["holidays"]
    weekends = set(grid.get("weekends") or [])
    section_bit = f" [section={args.section}]" if args.section else ""
    print(f"Calendar {grid['from']} → {grid['to']}{section_bit}")
    print("P=Present  L=Late  E=Excused  .=unmarked  H=holiday  W=weekend")
    if not grid["students"]:
        print("  (no students)")
        return 0
    ruler = "".join(day[8:10] for day in grid["dates"])
    print(f"{'ROLL':<12}  {'NAME':<24}  {ruler}")
    print("-" * (40 + len(grid["dates"])))
    for student in grid["students"]:
        heat = []
        for day in grid["dates"]:
            heat.append(
                calendar_cell_letter(
                    day, student["days"].get(day), holidays, weekends
                )
            )
        name = (student["name"] or "")[:24]
        print(f"{student['roll_number']:<12}  {name:<24}  {''.join(heat)}")
    print(f"\n{len(grid['students'])} student(s), {len(grid['dates'])} day(s).")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    before = _parse_date(args.before)
    deleted = _db().purge_attendance_before(before)
    print(f"Deleted {deleted} attendance row(s) before {before}.")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    report = doctor_report()
    print(f"OpenCV           : {report['opencv_version'] or 'missing'}")
    print(f"cv2.face         : {'yes' if report['cv2_face'] else 'no'}")
    print(f"Cascade          : {report['cascade_path'] or 'missing'}")
    exists = "exists" if report["db_exists"] else "missing"
    print(f"Database         : {report['db_path']} ({exists})")
    print(f"Students         : {report['student_count']}")
    print(
        f"Sample folders   : {report['sample_folders']} "
        f"({report['sample_images']} images)"
    )
    print(f"Model            : {'yes' if report['model_exists'] else 'no'}")
    print(f"Model stale      : {'yes' if report['model_stale'] else 'no'}")
    print(f"Holidays         : {report['holiday_count']}")
    return 0 if report["ok"] else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from attendance.server import serve_forever

    port = int(getattr(args, "port", 8768) or 8768)
    if port < 1 or port > 65535:
        print("ERROR: --port must be between 1 and 65535.", file=sys.stderr)
        return 1
    print(f"Attendance API on http://127.0.0.1:{port} (local only, no auth)")
    print("  GET /health")
    print("  GET /stats")
    print("  GET /doctor")
    print("  GET /students")
    print("  GET /roster?min=8")
    print("  GET /attendance?date=YYYY-MM-DD")
    print("  GET /holidays")
    print("  GET /alerts?from=&to=&threshold=75")
    print("  GET /calendar?from=&to=&section=")
    print("  GET /calendar.ics?from=&to=&section=")
    print("  GET /streaks?as_of=&section=")
    print("  POST /pin   {\"roll\":\"R1\",\"pin\":\"1234\"}")
    print("  POST /mark  {\"roll\":\"R1\",\"status\":\"Present\"}  (optional pin, note, touch/out)")
    serve_forever("127.0.0.1", port)
    return 0


def cmd_periods_add(args: argparse.Namespace) -> int:
    name = (args.name or "").strip()
    start = (args.start or "").strip()
    if not name or not start:
        print("ERROR: name and start (HH:MM) are required.", file=sys.stderr)
        return 1
    try:
        parse_hhmm(start)
        _db().upsert_period(name, start)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved period {name} starting {start}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="LBPH distance threshold (lower = stricter). Persisted in settings.",
    )

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Face Recognition Attendance System — GUI and headless CLI.",
        parents=[shared],
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser(
        "train", parents=[shared], help="Train the LBPH face recognition model"
    )
    p_train.set_defaults(func=cmd_train)

    # register-folder
    p_reg = sub.add_parser(
        "register-folder",
        parents=[shared],
        help="Enroll a student from a folder of photos (no webcam)",
    )
    p_reg.add_argument("roll", help="Unique roll number")
    p_reg.add_argument("name", help="Full name")
    p_reg.add_argument("dir", help="Directory of PNG/JPG photos")
    p_reg.add_argument("--email", default="", help="Email address")
    p_reg.add_argument("--department", default="", help="Department")
    p_reg.add_argument("--section", default="", help="Class / section")
    p_reg.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on samples to write (default: SAMPLES_PER_STUDENT; 0 = unlimited)",
    )
    p_reg.set_defaults(func=cmd_register_folder)

    # import-students
    p_import = sub.add_parser(
        "import-students",
        parents=[shared],
        help="Import students from a CSV (skip existing rolls; --update upserts)",
    )
    p_import.add_argument("file", help="CSV with roll_number,name,email,department,section")
    p_import.add_argument(
        "--update",
        action="store_true",
        help="Update name/email/department/section for existing roll numbers",
    )
    p_import.set_defaults(func=cmd_import_students)

    # export
    p_export = sub.add_parser(
        "export", parents=[shared], help="Export attendance records to CSV or JSON"
    )
    p_export.add_argument(
        "--date",
        default=None,
        help="Filter by date (YYYY-MM-DD or 'today'). Omit for all records.",
    )
    p_export.add_argument(
        "-o", "--output", default=None, help="Output path (default: exports/)"
    )
    p_export.add_argument(
        "--section", default=None, help="Filter by student section"
    )
    p_export.add_argument(
        "--period", default=None, help="Filter by period name"
    )
    p_export.add_argument(
        "--json",
        action="store_true",
        help="Write JSON instead of CSV (same filters)",
    )
    p_export.set_defaults(func=cmd_export)

    # export-unknowns
    p_unk = sub.add_parser(
        "export-unknowns",
        parents=[shared],
        help="Export unknown-face log to CSV or JSON",
    )
    p_unk.add_argument(
        "--date",
        default=None,
        help="Filter by date (YYYY-MM-DD or 'today'). Omit for all rows.",
    )
    p_unk.add_argument(
        "-o", "--output", default=None, help="Output path (default: exports/)"
    )
    p_unk.add_argument(
        "--json",
        action="store_true",
        help="Write JSON instead of CSV",
    )
    p_unk.set_defaults(func=cmd_export_unknowns)

    p_unknowns = sub.add_parser(
        "unknowns",
        parents=[shared],
        help="List unknown-face crops or assign one to a student",
    )
    unknowns_sub = p_unknowns.add_subparsers(dest="unknowns_cmd", required=True)
    p_unk_list = unknowns_sub.add_parser(
        "list",
        parents=[shared],
        help="List logged unknown-face crops (and files under data/unknowns/)",
    )
    p_unk_list.add_argument(
        "--date",
        default=None,
        help="Filter by date (YYYY-MM-DD or 'today'). Omit for all.",
    )
    p_unk_list.set_defaults(func=cmd_unknowns_list)
    p_unk_assign = unknowns_sub.add_parser(
        "assign",
        parents=[shared],
        help="Copy an unknown-face PNG into a student's sample folder",
    )
    p_unk_assign.add_argument("roll", help="Student roll number")
    p_unk_assign.add_argument("file", help="Path to the unknown-face image")
    p_unk_assign.set_defaults(func=cmd_unknowns_assign)

    # students
    p_students = sub.add_parser("students", parents=[shared], help="Manage students")
    students_sub = p_students.add_subparsers(dest="students_cmd", required=True)

    p_list = students_sub.add_parser(
        "list", parents=[shared], help="List registered students"
    )
    p_list.add_argument("--section", default=None, help="Filter by section")
    p_list.set_defaults(func=cmd_students_list)

    p_st_export = students_sub.add_parser(
        "export",
        parents=[shared],
        help="Export the roster CSV (roll, name, email, department, section, active, samples)",
    )
    p_st_export.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: exports/roster.csv)",
    )
    p_st_export.set_defaults(func=cmd_students_export)

    p_add = students_sub.add_parser(
        "add", parents=[shared], help="Add a student (no face capture)"
    )
    p_add.add_argument("--roll", required=True, help="Roll number (unique)")
    p_add.add_argument("--name", required=True, help="Full name")
    p_add.add_argument("--email", default="", help="Email address")
    p_add.add_argument("--department", default="", help="Department")
    p_add.add_argument("--section", default="", help="Class / section")
    p_add.set_defaults(func=cmd_students_add)

    p_merge = students_sub.add_parser(
        "merge",
        parents=[shared],
        help="Move attendance from FROM_ROLL to TO_ROLL, then delete FROM",
    )
    p_merge.add_argument("from_roll", help="Roll number to absorb and delete")
    p_merge.add_argument("to_roll", help="Roll number to keep")
    p_merge.set_defaults(func=cmd_students_merge)

    p_archive = students_sub.add_parser(
        "archive", parents=[shared], help="Hide a student from reports without deleting history"
    )
    p_archive.add_argument("roll", help="Roll number to archive")
    p_archive.set_defaults(func=cmd_students_archive, restore=False)

    p_restore = students_sub.add_parser(
        "restore", parents=[shared], help="Return an archived student to the active roster"
    )
    p_restore.add_argument("roll", help="Roll number to restore")
    p_restore.set_defaults(func=cmd_students_archive, restore=True)

    p_undo = students_sub.add_parser(
        "undo", parents=[shared], help="Delete the most recent attendance row for a roll"
    )
    p_undo.add_argument("roll", help="Roll number")
    p_undo.set_defaults(func=cmd_students_undo)

    p_pin = students_sub.add_parser(
        "pin", parents=[shared], help="Set a 4–8 digit kiosk PIN for a roll"
    )
    p_pin.add_argument("roll", help="Roll number")
    p_pin.add_argument(
        "--pin",
        required=True,
        help="4–8 digit PIN (stored as scrypt, never in the clear)",
    )
    p_pin.set_defaults(func=cmd_students_pin)

    # mark
    p_mark = sub.add_parser(
        "mark", parents=[shared], help="Mark attendance for one student (headless)"
    )
    p_mark.add_argument("--roll", required=True, help="Student roll number")
    p_mark.add_argument("--period", default=None, help="Period name (default from settings)")
    p_mark.add_argument(
        "--status",
        default=None,
        choices=["Present", "Late", "Excused", "Sick"],
        help="Force a status instead of classifying from the clock (Sick → Excused)",
    )
    p_mark.add_argument(
        "--at",
        type=_parse_datetime,
        default=None,
        help="When to mark (ISO or YYYY-MM-DD HH:MM[:SS]); default now",
    )
    p_mark.add_argument("--confidence", type=float, default=None)
    p_mark.add_argument(
        "--pin",
        default=None,
        help="Kiosk PIN; when set, verifies then marks via mark_with_pin",
    )
    p_mark.add_argument(
        "--note",
        default=None,
        help="Optional free-text note stored with the mark",
    )
    p_mark.add_argument(
        "--out",
        action="store_true",
        help="Update time_out (dwell) on an existing mark for this period",
    )
    p_mark.set_defaults(func=cmd_mark)

    # report
    report_common = argparse.ArgumentParser(add_help=False)
    report_common.add_argument(
        "--from",
        dest="range_from",
        default=None,
        help="Range start date (YYYY-MM-DD or 'today')",
    )
    report_common.add_argument(
        "--to",
        dest="range_to",
        default=None,
        help="Range end date (YYYY-MM-DD or 'today')",
    )
    report_common.add_argument(
        "-o", "--output", default=None, help="Optional output path"
    )
    report_common.add_argument(
        "--section", default=None, help="Filter by student section"
    )
    report_common.add_argument(
        "--json", action="store_true", help="Write JSON when exporting"
    )
    report_common.add_argument(
        "--html",
        action="store_true",
        help="Write a self-contained HTML report (digest or range)",
    )

    p_report = sub.add_parser(
        "report",
        parents=[shared, report_common],
        help="Generate attendance reports",
    )
    p_report.set_defaults(func=cmd_report_auto)
    report_sub = p_report.add_subparsers(dest="report_cmd", required=False)

    p_abs = report_sub.add_parser(
        "absentee",
        parents=[shared, report_common],
        help="List students with no attendance on a date",
    )
    p_abs.add_argument(
        "--date", default="today", help="YYYY-MM-DD or 'today' (default: today)"
    )
    p_abs.set_defaults(func=cmd_report_absentee)

    p_digest = report_sub.add_parser(
        "digest",
        parents=[shared, report_common],
        help="Daily summary (present/late/excused/absent/rate)",
    )
    p_digest.add_argument(
        "--date", default="today", help="YYYY-MM-DD or 'today' (default: today)"
    )
    p_digest.add_argument(
        "--format",
        default="md",
        choices=["md", "txt", "markdown", "text", "html", "htm"],
        help="Output format (default: md)",
    )
    p_digest.add_argument(
        "--csv", default=None, help="Also write absentees CSV to this path"
    )
    p_digest.set_defaults(func=cmd_report_digest)

    p_range = report_sub.add_parser(
        "range",
        parents=[shared, report_common],
        help="Per-student present/late/excused/absent over a date range",
    )
    p_range.set_defaults(func=cmd_report_range)

    # stats
    p_stats = sub.add_parser(
        "stats", parents=[shared], help="Show database, model, and weekly statistics"
    )
    p_stats.add_argument(
        "--week",
        action="store_true",
        help="Print a per-student weekly present/late/absent table",
    )
    p_stats.add_argument(
        "--from",
        dest="week_from",
        default=None,
        help="Week start date (YYYY-MM-DD). Default: Monday of the current week.",
    )
    p_stats.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days in the window (default: 7; use 5 for weekdays)",
    )
    p_stats.add_argument("--section", default=None, help="Filter by section")
    p_stats.set_defaults(func=cmd_stats)

    # settings
    p_set = sub.add_parser(
        "settings", parents=[shared], help="Show or update persisted settings"
    )
    p_set.add_argument(
        "--theme", choices=["light", "dark"], default=None, help="UI theme"
    )
    p_set.add_argument(
        "--grace", type=int, default=None, help="Late-arrival grace minutes"
    )
    p_set.add_argument("--period", default=None, help="Default period name")
    p_set.set_defaults(func=cmd_settings)

    # periods
    p_periods = sub.add_parser(
        "periods", parents=[shared], help="List or add class periods"
    )
    periods_sub = p_periods.add_subparsers(dest="periods_cmd", required=True)
    p_pl = periods_sub.add_parser("list", parents=[shared], help="List periods")
    p_pl.set_defaults(func=cmd_periods_list)
    p_pa = periods_sub.add_parser("add", parents=[shared], help="Add or update a period")
    p_pa.add_argument("name", help="Period name (e.g. Morning)")
    p_pa.add_argument("start", help="Start time HH:MM")
    p_pa.set_defaults(func=cmd_periods_add)

    # holidays
    p_holidays = sub.add_parser(
        "holidays", parents=[shared], help="List or add no-class / holiday dates"
    )
    holidays_sub = p_holidays.add_subparsers(dest="holidays_cmd", required=True)
    p_hl = holidays_sub.add_parser("list", parents=[shared], help="List holidays")
    p_hl.set_defaults(func=cmd_holidays_list)
    p_hi = holidays_sub.add_parser("import", parents=[shared], help="Import holidays from CSV")
    p_hi.add_argument("file", help="CSV with date,name")
    p_hi.set_defaults(func=cmd_holidays_import)
    p_ha = holidays_sub.add_parser("add", parents=[shared], help="Add or update a holiday")
    p_ha.add_argument("date", help="Date (YYYY-MM-DD or 'today')")
    p_ha.add_argument("--name", default="Holiday", help="Holiday name")
    p_ha.set_defaults(func=cmd_holidays_add)

    p_alerts = sub.add_parser(
        "alerts", parents=[shared], help="List at-risk students and consecutive absences"
    )
    p_alerts.add_argument("--from", dest="start", default="today", help="Range start")
    p_alerts.add_argument("--to", dest="end", default="today", help="Range end")
    p_alerts.add_argument(
        "--below",
        dest="threshold",
        type=float,
        default=75.0,
        help="Flag students below this attendance percent",
    )
    p_alerts.add_argument("--min-days", type=int, default=3, help="Consecutive absence threshold")
    p_alerts.add_argument("--section", default=None)
    p_alerts.set_defaults(func=cmd_alerts)

    p_excuse = sub.add_parser(
        "excuse-all",
        parents=[shared],
        help="Mark remaining unmarked students Excused for a date",
    )
    p_excuse.add_argument("--date", default="today")
    p_excuse.add_argument("--section", default=None)
    p_excuse.add_argument("--note", default="bulk excuse")
    p_excuse.set_defaults(func=cmd_excuse_all)

    p_backup = sub.add_parser("backup", parents=[shared], help="Copy the SQLite database")
    p_backup.add_argument("-o", "--output", default=None, help="Destination .db path")
    p_backup.set_defaults(func=cmd_backup)

    p_restore = sub.add_parser(
        "restore-db",
        parents=[shared],
        help="Replace the configured database with a backup file",
    )
    p_restore.add_argument("file", help="Backup .db to copy over DB_PATH")
    p_restore.set_defaults(func=cmd_restore_db)

    p_cal = sub.add_parser(
        "calendar",
        parents=[shared],
        help="ASCII attendance heatmap for a date range",
    )
    p_cal.add_argument(
        "--from",
        dest="cal_from",
        required=True,
        help="Range start (YYYY-MM-DD or 'today')",
    )
    p_cal.add_argument(
        "--to",
        dest="cal_to",
        required=True,
        help="Range end (YYYY-MM-DD or 'today')",
    )
    p_cal.add_argument("--section", default=None, help="Filter by student section")
    p_cal.add_argument(
        "-o", "--output", default=None, help="Write CSV (or JSON with --json)"
    )
    p_cal.add_argument("--json", action="store_true", help="Write JSON instead of CSV/ASCII")
    p_cal.add_argument(
        "--ics",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Write iCalendar (.ics). Default path under exports/ if FILE omitted.",
    )
    p_cal.set_defaults(func=cmd_calendar)

    p_export_ics = sub.add_parser(
        "export-ics",
        parents=[shared],
        help="Export Present/Late/Excused marks as an iCalendar (.ics) file",
    )
    p_export_ics.add_argument(
        "--from",
        dest="cal_from",
        required=True,
        help="Range start (YYYY-MM-DD or 'today')",
    )
    p_export_ics.add_argument(
        "--to",
        dest="cal_to",
        required=True,
        help="Range end (YYYY-MM-DD or 'today')",
    )
    p_export_ics.add_argument("--section", default=None, help="Filter by student section")
    p_export_ics.add_argument(
        "-o",
        "--output",
        "--ics",
        dest="output",
        default=None,
        help="Output .ics path (default: exports/)",
    )
    p_export_ics.set_defaults(func=cmd_export_ics)

    p_roster = sub.add_parser(
        "roster",
        parents=[shared],
        help="Enrollment roster with face-sample counts",
    )
    p_roster.add_argument(
        "--min",
        dest="min_samples",
        type=int,
        default=8,
        help="Samples required to be ready (default: 8)",
    )
    p_roster.add_argument(
        "-o", "--output", default=None, help="Write CSV (or JSON with --json)"
    )
    p_roster.add_argument("--json", action="store_true", help="Write JSON")
    p_roster.set_defaults(func=cmd_roster)

    p_streaks = sub.add_parser(
        "streaks",
        parents=[shared],
        help="Present/Late streaks as of a date (skips holidays and weekends)",
    )
    p_streaks.add_argument(
        "--as-of",
        dest="as_of",
        default="today",
        help="End date (YYYY-MM-DD or 'today')",
    )
    p_streaks.add_argument("--section", default=None, help="Filter by student section")
    p_streaks.set_defaults(func=cmd_streaks)

    p_purge = sub.add_parser(
        "purge", parents=[shared], help="Delete attendance rows older than a date"
    )
    p_purge.add_argument("--before", required=True, help="Delete rows with date < this")
    p_purge.set_defaults(func=cmd_purge)

    # serve
    p_serve = sub.add_parser(
        "serve",
        parents=[shared],
        help="Local JSON API on 127.0.0.1 (no auth)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8768,
        help="Port to bind (default: 8768). Host is always 127.0.0.1.",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_doctor = sub.add_parser(
        "doctor",
        parents=[shared],
        help="Print OpenCV / database / model diagnostics",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # No subcommand → launch GUI (preserves historical behaviour).
    if not argv:
        try:
            _apply_runtime_settings()
            return cmd_gui(argparse.Namespace())
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 1

    # `--threshold` alone (or with only global flags) should not open the GUI.
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    try:
        _apply_runtime_settings(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

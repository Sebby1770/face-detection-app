"""SQLite-backed persistence layer for students and attendance records."""
from __future__ import annotations

import csv
import hashlib
import hmac
import html
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from . import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number   TEXT    UNIQUE NOT NULL,
    name          TEXT    NOT NULL,
    email         TEXT,
    department    TEXT,
    section       TEXT,
    registered_on TEXT    NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    pin_hash      TEXT
);

CREATE TABLE IF NOT EXISTS attendance (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    date       TEXT    NOT NULL,
    time       TEXT    NOT NULL,
    time_out   TEXT,
    status     TEXT    NOT NULL DEFAULT 'Present',
    confidence REAL,
    period     TEXT,
    source     TEXT    NOT NULL DEFAULT 'cli',
    note       TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attendance_date     ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_student  ON attendance(student_id);

CREATE TABLE IF NOT EXISTS periods (
    name       TEXT PRIMARY KEY,
    start_hhmm TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unknown_faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,
    time       TEXT    NOT NULL,
    confidence REAL,
    path       TEXT
);

CREATE INDEX IF NOT EXISTS idx_unknown_faces_date ON unknown_faces(date);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS holidays (
    date TEXT PRIMARY KEY,
    name TEXT
);

CREATE TABLE IF NOT EXISTS pin_attempts (
    roll_number  TEXT PRIMARY KEY,
    fails        INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT
);

CREATE TABLE IF NOT EXISTS attendance_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_id INTEGER,
    student_id    INTEGER,
    old_status    TEXT,
    new_status    TEXT,
    note          TEXT,
    changed_at    TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Pure helpers (clock-driven; no database required)
# ---------------------------------------------------------------------------
def parse_hhmm(value: str) -> tuple[int, int]:
    """Parse ``HH:MM``, ``HH.MM``, or ``HHMM`` into a 24-hour (hour, minute)."""
    raw = (value or "").strip()
    if not raw:
        raise ValueError("time is required (HH:MM)")
    if raw.isdigit() and len(raw) in (3, 4):
        raw = raw.zfill(4)
        raw = f"{raw[:2]}:{raw[2:]}"
    parts = raw.replace(".", ":").split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time '{value}'. Use HH:MM.")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid time '{value}'. Use HH:MM.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time '{value}'.")
    return hour, minute


def format_hhmm(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def classify_attendance_status(
    at: datetime,
    start_hhmm: str = config.DEFAULT_PERIOD_START,
    grace_minutes: int = config.LATE_GRACE_MINUTES,
) -> str:
    """Return ``Present`` if ``at`` is on or before start+grace, else ``Late``.

    Arrival *before* the period start is on time. The grace window is inclusive.
    """
    hour, minute = parse_hhmm(start_hhmm)
    start = at.replace(hour=hour, minute=minute, second=0, microsecond=0)
    deadline = start + timedelta(minutes=int(grace_minutes))
    if at <= deadline:
        return "Present"
    return "Late"


def week_start(date: Optional[str] = None) -> str:
    """Monday (ISO) of the week containing ``date`` (YYYY-MM-DD) or today."""
    if date:
        day = datetime.strptime(date, "%Y-%m-%d")
    else:
        day = datetime.now()
    monday = day - timedelta(days=day.weekday())
    return monday.strftime("%Y-%m-%d")


def week_dates(start_date: str, days: int = 7) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    span = max(1, int(days))
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(span)]


DEFAULT_WEEKEND_DAYS = "6,7"
_WEEKDAY_NAME_TO_ISO: dict[str, int] = {
    "mon": 1,
    "monday": 1,
    "tue": 2,
    "tues": 2,
    "tuesday": 2,
    "wed": 3,
    "wednesday": 3,
    "thu": 4,
    "thur": 4,
    "thurs": 4,
    "thursday": 4,
    "fri": 5,
    "friday": 5,
    "sat": 6,
    "saturday": 6,
    "sun": 7,
    "sunday": 7,
}


def iso_weekday(date_str: str) -> int:
    """Return ISO weekday for ``YYYY-MM-DD`` (Monday=1 … Sunday=7)."""
    return datetime.strptime((date_str or "").strip(), "%Y-%m-%d").isoweekday()


def parse_weekend_days(raw: str | None) -> frozenset[int]:
    """Parse weekend ISO weekdays. Default ``"6,7"`` (Saturday, Sunday).

    Accepts comma-separated numbers (``"6,7"``) or names (``"Sat,Sun"``,
    ``"saturday,sunday"``). ``None`` / blank → ``{6, 7}``. ``none`` / ``off``
    → no weekend days.
    """
    if raw is None:
        return frozenset({6, 7})
    text = str(raw).strip()
    if not text:
        return frozenset({6, 7})
    lowered = text.lower()
    if lowered in {"none", "off", "no", "-"}:
        return frozenset()
    days: set[int] = set()
    for token in lowered.replace(";", ",").split(","):
        part = token.strip().strip(".")
        if not part:
            continue
        if part.isdigit():
            number = int(part)
            if 1 <= number <= 7:
                days.add(number)
            continue
        mapped = _WEEKDAY_NAME_TO_ISO.get(part)
        if mapped is not None:
            days.add(mapped)
    return frozenset(days) if days else frozenset({6, 7})


def format_weekend_days(days: set[int] | frozenset[int]) -> str:
    cleaned = sorted({int(day) for day in days if 1 <= int(day) <= 7})
    if not cleaned:
        return "none"
    return ",".join(str(day) for day in cleaned)


def is_school_day(
    date_str: str,
    holidays: set[str],
    weekend_days: set[int],
) -> bool:
    """True when ``date_str`` is not a holiday and not a configured weekend."""
    if (date_str or "") in holidays:
        return False
    return iso_weekday(date_str) not in weekend_days


def _enroll_suffixes() -> set[str]:
    return {suffix.lower() for suffix in config.ENROLL_IMAGE_SUFFIXES}


def count_enroll_samples(folder: Path) -> int:
    """Count files in ``folder`` whose suffix is an enroll image type."""
    if not folder.is_dir():
        return 0
    suffixes = _enroll_suffixes()
    count = 0
    try:
        entries = folder.iterdir()
    except OSError:
        return 0
    for path in entries:
        try:
            if path.is_file() and path.suffix.lower() in suffixes:
                count += 1
        except OSError:
            continue
    return count


def next_enroll_index(folder: Path) -> int:
    """Next ``NNN`` index among enroll-suffix files with numeric stems."""
    existing: list[int] = []
    if not folder.is_dir():
        return 1
    suffixes = _enroll_suffixes()
    try:
        entries = folder.iterdir()
    except OSError:
        return 1
    for path in entries:
        try:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            existing.append(int(path.stem))
        except (OSError, ValueError):
            continue
    return (max(existing) + 1) if existing else 1


ATTENDANCE_STATUSES: tuple[str, ...] = ("Present", "Late", "Excused")
EXCUSED_STATUSES: frozenset[str] = frozenset({"Excused", "Sick"})
MARK_SOURCES: frozenset[str] = frozenset(
    {"camera", "cli", "api", "pin", "gui", "kiosk"}
)
CALENDAR_MAX_DAYS = 93
_STATUS_ALIASES: dict[str, str] = {
    "present": "Present",
    "late": "Late",
    "excused": "Excused",
    "excuse": "Excused",
    "sick": "Excused",
}


def normalize_attendance_status(status: str) -> str:
    """Map a free-text status to ``Present``, ``Late``, or ``Excused``."""
    key = (status or "").strip().lower()
    if key not in _STATUS_ALIASES:
        raise ValueError(
            f"Invalid status '{status}'. Use Present, Late, or Excused."
        )
    return _STATUS_ALIASES[key]


def is_excused_status(status: Optional[str]) -> bool:
    if status is None:
        return False
    return (status or "").strip().lower() in {"excused", "sick"}


def normalize_pin(pin: str) -> str:
    """Require a 4–8 digit PIN. Returns the stripped value."""
    cleaned = (pin or "").strip()
    if not cleaned.isdigit() or not (4 <= len(cleaned) <= 8):
        raise ValueError("PIN must be 4–8 digits")
    return cleaned


PIN_MAX_FAILS = 5
PIN_LOCKOUT_SECONDS = 60
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_SALT_BYTES = 16


def _pin_key(roll_number: str, pin: str) -> bytes:
    return f"{roll_number}:{pin}".encode("utf-8")


def _legacy_pin_hash(roll_number: str, pin: str) -> str:
    return hashlib.sha256(_pin_key(roll_number, pin)).hexdigest()


def hash_student_pin(roll_number: str, pin: str) -> str:
    """Return a salted scrypt hash ``scrypt$n$r$p$salt_hex$dk_hex``.

    The PIN itself is never stored. A random 16-byte salt means the same
    digits produce different hashes for different students (and re-hashes).
    The roll number is mixed into the key so a stolen hash is roll-bound.
    """
    roll = (roll_number or "").strip()
    cleaned = normalize_pin(pin)
    salt = os.urandom(_SCRYPT_SALT_BYTES)
    dk = hashlib.scrypt(
        _pin_key(roll, cleaned),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return (
        f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"
    )


def verify_stored_pin(stored: str, roll_number: str, pin: str) -> bool:
    """Check ``pin`` against a scrypt hash or a legacy sha256(roll:pin) hex."""
    roll = (roll_number or "").strip()
    try:
        cleaned = normalize_pin(pin)
    except ValueError:
        return False
    raw = str(stored or "")
    if raw.startswith("scrypt$"):
        parts = raw.split("$")
        if len(parts) != 6:
            return False
        _, n_s, r_s, p_s, salt_hex, dk_hex = parts
        try:
            n, r, p = int(n_s), int(r_s), int(p_s)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(dk_hex)
        except ValueError:
            return False
        if n < 2 or n > 2**16 or r < 1 or p < 1 or not salt or not expected:
            return False
        try:
            actual = hashlib.scrypt(
                _pin_key(roll, cleaned),
                salt=salt,
                n=n,
                r=r,
                p=p,
                dklen=len(expected),
            )
        except (ValueError, TypeError, OSError):
            return False
        return hmac.compare_digest(actual, expected)
    if len(raw) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in raw):
        return hmac.compare_digest(raw.lower(), _legacy_pin_hash(roll, cleaned))
    return False


def normalize_mark_source(source: str | None) -> str:
    raw = (source or "cli").strip().lower()
    if raw not in MARK_SOURCES:
        return "cli"
    return raw


def _df_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def normalize_mark_note(note: str | None) -> Optional[str]:
    """Strip a free-text mark note; empty values become ``None``."""
    if note is None:
        return None
    text = str(note).strip()
    return text or None


def _is_blank_time(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan", "nat"}


def parse_attendance_datetime(
    date_str: object, time_str: object
) -> Optional[datetime]:
    """Combine a stored ``YYYY-MM-DD`` date with ``HH:MM[:SS]`` time."""
    if _is_blank_time(date_str) or _is_blank_time(time_str):
        return None
    raw = f"{str(date_str).strip()} {str(time_str).strip()}"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def duration_seconds_between(
    date_str: object, time_in: object, time_out: object
) -> Optional[int]:
    """Seconds between in-time and time_out on ``date_str``, or ``None``."""
    start = parse_attendance_datetime(date_str, time_in)
    end = parse_attendance_datetime(date_str, time_out)
    if start is None or end is None:
        return None
    delta = int((end - start).total_seconds())
    if delta < 0:
        return None
    return delta


def calendar_cell_letter(
    day: str,
    status: Optional[str],
    holidays: dict[str, str] | set[str],
    weekends: set[str] | list[str] | frozenset[str],
) -> str:
    """Map a calendar cell to ``P`` / ``L`` / ``E`` / ``.`` / ``W`` / ``H``."""
    if day in holidays:
        return "H"
    weekend_set = set(weekends)
    if day in weekend_set:
        return "W"
    if not status:
        return "."
    return {"Present": "P", "Late": "L", "Excused": "E"}.get(str(status), ".")


def _ics_escape(value: str) -> str:
    return (
        (value or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


def _ics_fold_line(line: str) -> str:
    """Fold an iCalendar content line at 75 octets (RFC 5545)."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    chunks: list[str] = []
    start = 0
    first = True
    while start < len(raw):
        limit = 75 if first else 74
        piece = raw[start : start + limit]
        while True:
            try:
                text = piece.decode("utf-8")
                break
            except UnicodeDecodeError:
                piece = piece[:-1]
                if not piece:
                    text = ""
                    break
        consumed = len(piece)
        if consumed <= 0:
            break
        chunks.append(text if first else f" {text}")
        start += consumed
        first = False
    return "\r\n".join(chunks)


def _ics_uid(roll_number: str, date: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ".-_" else "-" for ch in (roll_number or "")
    )
    return f"{safe}-{date}@face-recognition-attendance"


def _present_streak_from_days(
    first_by_date: dict[str, str],
    holidays: set[str],
    as_of: str,
    weekend_days: set[int] | frozenset[int] | None = None,
) -> int:
    """Count consecutive Present/Late school days ending at ``as_of``.

    Holidays and weekend dates are skipped (not counted, not a break).
    Excused days also skip — they do not increment or break the streak,
    matching ``consecutive_absences``. Unmarked or other statuses break
    the run.
    """
    if not first_by_date:
        return 0
    skip_weekends: set[int] | frozenset[int] = (
        weekend_days if weekend_days is not None else parse_weekend_days(None)
    )
    streak = 0
    day = datetime.strptime(as_of, "%Y-%m-%d")
    earliest = datetime.strptime(min(first_by_date), "%Y-%m-%d")
    while day >= earliest:
        key = day.strftime("%Y-%m-%d")
        if not is_school_day(key, holidays, skip_weekends):
            day -= timedelta(days=1)
            continue
        status = first_by_date.get(key)
        if status is not None and is_excused_status(status):
            day -= timedelta(days=1)
            continue
        counted = False
        if status is not None:
            try:
                counted = normalize_attendance_status(status) in {"Present", "Late"}
            except ValueError:
                counted = status in {"Present", "Late"}
        if counted:
            streak += 1
            day -= timedelta(days=1)
            continue
        break
    return streak


def _empty_absentees_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "roll_number",
            "name",
            "email",
            "department",
            "section",
        ]
    )


def _html_escape(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return html.escape(text, quote=True)


_HTML_REPORT_CSS = (
    ":root { color-scheme: dark light; }\n"
    "body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0;"
    " background: #0f172a; color: #e2e8f0; }\n"
    "main { max-width: 880px; margin: 0 auto; padding: 32px 24px 48px; }\n"
    "h1 { font-size: 1.6rem; margin: 0 0 8px; }\n"
    "h2 { font-size: 1.15rem; margin: 28px 0 10px; }\n"
    ".muted { color: #94a3b8; }\n"
    ".note { color: #fbbf24; }\n"
    "table { width: 100%; border-collapse: collapse; background: #1e293b;"
    " border-radius: 8px; overflow: hidden; }\n"
    "th, td { text-align: left; padding: 8px 10px;"
    " border-bottom: 1px solid #334155; }\n"
    "th { color: #94a3b8; font-weight: 600; }\n"
    ".kpis { display: grid; grid-template-columns: repeat(auto-fit,"
    " minmax(140px, 1fr)); gap: 10px; margin: 18px 0; }\n"
    ".kpi { background: #1e293b; border: 1px solid #334155;"
    " border-radius: 10px; padding: 12px 14px; }\n"
    ".kpi strong { display: block; font-size: 1.25rem; }\n"
    ".empty { color: #94a3b8; font-style: italic; }\n"
    "td.num { text-align: right; font-variant-numeric: tabular-nums; }\n"
    "@media (prefers-color-scheme: light) {\n"
    "  body { background: #f8fafc; color: #0f172a; }\n"
    "  .muted { color: #64748b; }\n"
    "  table, .kpi { background: #fff; border-color: #e2e8f0; }\n"
    "  th, td { border-bottom-color: #e2e8f0; }\n"
    "}\n"
)


def _digest_people_table_html(df: pd.DataFrame, empty_label: str) -> str:
    if df is None or df.empty:
        return f"<p class=\"empty\">{_html_escape(empty_label)}</p>"
    rows: list[str] = []
    for _, row in df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_html_escape(row.get('roll_number'))}</td>"
            f"<td>{_html_escape(row.get('name'))}</td>"
            f"<td>{_html_escape(row.get('department') or '—')}</td>"
            f"<td>{_html_escape(row.get('section') or '—')}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Roll</th><th>Name</th><th>Department</th><th>Section</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_daily_digest_html(
    digest: dict,
    *,
    section_line: str,
    note: str,
    present_shown: pd.DataFrame,
    absentees: pd.DataFrame,
    excused: pd.DataFrame,
    excused_count: int,
    rate: float,
) -> str:
    note_html = (
        f"<p class=\"note\">{_html_escape(note)}</p>" if note else ""
    )
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>Daily Attendance Digest — {_html_escape(digest['date'])}</title>\n"
        "<style>\n"
        f"{_HTML_REPORT_CSS}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"<h1>Daily Attendance Digest — {_html_escape(digest['date'])}</h1>\n"
        f"<p class=\"muted\">{_html_escape(section_line)}</p>\n"
        f"{note_html}"
        "<div class=\"kpis\">\n"
        f"<div class=\"kpi\"><span>Total students</span>"
        f"<strong>{int(digest['total_students'])}</strong></div>\n"
        f"<div class=\"kpi\"><span>Present</span>"
        f"<strong>{int(digest['present_count'])}</strong></div>\n"
        f"<div class=\"kpi\"><span>On time</span>"
        f"<strong>{int(digest['on_time_count'])}</strong></div>\n"
        f"<div class=\"kpi\"><span>Late</span>"
        f"<strong>{int(digest['late_count'])}</strong></div>\n"
        f"<div class=\"kpi\"><span>Excused</span>"
        f"<strong>{int(excused_count)}</strong></div>\n"
        f"<div class=\"kpi\"><span>Absent</span>"
        f"<strong>{int(digest['absentee_count'])}</strong></div>\n"
        f"<div class=\"kpi\"><span>Attendance rate</span>"
        f"<strong>{rate:.1f}%</strong></div>\n"
        "</div>\n"
        "<h2>Present</h2>\n"
        f"{_digest_people_table_html(present_shown, 'None present.')}\n"
        "<h2>Absent</h2>\n"
        f"{_digest_people_table_html(absentees, 'None — full attendance.')}\n"
        "<h2>Excused</h2>\n"
        f"{_digest_people_table_html(excused, 'None.')}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


def _render_range_report_html(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    section: Optional[str],
    overall_rate: float,
) -> str:
    section_line = f"Section: {section}" if section else "Section: (all)"
    count = 0 if df is None or df.empty else int(len(df))
    if df is None or df.empty:
        table = "<p class=\"empty\">No students in this range.</p>"
    else:
        rows: list[str] = []
        for _, row in df.iterrows():
            rate = float(row.get("attendance_rate") or 0.0)
            rows.append(
                "<tr>"
                f"<td>{_html_escape(row.get('roll_number'))}</td>"
                f"<td>{_html_escape(row.get('name'))}</td>"
                f"<td>{_html_escape(row.get('section') or '—')}</td>"
                f"<td class=\"num\">{int(row.get('present') or 0)}</td>"
                f"<td class=\"num\">{int(row.get('late') or 0)}</td>"
                f"<td class=\"num\">{int(row.get('excused') or 0)}</td>"
                f"<td class=\"num\">{int(row.get('absent') or 0)}</td>"
                f"<td class=\"num\">{int(row.get('days') or 0)}</td>"
                f"<td class=\"num\">{rate:.1f}%</td>"
                "</tr>"
            )
        table = (
            "<table><thead><tr>"
            "<th>Roll</th><th>Name</th><th>Section</th>"
            "<th>Present</th><th>Late</th><th>Excused</th>"
            "<th>Absent</th><th>Days</th><th>Rate</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    title = f"Range report — {start_date} → {end_date}"
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{_html_escape(title)}</title>\n"
        "<style>\n"
        f"{_HTML_REPORT_CSS}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"<h1>{_html_escape(title)}</h1>\n"
        f"<p class=\"muted\">{_html_escape(section_line)}</p>\n"
        "<div class=\"kpis\">\n"
        f"<div class=\"kpi\"><span>Students</span>"
        f"<strong>{count}</strong></div>\n"
        f"<div class=\"kpi\"><span>Overall rate</span>"
        f"<strong>{overall_rate:.1f}%</strong></div>\n"
        "</div>\n"
        "<h2>Weekly / range stats</h2>\n"
        f"{table}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


class Database:
    """Thin wrapper around the SQLite database for the attendance app."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = Path(db_path) if db_path else config.DB_PATH
        self._init_schema()

    # ---------------------------------------------------------------------
    # Connection helpers
    # ---------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            self._seed_defaults(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply lightweight schema migrations for existing databases."""
        student_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(students)").fetchall()
        }
        if "section" not in student_cols:
            conn.execute("ALTER TABLE students ADD COLUMN section TEXT")

        attendance_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(attendance)").fetchall()
        }
        if "period" not in attendance_cols:
            conn.execute("ALTER TABLE attendance ADD COLUMN period TEXT")
        if "note" not in attendance_cols:
            conn.execute("ALTER TABLE attendance ADD COLUMN note TEXT")
        if "source" not in attendance_cols:
            conn.execute(
                "ALTER TABLE attendance ADD COLUMN source TEXT DEFAULT 'cli'"
            )
        if "time_out" not in attendance_cols:
            conn.execute("ALTER TABLE attendance ADD COLUMN time_out TEXT")

        if "active" not in student_cols:
            conn.execute(
                "ALTER TABLE students ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
            )
        if "pin_hash" not in student_cols:
            conn.execute("ALTER TABLE students ADD COLUMN pin_hash TEXT")

        unknown_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(unknown_faces)").fetchall()
        }
        if "path" not in unknown_cols:
            conn.execute("ALTER TABLE unknown_faces ADD COLUMN path TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pin_attempts (
                roll_number  TEXT PRIMARY KEY,
                fails        INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            )
            """
        )

        conn.execute(
            "UPDATE attendance SET period = ? "
            "WHERE period IS NULL OR TRIM(period) = ''",
            (config.DEFAULT_PERIOD_NAME,),
        )
        conn.execute(
            """
            DELETE FROM attendance
            WHERE id NOT IN (
                SELECT MIN(id) FROM attendance
                GROUP BY student_id, date, period
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_attendance_student_date_period
            ON attendance(student_id, date, period)
            """
        )

    def _seed_defaults(self, conn: sqlite3.Connection) -> None:
        existing = conn.execute("SELECT COUNT(*) AS c FROM periods").fetchone()["c"]
        if int(existing) == 0:
            for period in config.PERIODS:
                conn.execute(
                    "INSERT OR IGNORE INTO periods (name, start_hhmm) VALUES (?, ?)",
                    (period["name"], period["start_hhmm"]),
                )
        defaults = {
            "theme": "light",
            "confidence_threshold": str(config.CONFIDENCE_THRESHOLD),
            "grace_minutes": str(config.LATE_GRACE_MINUTES),
            "default_period": config.DEFAULT_PERIOD_NAME,
            "week_days": str(config.WEEK_DAYS_DEFAULT),
            "weekend_days": DEFAULT_WEEKEND_DAYS,
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )

    # ---------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )

    def all_settings(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM settings ORDER BY key"
            ).fetchall()
        return {str(r["key"]): str(r["value"]) for r in rows}

    def get_grace_minutes(self) -> int:
        raw = self.get_setting("grace_minutes")
        if raw is not None:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
        return int(config.LATE_GRACE_MINUTES)

    def get_default_period_name(self) -> str:
        return (
            self.get_setting("default_period") or config.DEFAULT_PERIOD_NAME
        ).strip() or config.DEFAULT_PERIOD_NAME

    def get_weekend_days(self) -> frozenset[int]:
        return parse_weekend_days(self.get_setting("weekend_days"))

    def set_weekend_days(
        self,
        raw: str | None | set[int] | frozenset[int] | list[int] = None,
    ) -> None:
        if isinstance(raw, (set, frozenset, list, tuple)):
            days = frozenset(int(day) for day in raw if 1 <= int(day) <= 7)
        else:
            days = parse_weekend_days(raw)
        self.set_setting("weekend_days", format_weekend_days(days))

    # ---------------------------------------------------------------------
    # Periods
    # ---------------------------------------------------------------------
    def list_periods(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT name, start_hhmm FROM periods ORDER BY start_hhmm, name"
                ).fetchall()
            )

    def get_period(self, name: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT name, start_hhmm FROM periods WHERE name = ? COLLATE NOCASE",
                (name.strip(),),
            ).fetchone()

    def upsert_period(self, name: str, start_hhmm: str) -> None:
        hour, minute = parse_hhmm(start_hhmm)
        normalized = format_hhmm(hour, minute)
        label = name.strip()
        if not label:
            raise ValueError("period name is required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO periods (name, start_hhmm) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET start_hhmm = excluded.start_hhmm
                """,
                (label, normalized),
            )

    def delete_period(self, name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM periods WHERE name = ? COLLATE NOCASE", (name.strip(),))

    def resolve_period_start(self, period: Optional[str] = None) -> tuple[str, str]:
        """Return ``(period_name, start_hhmm)`` for marking / classification."""
        name = (period or self.get_default_period_name() or "").strip()
        if name:
            row = self.get_period(name)
            if row is not None:
                return str(row["name"]), str(row["start_hhmm"])
        if name:
            return name, config.DEFAULT_PERIOD_START
        return config.DEFAULT_PERIOD_NAME, config.DEFAULT_PERIOD_START

    # ---------------------------------------------------------------------
    # Holidays
    # ---------------------------------------------------------------------
    def add_holiday(self, date: str, name: str = "Holiday") -> None:
        """Insert or replace a no-class day. ``date`` is YYYY-MM-DD."""
        day = (date or "").strip()
        datetime.strptime(day, "%Y-%m-%d")
        label = (name or "").strip() or "Holiday"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO holidays (date, name) VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET name = excluded.name
                """,
                (day, label),
            )

    def list_holidays(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT date, name FROM holidays ORDER BY date"
                ).fetchall()
            )

    def get_holiday(self, date: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT date, name FROM holidays WHERE date = ?",
                ((date or "").strip(),),
            ).fetchone()

    def delete_holiday(self, date: str) -> None:
        """Remove a holiday date (YYYY-MM-DD). Missing rows are a no-op."""
        day = (date or "").strip()
        datetime.strptime(day, "%Y-%m-%d")
        with self._connect() as conn:
            conn.execute("DELETE FROM holidays WHERE date = ?", (day,))

    def holiday_dates_between(self, start_date: str, end_date: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date FROM holidays WHERE date >= ? AND date <= ?",
                (start_date, end_date),
            ).fetchall()
        return {str(r["date"]) for r in rows}

    # ---------------------------------------------------------------------
    # Students
    # ---------------------------------------------------------------------
    def add_student(
        self,
        roll_number: str,
        name: str,
        email: str = "",
        department: str = "",
        section: str = "",
    ) -> int:
        """Insert a student and return the generated ID."""
        registered_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO students
                    (roll_number, name, email, department, section, registered_on)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    roll_number.strip(),
                    name.strip(),
                    email.strip(),
                    department.strip(),
                    section.strip(),
                    registered_on,
                ),
            )
            return int(cur.lastrowid)

    def get_student(self, student_id: int) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM students WHERE id = ?", (student_id,)
            ).fetchone()

    def get_student_by_roll(self, roll_number: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM students WHERE roll_number = ?", (roll_number.strip(),)
            ).fetchone()

    def list_students(
        self,
        section: Optional[str] = None,
        include_inactive: bool = False,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM students"
        clauses: list[str] = []
        params: list = []
        if not include_inactive:
            clauses.append("COALESCE(active, 1) = 1")
        if section is not None and section != "":
            clauses.append("section = ?")
            params.append(section.strip())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY name COLLATE NOCASE"
        with self._connect() as conn:
            return conn.execute(query, params).fetchall()

    def enrollment_roster(
        self,
        faces_dir: Path | None = None,
        min_samples: int = 8,
    ) -> list[dict]:
        """Active students with face-sample counts under ``faces_dir``.

        ``ready`` is True when ``samples >= min_samples``. Counts files whose
        suffix is in ``config.ENROLL_IMAGE_SUFFIXES`` (case-insensitive) in
        ``faces_dir / <student_id>/`` (default ``config.FACES_DIR``).
        """
        root = Path(faces_dir) if faces_dir is not None else Path(config.FACES_DIR)
        threshold = max(0, int(min_samples))
        rows: list[dict] = []
        for student in self.list_students():
            sid = int(student["id"])
            folder = root / str(sid)
            samples = count_enroll_samples(folder)
            rows.append(
                {
                    "id": sid,
                    "roll_number": student["roll_number"],
                    "name": student["name"],
                    "section": student["section"] or "",
                    "samples": int(samples),
                    "ready": samples >= threshold,
                }
            )
        return rows

    def list_sections(self) -> list[str]:
        """Return distinct non-empty section values, sorted case-insensitively."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT section FROM students
                WHERE section IS NOT NULL AND TRIM(section) != ''
                ORDER BY section COLLATE NOCASE
                """
            ).fetchall()
        return [str(r["section"]) for r in rows]

    def delete_student(self, student_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM students WHERE id = ?", (student_id,))

    def update_student(
        self,
        roll_number: str,
        name: str,
        email: str = "",
        department: str = "",
        section: str = "",
    ) -> None:
        """Update name/email/department/section for an existing roll number."""
        roll = roll_number.strip()
        if not roll:
            raise ValueError("roll number is required")
        if not name.strip():
            raise ValueError("name is required")
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE students
                   SET name = ?, email = ?, department = ?, section = ?
                 WHERE roll_number = ?
                """,
                (
                    name.strip(),
                    email.strip(),
                    department.strip(),
                    section.strip(),
                    roll,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError(f"no student with roll '{roll}'")

    def set_pin(self, roll_number: str, pin: str) -> None:
        """Store a scrypt hash of the PIN. Never persists the raw PIN."""
        roll = (roll_number or "").strip()
        if not roll:
            raise ValueError("roll number is required")
        student = self.get_student_by_roll(roll)
        if student is None:
            raise ValueError(f"no student with roll '{roll}'")
        pin_hash = hash_student_pin(roll, pin)
        with self._connect() as conn:
            conn.execute(
                "UPDATE students SET pin_hash = ? WHERE roll_number = ?",
                (pin_hash, roll),
            )

    def verify_pin(self, roll_number: str, pin: str) -> bool:
        """Return True when the PIN matches. False if unset, unknown, or wrong.

        After ``PIN_MAX_FAILS`` failures for a roll, further verifies fail for
        ``PIN_LOCKOUT_SECONDS``. A successful verify clears the attempt counter.
        Accepts both scrypt hashes and legacy ``sha256(roll:pin)`` hex.
        """
        roll = (roll_number or "").strip()
        student = self.get_student_by_roll(roll)
        if student is None:
            return False
        now = datetime.now()
        with self._connect() as conn:
            attempt = conn.execute(
                "SELECT fails, locked_until FROM pin_attempts WHERE roll_number = ?",
                (roll,),
            ).fetchone()
            fails = int(attempt["fails"]) if attempt is not None else 0
            locked_until_raw = (
                str(attempt["locked_until"]) if attempt is not None else ""
            )
            if locked_until_raw:
                try:
                    locked_until = datetime.fromisoformat(locked_until_raw)
                except ValueError:
                    locked_until = None
                else:
                    if now < locked_until:
                        return False
                    fails = 0

            stored = None
            if "pin_hash" in student.keys():
                stored = student["pin_hash"]
            ok = bool(stored) and verify_stored_pin(str(stored), roll, pin)
            if ok:
                conn.execute(
                    "DELETE FROM pin_attempts WHERE roll_number = ?", (roll,)
                )
                return True

            fails += 1
            locked_until_s = None
            if fails >= PIN_MAX_FAILS:
                locked_until_s = (
                    now + timedelta(seconds=PIN_LOCKOUT_SECONDS)
                ).isoformat(timespec="seconds")
                fails = PIN_MAX_FAILS
            conn.execute(
                """
                INSERT INTO pin_attempts (roll_number, fails, locked_until)
                VALUES (?, ?, ?)
                ON CONFLICT(roll_number) DO UPDATE SET
                    fails = excluded.fails,
                    locked_until = excluded.locked_until
                """,
                (roll, fails, locked_until_s),
            )
            return False

    def mark_with_pin(self, roll_number: str, pin: str, **kwargs) -> bool:
        """Verify PIN then ``mark_attendance``. Raises on bad PIN / missing / archived."""
        roll = (roll_number or "").strip()
        student = self.get_student_by_roll(roll)
        if student is None:
            raise ValueError(f"no student with roll '{roll}'")
        if int(student["active"] if "active" in student.keys() else 1) == 0:
            raise ValueError("cannot mark an archived student")
        normalize_pin(pin)
        if not self.verify_pin(roll, pin):
            raise ValueError("invalid PIN")
        kwargs.setdefault("source", "pin")
        return self.mark_attendance(int(student["id"]), **kwargs)

    def merge_students(
        self,
        from_roll: str,
        to_roll: str,
        faces_dir: Optional[Path] = None,
    ) -> dict:
        """Move attendance from ``from_roll`` onto ``to_roll``, then delete FROM.

        Face samples under ``faces_dir/<from_id>/`` are removed. Returns counts.
        """
        src_roll = (from_roll or "").strip()
        dst_roll = (to_roll or "").strip()
        if not src_roll or not dst_roll:
            raise ValueError("FROM and TO roll numbers are required")
        if src_roll == dst_roll:
            raise ValueError("cannot merge a student into themselves")
        src = self.get_student_by_roll(src_roll)
        dst = self.get_student_by_roll(dst_roll)
        if src is None:
            raise ValueError(f"no student with roll '{src_roll}'")
        if dst is None:
            raise ValueError(f"no student with roll '{dst_roll}'")
        src_id = int(src["id"])
        dst_id = int(dst["id"])
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM attendance
                WHERE student_id = ?
                  AND EXISTS (
                    SELECT 1 FROM attendance d
                    WHERE d.student_id = ?
                      AND d.date = attendance.date
                      AND COALESCE(d.period, '') = COALESCE(attendance.period, '')
                  )
                """,
                (src_id, dst_id),
            )
            cur = conn.execute(
                "UPDATE attendance SET student_id = ? WHERE student_id = ?",
                (dst_id, src_id),
            )
            moved = int(cur.rowcount)
            conn.execute("DELETE FROM students WHERE id = ?", (src_id,))

        faces_root = Path(faces_dir) if faces_dir else config.FACES_DIR
        face_dir = faces_root / str(src_id)
        removed_faces = False
        if face_dir.is_dir():
            shutil.rmtree(face_dir, ignore_errors=True)
            removed_faces = not face_dir.exists()
        return {
            "from_roll": src_roll,
            "to_roll": dst_roll,
            "from_id": src_id,
            "to_id": dst_id,
            "moved": moved,
            "removed_faces": removed_faces,
        }

    def import_students_csv(
        self, csv_path: Path | str, update: bool = False
    ) -> dict:
        """Add students from a CSV with roll_number,name,email,department,section.

        Existing roll numbers are skipped unless ``update`` is True, in which
        case name/email/department/section are overwritten. Returns counts and
        the rolls that were added, updated, or skipped.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        required = ("roll_number", "name")
        optional = ("email", "department", "section")

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")
            headers = {
                (name or "").strip().lower().replace(" ", "_"): name
                for name in reader.fieldnames
                if name is not None
            }
            aliases = {
                "roll_number": ("roll_number", "roll", "roll_no", "rollnumber"),
                "name": ("name", "full_name", "student_name"),
                "email": ("email", "e_mail"),
                "department": ("department", "dept"),
                "section": ("section", "class"),
            }
            columns: dict[str, str] = {}
            for dest, keys in aliases.items():
                for key in keys:
                    if key in headers:
                        columns[dest] = headers[key]
                        break
            missing = [col for col in required if col not in columns]
            if missing:
                raise ValueError(
                    "CSV is missing required column(s): "
                    + ", ".join(missing)
                    + ". Expected: roll_number,name,email,department,section"
                )

            added = 0
            skipped = 0
            updated = 0
            added_rolls: list[str] = []
            skipped_rolls: list[str] = []
            updated_rolls: list[str] = []
            for raw in reader:
                roll = (raw.get(columns["roll_number"]) or "").strip()
                name = (raw.get(columns["name"]) or "").strip()
                if not roll and not name:
                    continue
                if not roll or not name:
                    skipped += 1
                    if roll:
                        skipped_rolls.append(roll)
                    continue

                def _opt(field: str) -> str:
                    src = columns.get(field)
                    if not src:
                        return ""
                    return (raw.get(src) or "").strip()

                email = _opt("email") if "email" in optional else ""
                department = _opt("department")
                section = _opt("section")

                if self.get_student_by_roll(roll) is not None:
                    if update:
                        self.update_student(
                            roll_number=roll,
                            name=name,
                            email=email,
                            department=department,
                            section=section,
                        )
                        updated += 1
                        updated_rolls.append(roll)
                    else:
                        skipped += 1
                        skipped_rolls.append(roll)
                    continue

                self.add_student(
                    roll_number=roll,
                    name=name,
                    email=email,
                    department=department,
                    section=section,
                )
                added += 1
                added_rolls.append(roll)

        return {
            "added": added,
            "skipped": skipped,
            "updated": updated,
            "added_rolls": added_rolls,
            "skipped_rolls": skipped_rolls,
            "updated_rolls": updated_rolls,
        }

    # ---------------------------------------------------------------------
    # Attendance
    # ---------------------------------------------------------------------
    def mark_attendance(
        self,
        student_id: int,
        confidence: float | None = None,
        status: str | None = None,
        at: Optional[datetime] = None,
        period: Optional[str] = None,
        source: str | None = "cli",
        note: str | None = None,
        touch: bool = False,
    ) -> bool:
        """Insert an attendance row if this student+date+period is unmarked.

        Returns True if a new row was inserted, False if a row already exists
        for ``(student_id, date, period)`` or the same-period cooldown is
        still active. Morning then Afternoon on the same day is allowed
        immediately (cooldown is per period).

        When a row already exists and ``touch`` is True, ``time_out`` is
        updated to ``at`` when that clock is after the in-time and after any
        existing ``time_out``. Touch never inserts a second row.

        When ``status`` is omitted, classify Present vs Late from ``at`` (or
        now), the chosen period start, and the configured grace window.
        Pass ``Excused`` (or ``Sick``) to record an excused absence.

        ``at`` can be provided for tests; defaults to now.
        ``source`` is one of camera|cli|api|pin|gui|kiosk (unknown → cli).
        ``note`` is optional free text stored on the attendance row.
        Period is always stored as a name (never NULL).
        """
        now = at or datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        student = self.get_student(student_id)
        if student is None:
            raise ValueError(f"no student with id {student_id}")
        if int(student["active"] if "active" in student.keys() else 1) == 0:
            raise ValueError("cannot mark an archived student")
        period_name, start_hhmm = self.resolve_period_start(period)
        if not period_name:
            period_name = config.DEFAULT_PERIOD_NAME
        if status is None:
            status = classify_attendance_status(
                now,
                start_hhmm=start_hhmm,
                grace_minutes=self.get_grace_minutes(),
            )
        else:
            status = normalize_attendance_status(status)
        origin = normalize_mark_source(source)
        remark = normalize_mark_note(note)

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id, time, time_out FROM attendance
                WHERE student_id = ? AND date = ? AND period = ?
                LIMIT 1
                """,
                (student_id, date_str, period_name),
            ).fetchone()
            if existing is not None:
                if not touch:
                    return False
                in_dt = parse_attendance_datetime(date_str, existing["time"])
                if in_dt is None or now <= in_dt:
                    return False
                existing_out = None
                if "time_out" in existing.keys():
                    existing_out = existing["time_out"]
                out_dt = parse_attendance_datetime(date_str, existing_out)
                if out_dt is not None and now <= out_dt:
                    return False
                conn.execute(
                    "UPDATE attendance SET time_out = ? WHERE id = ?",
                    (time_str, int(existing["id"])),
                )
                return False

            last = conn.execute(
                """
                SELECT date, time FROM attendance
                WHERE student_id = ? AND period = ?
                ORDER BY id DESC LIMIT 1
                """,
                (student_id, period_name),
            ).fetchone()

            if last is not None:
                last_dt = datetime.strptime(
                    f"{last['date']} {last['time']}", "%Y-%m-%d %H:%M:%S"
                )
                delta = (now - last_dt).total_seconds()
                if delta < config.ATTENDANCE_COOLDOWN_SECONDS:
                    return False

            try:
                conn.execute(
                    """
                    INSERT INTO attendance
                        (student_id, date, time, time_out, status,
                         confidence, period, source, note)
                    VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?)
                    """,
                    (
                        student_id,
                        date_str,
                        time_str,
                        status,
                        confidence,
                        period_name,
                        origin,
                        remark,
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def get_attendance(
        self,
        date: Optional[str] = None,
        student_id: Optional[int] = None,
        section: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return attendance joined with student info as a DataFrame."""
        query = """
            SELECT a.id          AS id,
                   a.date        AS date,
                   a.time        AS time,
                   a.time_out    AS time_out,
                   a.status      AS status,
                   a.confidence  AS confidence,
                   a.period      AS period,
                   a.source      AS source,
                   a.note        AS note,
                   s.roll_number AS roll_number,
                   s.name        AS name,
                   s.department  AS department,
                   s.section     AS section
            FROM attendance a
            JOIN students   s ON s.id = a.student_id
            WHERE 1=1
        """
        params: list = []
        if date:
            query += " AND a.date = ?"
            params.append(date)
        if student_id is not None:
            query += " AND a.student_id = ?"
            params.append(student_id)
        if section is not None and section != "":
            query += " AND s.section = ?"
            params.append(section.strip())
        if period is not None and period != "":
            query += " AND a.period = ?"
            params.append(period.strip())
        query += " ORDER BY a.date DESC, a.time DESC"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if "time_out" not in df.columns:
            df["time_out"] = None
        durations: list[Optional[int]] = []
        for _, row in df.iterrows():
            durations.append(
                duration_seconds_between(
                    row.get("date"), row.get("time"), row.get("time_out")
                )
            )
        df = df.copy()
        df["duration_seconds"] = pd.Series(
            durations, index=df.index, dtype=object
        )
        return df

    def get_absentees(
        self,
        date: str,
        section: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return students with no attendance row on ``date`` (YYYY-MM-DD).

        Holiday / weekend / no-class dates return an empty frame (nobody
        is absent).
        """
        holidays: set[str] = set()
        if self.get_holiday(date) is not None:
            holidays.add(date)
        if not is_school_day(date, holidays, self.get_weekend_days()):
            return _empty_absentees_frame()
        query = """
            SELECT s.id          AS id,
                   s.roll_number AS roll_number,
                   s.name        AS name,
                   s.email       AS email,
                   s.department  AS department,
                   s.section     AS section
            FROM students s
            WHERE COALESCE(s.active, 1) = 1
              AND s.id NOT IN (
                SELECT DISTINCT student_id FROM attendance WHERE date = ?
            )
        """
        params: list = [date]
        if section is not None and section != "":
            query += " AND s.section = ?"
            params.append(section.strip())
        query += " ORDER BY s.name COLLATE NOCASE"

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def export_attendance_csv(
        self,
        out_path: Path,
        date: Optional[str] = None,
        section: Optional[str] = None,
        period: Optional[str] = None,
    ) -> Path:
        df = self.get_attendance(date=date, section=section, period=period)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    def export_absentees_csv(
        self,
        out_path: Path,
        date: str,
        section: Optional[str] = None,
    ) -> Path:
        df = self.get_absentees(date=date, section=section)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    def export_attendance_json(
        self,
        out_path: Path,
        date: Optional[str] = None,
        section: Optional[str] = None,
        period: Optional[str] = None,
    ) -> Path:
        df = self.get_attendance(date=date, section=section, period=period)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "attendance",
            "date": date,
            "section": section or "",
            "period": period or "",
            "count": 0 if df.empty else int(len(df)),
            "records": _df_records(df),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def export_absentees_json(
        self,
        out_path: Path,
        date: str,
        section: Optional[str] = None,
    ) -> Path:
        df = self.get_absentees(date=date, section=section)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "absentees",
            "date": date,
            "section": section or "",
            "count": 0 if df.empty else int(len(df)),
            "records": _df_records(df),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def daily_digest(
        self,
        date: str,
        section: Optional[str] = None,
    ) -> dict:
        """Build a summary for one day: present, late, excused, absentees, rate."""
        present_df = self.get_attendance(date=date, section=section)
        excused_df = pd.DataFrame()
        if present_df.empty:
            present_count = 0
            late_count = 0
            on_time_count = 0
            excused_count = 0
        else:
            first = (
                present_df.sort_values(["time", "id"], kind="mergesort")
                .drop_duplicates(subset=["roll_number"], keep="first")
            )
            excused_mask = first["status"].map(is_excused_status)
            excused_df = first.loc[excused_mask].copy()
            shown = first.loc[~excused_mask]
            present_count = int(shown["roll_number"].nunique()) if not shown.empty else 0
            late_count = int((shown["status"] == "Late").sum()) if not shown.empty else 0
            on_time_count = (
                int((shown["status"] == "Present").sum()) if not shown.empty else 0
            )
            excused_count = int(len(excused_df))

        holiday_row = self.get_holiday(date)
        is_holiday = holiday_row is not None
        holiday_name = (
            str(holiday_row["name"] or "").strip() or "Holiday"
            if is_holiday
            else ""
        )
        weekend_days = self.get_weekend_days()
        is_weekend = iso_weekday(date) in weekend_days
        school_day = is_school_day(
            date, {date} if is_holiday else set(), weekend_days
        )

        absentees_df = self.get_absentees(date=date, section=section)
        absentee_count = len(absentees_df)

        students = self.list_students(section=section)
        total_students = len(students)
        if not school_day:
            attendance_rate = 100.0
        else:
            attendance_rate = (
                (present_count / total_students * 100.0) if total_students else 0.0
            )

        return {
            "date": date,
            "section": section or "",
            "total_students": total_students,
            "present_count": present_count,
            "on_time_count": on_time_count,
            "late_count": late_count,
            "excused_count": excused_count,
            "absentee_count": absentee_count,
            "attendance_rate": attendance_rate,
            "holiday": is_holiday,
            "holiday_name": holiday_name,
            "weekend": is_weekend,
            "school_day": school_day,
            "absentees": absentees_df,
            "excused": excused_df,
            "present": present_df,
        }

    def write_daily_digest(
        self,
        out_path: Path,
        date: str,
        section: Optional[str] = None,
        fmt: str = "md",
    ) -> Path:
        """Write a daily digest as Markdown (default), plain text, or HTML."""
        digest = self.daily_digest(date=date, section=section)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        section_line = (
            f"Section: {digest['section']}" if digest["section"] else "Section: (all)"
        )
        rate = digest["attendance_rate"]
        absentees = digest["absentees"]

        excused = digest.get("excused")
        if excused is None:
            excused = pd.DataFrame()
        excused_count = int(digest.get("excused_count") or 0)
        kind = (fmt or "md").strip().lower()
        if kind in {"markdown", "md"}:
            kind = "md"
        elif kind in {"text", "txt"}:
            kind = "txt"
        elif kind in {"html", "htm"}:
            kind = "html"

        note = ""
        if digest.get("holiday"):
            note = f"Holiday — {digest.get('holiday_name') or 'Holiday'} (no class)"
        elif digest.get("weekend"):
            note = "Weekend (no class)"

        present_shown = pd.DataFrame()
        present_all = digest.get("present")
        if present_all is not None and not present_all.empty:
            first = (
                present_all.sort_values(["time", "id"], kind="mergesort")
                .drop_duplicates(subset=["roll_number"], keep="first")
            )
            if "status" in first.columns:
                present_shown = first.loc[~first["status"].map(is_excused_status)].copy()
            else:
                present_shown = first

        if kind == "html":
            out_path.write_text(
                _render_daily_digest_html(
                    digest,
                    section_line=section_line,
                    note=note,
                    present_shown=present_shown,
                    absentees=absentees,
                    excused=excused,
                    excused_count=excused_count,
                    rate=rate,
                ),
                encoding="utf-8",
            )
            return out_path

        if kind == "txt":
            lines = [
                f"Daily Attendance Digest — {digest['date']}",
                section_line,
            ]
            if note:
                lines.append(note)
            lines.extend(
                [
                    "",
                    f"Total students : {digest['total_students']}",
                    f"Present        : {digest['present_count']}",
                    f"On time        : {digest['on_time_count']}",
                    f"Late           : {digest['late_count']}",
                    f"Excused        : {excused_count}",
                    f"Absent         : {digest['absentee_count']}",
                    f"Attendance rate: {rate:.1f}%",
                    "",
                    "Absentees:",
                ]
            )
            if absentees.empty:
                lines.append("  (none)")
            else:
                for _, row in absentees.iterrows():
                    sec = row.get("section") or "—"
                    lines.append(
                        f"  - {row['roll_number']}: {row['name']} [{sec}]"
                    )
            lines.extend(["", "Excused:"])
            if excused.empty:
                lines.append("  (none)")
            else:
                for _, row in excused.iterrows():
                    sec = row.get("section") or "—"
                    lines.append(
                        f"  - {row['roll_number']}: {row['name']} [{sec}]"
                    )
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            # Markdown
            lines = [
                f"# Daily Attendance Digest — {digest['date']}",
                "",
                f"**{section_line}**",
            ]
            if note:
                lines.extend(["", f"_{note}_"])
            lines.extend(
                [
                    "",
                    "| Metric | Value |",
                    "|--------|------:|",
                    f"| Total students | {digest['total_students']} |",
                    f"| Present | {digest['present_count']} |",
                    f"| On time | {digest['on_time_count']} |",
                    f"| Late | {digest['late_count']} |",
                    f"| Excused | {excused_count} |",
                    f"| Absent | {digest['absentee_count']} |",
                    f"| Attendance rate | {rate:.1f}% |",
                    "",
                    "## Absentees",
                    "",
                ]
            )
            if absentees.empty:
                lines.append("_None — full attendance._")
            else:
                lines.append("| Roll | Name | Department | Section |")
                lines.append("|------|------|------------|---------|")
                for _, row in absentees.iterrows():
                    lines.append(
                        f"| {row['roll_number']} | {row['name']} | "
                        f"{row.get('department') or '—'} | "
                        f"{row.get('section') or '—'} |"
                    )
            lines.extend(["", "## Excused", ""])
            if excused.empty:
                lines.append("_None._")
            else:
                lines.append("| Roll | Name | Department | Section |")
                lines.append("|------|------|------------|---------|")
                for _, row in excused.iterrows():
                    lines.append(
                        f"| {row['roll_number']} | {row['name']} | "
                        f"{row.get('department') or '—'} | "
                        f"{row.get('section') or '—'} |"
                    )
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    # ---------------------------------------------------------------------
    # Weekly / rate stats
    # ---------------------------------------------------------------------
    def weekly_summary(
        self,
        start_date: str,
        days: int = 7,
        section: Optional[str] = None,
    ) -> pd.DataFrame:
        """Per-student present / late / excused / absent over ``days`` from start.

        The first mark of each day wins. Students with no mark that day are
        absent. Excused (or Sick) days are counted separately and are not
        absent. Holiday and weekend dates are excluded from both absent and
        the day denominator. ``attendance_rate`` is (present + late) / school
        days * 100.
        """
        datetime.strptime(start_date, "%Y-%m-%d")  # validate
        dates = week_dates(start_date, days=days)
        end_date = dates[-1]
        holiday_set = self.holiday_dates_between(start_date, end_date)
        weekend_days = self.get_weekend_days()
        dates = [
            day
            for day in dates
            if is_school_day(day, holiday_set, weekend_days)
        ]
        students = self.list_students(section=section)
        columns = [
            "student_id",
            "roll_number",
            "name",
            "section",
            "present",
            "late",
            "excused",
            "absent",
            "days",
            "attendance_rate",
        ]
        if not students:
            return pd.DataFrame(columns=columns)

        query = """
            SELECT student_id, date, time, status, id
            FROM attendance
            WHERE date >= ? AND date <= ?
            ORDER BY date, time, id
        """
        with self._connect() as conn:
            raw = pd.read_sql_query(query, conn, params=[start_date, end_date])

        first_status: dict[tuple[int, str], str] = {}
        if not raw.empty:
            for _, row in raw.iterrows():
                key = (int(row["student_id"]), str(row["date"]))
                if key not in first_status:
                    first_status[key] = str(row["status"])

        span = len(dates)
        records: list[dict] = []
        for student in students:
            sid = int(student["id"])
            present = late = excused = 0
            for day in dates:
                status = first_status.get((sid, day))
                if status is None:
                    continue
                if is_excused_status(status):
                    excused += 1
                elif status == "Late":
                    late += 1
                else:
                    present += 1
            attended = present + late
            absent = span - present - late - excused
            rate = (attended / span * 100.0) if span else 0.0
            records.append(
                {
                    "student_id": sid,
                    "roll_number": student["roll_number"],
                    "name": student["name"],
                    "section": student["section"] or "",
                    "present": present,
                    "late": late,
                    "excused": excused,
                    "absent": absent,
                    "days": span,
                    "attendance_rate": rate,
                }
            )
        return pd.DataFrame(records, columns=columns)

    def range_report(
        self,
        start_date: str,
        end_date: str,
        section: Optional[str] = None,
    ) -> pd.DataFrame:
        """Per-student present / late / excused / absent from ``start`` to ``end``.

        Holiday dates in the inclusive range are not absent and are excluded
        from the day denominator (same rules as ``weekly_summary``).
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if end < start:
            raise ValueError("end date must be on or after start date")
        days = (end - start).days + 1
        return self.weekly_summary(start_date, days=days, section=section)

    def calendar_grid(
        self,
        start_date: str,
        end_date: str,
        section: Optional[str] = None,
    ) -> dict:
        """Per-student first-mark status for each day in an inclusive range.

        Returns ``{from, to, dates, holidays, weekends, students}``.
        Archived students are skipped. Range is capped at 93 days.
        """
        start = datetime.strptime((start_date or "").strip(), "%Y-%m-%d")
        end = datetime.strptime((end_date or "").strip(), "%Y-%m-%d")
        if end < start:
            raise ValueError("end date must be on or after start date")
        span = (end - start).days + 1
        if span > CALENDAR_MAX_DAYS:
            raise ValueError(
                f"date range cannot exceed {CALENDAR_MAX_DAYS} days"
            )
        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")
        dates = week_dates(start_s, days=span)
        with self._connect() as conn:
            holiday_rows = conn.execute(
                """
                SELECT date, name FROM holidays
                WHERE date >= ? AND date <= ?
                ORDER BY date
                """,
                (start_s, end_s),
            ).fetchall()
        holidays = {
            str(row["date"]): (str(row["name"] or "").strip() or "Holiday")
            for row in holiday_rows
        }
        weekend_days = self.get_weekend_days()
        weekends = [
            day for day in dates if iso_weekday(day) in weekend_days
        ]
        students = self.list_students(section=section)
        query = """
            SELECT student_id, date, time, status, id
            FROM attendance
            WHERE date >= ? AND date <= ?
            ORDER BY date, time, id
        """
        with self._connect() as conn:
            raw = pd.read_sql_query(query, conn, params=[start_s, end_s])
        first_status: dict[tuple[int, str], str] = {}
        if not raw.empty:
            for _, row in raw.iterrows():
                key = (int(row["student_id"]), str(row["date"]))
                if key not in first_status:
                    first_status[key] = str(row["status"])

        payload_students: list[dict] = []
        for student in students:
            sid = int(student["id"])
            days: dict[str, Optional[str]] = {}
            for day in dates:
                status = first_status.get((sid, day))
                if status is None:
                    days[day] = None
                    continue
                try:
                    days[day] = normalize_attendance_status(status)
                except ValueError:
                    days[day] = status
            payload_students.append(
                {
                    "roll_number": student["roll_number"],
                    "name": student["name"],
                    "section": student["section"] or "",
                    "days": days,
                }
            )
        return {
            "from": start_s,
            "to": end_s,
            "dates": dates,
            "holidays": holidays,
            "weekends": weekends,
            "students": payload_students,
        }

    def export_attendance_ics(
        self,
        start: str,
        end: str,
        section: Optional[str] = None,
    ) -> str:
        """Return a VCALENDAR of Present/Late/Excused student-days.

        One ``VEVENT`` per student-day (first mark wins). ``UID`` is
        ``{roll}-{date}@face-recognition-attendance``. ``DTSTART`` is the
        date (VALUE=DATE). Inclusive range; archived students are skipped.
        """
        start_s = (start or "").strip()
        end_s = (end or "").strip()
        start_dt = datetime.strptime(start_s, "%Y-%m-%d")
        end_dt = datetime.strptime(end_s, "%Y-%m-%d")
        if end_dt < start_dt:
            raise ValueError("end date must be on or after start date")

        students = self.list_students(section=section)
        active_ids = {int(student["id"]) for student in students}
        meta = {
            int(student["id"]): student for student in students
        }
        query = """
            SELECT student_id, date, time, status, note, id
            FROM attendance
            WHERE date >= ? AND date <= ?
            ORDER BY date, time, id
        """
        with self._connect() as conn:
            raw = conn.execute(query, (start_s, end_s)).fetchall()

        seen: set[tuple[int, str]] = set()
        events: list[dict] = []
        for row in raw:
            sid = int(row["student_id"])
            if sid not in active_ids:
                continue
            day = str(row["date"])
            key = (sid, day)
            if key in seen:
                continue
            seen.add(key)
            try:
                status = normalize_attendance_status(str(row["status"]))
            except ValueError:
                continue
            if status not in {"Present", "Late", "Excused"}:
                continue
            student = meta[sid]
            remark = None
            if "note" in row.keys() and row["note"]:
                remark = normalize_mark_note(str(row["note"]))
            events.append(
                {
                    "roll": str(student["roll_number"]),
                    "name": str(student["name"] or ""),
                    "date": day,
                    "status": status,
                    "note": remark,
                }
            )
        events.sort(key=lambda item: (item["date"], item["roll"], item["name"]))

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Sebby1770//Face Recognition Attendance//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
        ]
        for event in events:
            uid = _ics_uid(event["roll"], event["date"])
            summary = f"{event['name']} ({event['roll']}) {event['status']}"
            lines.append("BEGIN:VEVENT")
            lines.append(f"UID:{uid}")
            lines.append(f"DTSTAMP:{stamp}")
            lines.append(
                f"DTSTART;VALUE=DATE:{event['date'].replace('-', '')}"
            )
            lines.append(f"SUMMARY:{_ics_escape(summary)}")
            if event["note"]:
                lines.append(f"DESCRIPTION:{_ics_escape(event['note'])}")
            lines.append("END:VEVENT")
        lines.append("END:VCALENDAR")
        folded = [_ics_fold_line(line) for line in lines]
        return "\r\n".join(folded) + "\r\n"

    def export_range_report_csv(
        self,
        out_path: Path,
        start_date: str,
        end_date: str,
        section: Optional[str] = None,
    ) -> Path:
        df = self.range_report(start_date, end_date, section=section)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    def export_range_report_json(
        self,
        out_path: Path,
        start_date: str,
        end_date: str,
        section: Optional[str] = None,
    ) -> Path:
        df = self.range_report(start_date, end_date, section=section)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        payload = {
            "format": "range_report",
            "from": start_date,
            "to": end_date,
            "section": section or "",
            "days": (end - start).days + 1,
            "count": 0 if df.empty else int(len(df)),
            "records": _df_records(df),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def export_range_report_html(
        self,
        out_path: Path,
        start_date: str,
        end_date: str,
        section: Optional[str] = None,
    ) -> Path:
        """Write a self-contained HTML table of ``range_report`` / weekly stats."""
        df = self.range_report(start_date, end_date, section=section)
        overall = self.weekly_rate(
            start_date,
            days=(
                datetime.strptime(end_date, "%Y-%m-%d")
                - datetime.strptime(start_date, "%Y-%m-%d")
            ).days
            + 1,
            section=section,
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            _render_range_report_html(
                df,
                start_date,
                end_date,
                section,
                overall_rate=overall,
            ),
            encoding="utf-8",
        )
        return out_path

    def weekly_rate(
        self,
        start_date: str,
        days: int = 7,
        section: Optional[str] = None,
    ) -> float:
        summary = self.weekly_summary(start_date, days=days, section=section)
        if summary.empty:
            return 0.0
        span = int(summary.iloc[0]["days"])
        slots = len(summary) * span
        attended = int(summary["present"].sum() + summary["late"].sum())
        return (attended / slots * 100.0) if slots else 0.0

    # ---------------------------------------------------------------------
    # Unknown faces
    # ---------------------------------------------------------------------
    def log_unknown_face(
        self,
        confidence: float | None = None,
        at: Optional[datetime] = None,
        path: Optional[str] = None,
    ) -> bool:
        """Insert an unknown-face row unless one was logged inside the cooldown.

        Returns True when a row was written.
        """
        now = at or datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        path_s = str(path).strip() if path is not None and str(path).strip() else None
        with self._connect() as conn:
            last = conn.execute(
                "SELECT date, time FROM unknown_faces ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if last is not None:
                last_dt = datetime.strptime(
                    f"{last['date']} {last['time']}", "%Y-%m-%d %H:%M:%S"
                )
                delta = (now - last_dt).total_seconds()
                if 0 <= delta < config.UNKNOWN_LOG_COOLDOWN_SECONDS:
                    return False
            conn.execute(
                "INSERT INTO unknown_faces (date, time, confidence, path) "
                "VALUES (?, ?, ?, ?)",
                (date_str, time_str, confidence, path_s),
            )
        return True

    def count_unknown_faces(self, date: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS c FROM unknown_faces"
        params: list = []
        if date:
            query += " WHERE date = ?"
            params.append(date)
        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()["c"])

    def list_unknown_faces(self, date: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT id, date, time, confidence, path FROM unknown_faces"
        params: list = []
        if date:
            query += " WHERE date = ?"
            params.append(date)
        query += " ORDER BY date DESC, time DESC, id DESC"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def list_unknown_crops(self, date: Optional[str] = None) -> list[dict]:
        """Unknown-face rows with optional crop path (id, date, time, confidence, path)."""
        query = "SELECT id, date, time, confidence, path FROM unknown_faces"
        params: list = []
        if date:
            query += " WHERE date = ?"
            params.append(date)
        query += " ORDER BY date DESC, time DESC, id DESC"
        with self._connect() as conn:
            raw = conn.execute(query, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "date": row["date"],
                "time": row["time"],
                "confidence": row["confidence"],
                "path": row["path"],
            }
            for row in raw
        ]

    def assign_unknown_crop(self, path: Path | str, roll_number: str) -> dict:
        """Copy an unknown-face PNG into ``FACES_DIR/<id>/`` as the next sample.

        Marks the model stale. Returns ``{roll, samples_written, dest}``.
        """
        src = Path(path)
        if not src.is_file():
            raise ValueError(f"unknown crop not found: {src}")
        roll = (roll_number or "").strip()
        student = self.get_student_by_roll(roll)
        if student is None:
            raise ValueError(f"no student with roll '{roll}'")
        sid = int(student["id"])
        dest_dir = Path(config.FACES_DIR) / str(sid)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{next_enroll_index(dest_dir):03d}.png"
        shutil.copy2(src, dest)
        from .face_engine import mark_model_stale

        mark_model_stale(f"Assigned unknown crop to {roll}")
        path_keys = {str(src)}
        try:
            path_keys.add(str(src.expanduser().resolve()))
        except OSError:
            pass
        with self._connect() as conn:
            for stored in path_keys:
                conn.execute(
                    "DELETE FROM unknown_faces WHERE path = ?", (stored,)
                )
        try:
            src.unlink()
        except OSError:
            pass
        return {
            "roll": str(student["roll_number"]),
            "samples_written": 1,
            "dest": str(dest),
        }

    def export_unknown_faces_csv(
        self,
        out_path: Path,
        date: Optional[str] = None,
    ) -> Path:
        df = self.list_unknown_faces(date=date)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    def export_unknown_faces_json(
        self,
        out_path: Path,
        date: Optional[str] = None,
    ) -> Path:
        df = self.list_unknown_faces(date=date)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "unknown_faces",
            "date": date,
            "count": 0 if df.empty else int(len(df)),
            "records": _df_records(df),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    # ---------------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------------
    def stats(self, date: Optional[str] = None) -> dict[str, int | float | str | bool]:
        today = date or datetime.now().strftime("%Y-%m-%d")
        holiday_row = self.get_holiday(today)
        is_holiday = holiday_row is not None
        holiday_name = (
            str(holiday_row["name"] or "").strip() or "Holiday"
            if is_holiday
            else ""
        )
        weekend_days = self.get_weekend_days()
        is_weekend = iso_weekday(today) in weekend_days
        school_day = is_school_day(
            today, {today} if is_holiday else set(), weekend_days
        )
        with self._connect() as conn:
            total_students = conn.execute(
                "SELECT COUNT(*) AS c FROM students WHERE COALESCE(active, 1) = 1"
            ).fetchone()["c"]
            total_records = conn.execute(
                "SELECT COUNT(*) AS c FROM attendance"
            ).fetchone()["c"]
            day_rows = conn.execute(
                """
                SELECT student_id, status
                FROM attendance
                WHERE date = ?
                ORDER BY time, id
                """,
                (today,),
            ).fetchall()
            unknown_today = conn.execute(
                "SELECT COUNT(*) AS c FROM unknown_faces WHERE date = ?",
                (today,),
            ).fetchone()["c"]
        first_status: dict[int, str] = {}
        for row in day_rows:
            sid = int(row["student_id"])
            if sid not in first_status:
                first_status[sid] = str(row["status"])
        on_time_today = 0
        late_today = 0
        excused_today = 0
        present_today = 0
        for status in first_status.values():
            if is_excused_status(status):
                excused_today += 1
                continue
            if status == "Late":
                late_today += 1
                present_today += 1
            elif status == "Present":
                on_time_today += 1
                present_today += 1
        total_students = int(total_students)
        if not school_day:
            rate = 100.0
        else:
            rate = (
                (present_today / total_students * 100.0) if total_students else 0.0
            )
        monday = week_start(today)
        return {
            "date": today,
            "total_students": total_students,
            "total_records": int(total_records),
            "present_today": present_today,
            "on_time_today": on_time_today,
            "late_today": late_today,
            "excused_today": excused_today,
            "unknown_today": int(unknown_today),
            "attendance_rate_today": rate,
            "holiday": is_holiday,
            "holiday_name": holiday_name,
            "weekend": is_weekend,
            "school_day": school_day,
            "week_start": monday,
            "week_rate": self.weekly_rate(monday, days=config.WEEK_DAYS_DEFAULT),
        }

    # ---------------------------------------------------------------------
    # Archive, alerts, holidays import, backup
    # ---------------------------------------------------------------------
    def set_student_active(self, roll_number: str, active: bool) -> None:
        roll = (roll_number or "").strip()
        if not roll:
            raise ValueError("roll number is required")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE students SET active = ? WHERE roll_number = ?",
                (1 if active else 0, roll),
            )
            if cur.rowcount == 0:
                raise ValueError(f"no student with roll '{roll}'")

    def archive_student(self, roll_number: str) -> None:
        self.set_student_active(roll_number, False)

    def restore_student(self, roll_number: str) -> None:
        self.set_student_active(roll_number, True)

    def bulk_excuse(
        self,
        date: str,
        section: Optional[str] = None,
        note: str = "bulk excuse",
        period: Optional[str] = None,
    ) -> int:
        """Mark every unmarked active student Excused for ``date``."""
        datetime.strptime(date, "%Y-%m-%d")
        absentees = self.get_absentees(date=date, section=section)
        if absentees.empty:
            return 0
        period_name, _ = self.resolve_period_start(period)
        stamped = f"{date} 08:00:00"
        marked = 0
        with self._connect() as conn:
            for _, row in absentees.iterrows():
                conn.execute(
                    """
                    INSERT INTO attendance
                        (student_id, date, time, status, confidence, period, source, note)
                    VALUES (?, ?, ?, 'Excused', NULL, ?, 'cli', ?)
                    """,
                    (int(row["id"]), date, "08:00:00", period_name, note),
                )
                attendance_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                conn.execute(
                    """
                    INSERT INTO attendance_audit
                        (attendance_id, student_id, old_status, new_status, note, changed_at)
                    VALUES (?, ?, NULL, 'Excused', ?, ?)
                    """,
                    (attendance_id, int(row["id"]), note, stamped),
                )
                marked += 1
        return marked

    def consecutive_absences(
        self,
        as_of: Optional[str] = None,
        min_days: int = 3,
        lookback: int = 14,
        section: Optional[str] = None,
    ) -> list[dict]:
        """Students with ``min_days`` school days in a row without Present/Late."""
        as_of = as_of or datetime.now().strftime("%Y-%m-%d")
        datetime.strptime(as_of, "%Y-%m-%d")
        start = (
            datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=max(1, int(lookback)) - 1)
        ).strftime("%Y-%m-%d")
        holidays = self.holiday_dates_between(start, as_of)
        weekend_days = self.get_weekend_days()
        days = [
            day
            for day in week_dates(start, days=int(lookback))
            if day <= as_of and is_school_day(day, holidays, weekend_days)
        ]
        days.reverse()
        students = self.list_students(section=section)
        with self._connect() as conn:
            raw = conn.execute(
                """
                SELECT student_id, date, status
                FROM attendance
                WHERE date >= ? AND date <= ?
                ORDER BY date, time, id
                """,
                (start, as_of),
            ).fetchall()
        first: dict[tuple[int, str], str] = {}
        for row in raw:
            key = (int(row["student_id"]), str(row["date"]))
            if key not in first:
                first[key] = str(row["status"])

        alerts: list[dict] = []
        for student in students:
            sid = int(student["id"])
            streak = 0
            for day in days:
                status = first.get((sid, day))
                if status is None:
                    streak += 1
                    continue
                if is_excused_status(status):
                    continue
                if status in {"Present", "Late"}:
                    break
                streak += 1
            if streak >= int(min_days):
                alerts.append(
                    {
                        "student_id": sid,
                        "roll_number": student["roll_number"],
                        "name": student["name"],
                        "section": student["section"] or "",
                        "consecutive_absences": streak,
                    }
                )
        alerts.sort(key=lambda item: (-int(item["consecutive_absences"]), item["name"]))
        return alerts

    def present_streak(
        self,
        roll_number: str,
        as_of: Optional[str] = None,
    ) -> int:
        """Consecutive school days ending at ``as_of`` with Present or Late.

        Holidays, weekends, and Excused days are skipped (not counted, not
        a break). Absent or unmarked school days break the streak. ``as_of``
        defaults to today.
        """
        as_of_s = (as_of or datetime.now().strftime("%Y-%m-%d")).strip()
        datetime.strptime(as_of_s, "%Y-%m-%d")
        roll = (roll_number or "").strip()
        student = self.get_student_by_roll(roll)
        if student is None:
            raise ValueError(f"no student with roll '{roll}'")
        sid = int(student["id"])
        with self._connect() as conn:
            raw = conn.execute(
                """
                SELECT date, time, status, id
                FROM attendance
                WHERE student_id = ? AND date <= ?
                ORDER BY date, time, id
                """,
                (sid, as_of_s),
            ).fetchall()
        first_by_date: dict[str, str] = {}
        for row in raw:
            day = str(row["date"])
            if day not in first_by_date:
                first_by_date[day] = str(row["status"])
        if not first_by_date:
            return 0
        holidays = self.holiday_dates_between(min(first_by_date), as_of_s)
        return _present_streak_from_days(
            first_by_date,
            holidays,
            as_of_s,
            weekend_days=self.get_weekend_days(),
        )

    def streaks_report(
        self,
        as_of: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[dict]:
        """Per-student present streaks as of ``as_of``, longest first."""
        as_of_s = (as_of or datetime.now().strftime("%Y-%m-%d")).strip()
        datetime.strptime(as_of_s, "%Y-%m-%d")
        students = self.list_students(section=section)
        if not students:
            return []
        ids = [int(student["id"]) for student in students]
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            raw = conn.execute(
                f"""
                SELECT student_id, date, time, status, id
                FROM attendance
                WHERE student_id IN ({placeholders}) AND date <= ?
                ORDER BY date, time, id
                """,
                [*ids, as_of_s],
            ).fetchall()
        by_student: dict[int, dict[str, str]] = {}
        for row in raw:
            sid = int(row["student_id"])
            day = str(row["date"])
            days = by_student.setdefault(sid, {})
            if day not in days:
                days[day] = str(row["status"])
        earliest = min((min(days) for days in by_student.values() if days), default=as_of_s)
        holidays = self.holiday_dates_between(earliest, as_of_s)
        weekend_days = self.get_weekend_days()
        records: list[dict] = []
        for student in students:
            sid = int(student["id"])
            streak = _present_streak_from_days(
                by_student.get(sid, {}),
                holidays,
                as_of_s,
                weekend_days=weekend_days,
            )
            records.append(
                {
                    "roll_number": student["roll_number"],
                    "name": student["name"],
                    "section": student["section"] or "",
                    "streak": streak,
                }
            )
        records.sort(
            key=lambda item: (
                -int(item["streak"]),
                str(item["name"] or "").lower(),
                str(item["roll_number"] or ""),
            )
        )
        return records

    def at_risk_students(
        self,
        start_date: str,
        end_date: str,
        threshold: float = 75.0,
        section: Optional[str] = None,
    ) -> list[dict]:
        """Students whose attendance rate is below ``threshold`` over the range."""
        report = self.range_report(start_date, end_date, section=section)
        if report.empty:
            return []
        flagged = report[report["attendance_rate"] < float(threshold)]
        records = []
        for _, row in flagged.iterrows():
            records.append(
                {
                    "student_id": int(row["student_id"]),
                    "roll_number": row["roll_number"],
                    "name": row["name"],
                    "section": row["section"] or "",
                    "attendance_rate": float(row["attendance_rate"]),
                    "absent": int(row["absent"]),
                    "present": int(row["present"]),
                    "late": int(row["late"]),
                    "excused": int(row["excused"]),
                }
            )
        records.sort(key=lambda item: item["attendance_rate"])
        return records

    def perfect_attendance(
        self,
        start_date: str,
        end_date: str,
        section: Optional[str] = None,
    ) -> list[dict]:
        report = self.range_report(start_date, end_date, section=section)
        if report.empty:
            return []
        perfect = report[(report["absent"] == 0) & (report["attendance_rate"] >= 100.0)]
        return _df_records(perfect)

    def import_holidays_csv(self, csv_path: Path | str) -> dict:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        added = 0
        skipped = 0
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")
            headers = {
                (name or "").strip().lower().replace(" ", "_"): name
                for name in reader.fieldnames
                if name is not None
            }
            date_key = None
            for key in ("date", "day", "holiday_date"):
                if key in headers:
                    date_key = headers[key]
                    break
            name_key = None
            for key in ("name", "label", "holiday"):
                if key in headers:
                    name_key = headers[key]
                    break
            if date_key is None:
                raise ValueError("CSV is missing a date column.")
            for raw in reader:
                day = (raw.get(date_key) or "").strip()
                if not day:
                    skipped += 1
                    continue
                label = (raw.get(name_key) or "").strip() if name_key else "Holiday"
                try:
                    self.add_holiday(day, label or "Holiday")
                except ValueError:
                    skipped += 1
                    continue
                added += 1
        return {"added": added, "skipped": skipped}

    def backup_database(self, dest: Path | str) -> Path:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_path, dest_path)
        for suffix in ("-wal", "-shm"):
            src_side = Path(str(self.db_path) + suffix)
            if src_side.is_file():
                shutil.copy2(src_side, Path(str(dest_path) + suffix))
        return dest_path

    def restore_database(self, source: Path | str) -> Path:
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(f"backup not found: {src}")
        shutil.copy2(src, self.db_path)
        for suffix in ("-wal", "-shm"):
            src_side = Path(str(src) + suffix)
            dest_side = Path(str(self.db_path) + suffix)
            if src_side.is_file():
                if src_side.resolve() != dest_side.resolve():
                    shutil.copy2(src_side, dest_side)
            elif dest_side.is_file():
                try:
                    dest_side.unlink()
                except OSError:
                    pass
        self._init_schema()
        return self.db_path

    def purge_attendance_before(self, before_date: str) -> int:
        datetime.strptime(before_date, "%Y-%m-%d")
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM attendance WHERE date < ?", (before_date,))
            return int(cur.rowcount)

    def undo_last_mark(self, roll_number: str) -> Optional[dict]:
        student = self.get_student_by_roll(roll_number)
        if student is None:
            raise ValueError(f"no student with roll '{roll_number}'")
        sid = int(student["id"])
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, date, time, status, period
                FROM attendance
                WHERE student_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM attendance WHERE id = ?", (int(row["id"]),))
            conn.execute(
                """
                INSERT INTO attendance_audit
                    (attendance_id, student_id, old_status, new_status, note, changed_at)
                VALUES (?, ?, ?, NULL, 'undo', ?)
                """,
                (
                    int(row["id"]),
                    sid,
                    str(row["status"]),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            return {
                "id": int(row["id"]),
                "date": row["date"],
                "time": row["time"],
                "status": row["status"],
                "period": row["period"],
            }


def doctor_report(db: Optional[Database] = None) -> dict:
    """Collect local diagnostics for CLI ``doctor`` and ``GET /doctor``."""
    opencv_version = None
    cv2_face = False
    cascade_path = None
    try:
        import cv2

        opencv_version = getattr(cv2, "__version__", None)
        cv2_face = hasattr(cv2, "face")
        try:
            from .face_engine import load_face_detector

            _detector, path = load_face_detector()
            cascade_path = str(path) if path is not None else None
        except Exception:  # noqa: BLE001
            cascade_path = None
    except ImportError:
        opencv_version = None
        cv2_face = False

    db_path = Path(db.db_path) if db is not None else config.DB_PATH
    db_exists = db_path.is_file()
    database = db if db is not None else Database(db_path=db_path)
    student_count = len(database.list_students(include_inactive=True))
    holiday_count = len(database.list_holidays())

    sample_folders = 0
    sample_images = 0
    faces_root = config.FACES_DIR
    if faces_root.exists():
        for entry in faces_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                int(entry.name)
            except ValueError:
                continue
            sample_folders += 1
            sample_images += count_enroll_samples(entry)

    model_exists = config.MODEL_PATH.exists() and config.LABEL_MAP_PATH.exists()
    model_stale = config.MODEL_STALE_FLAG.exists()
    return {
        "opencv_version": opencv_version,
        "cv2_face": bool(cv2_face),
        "cascade_path": cascade_path,
        "db_path": str(db_path),
        "db_exists": bool(db_exists or db_path.is_file()),
        "student_count": int(student_count),
        "sample_folders": int(sample_folders),
        "sample_images": int(sample_images),
        "model_exists": bool(model_exists),
        "model_stale": bool(model_stale),
        "holiday_count": int(holiday_count),
        "ok": bool(cv2_face),
    }

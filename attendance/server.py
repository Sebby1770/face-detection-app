"""Tiny local-only JSON API for attendance records (stdlib http.server)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .database import Database, _df_records, doctor_report

_MAX_BODY_BYTES = 1_000_000
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class AttendanceAPIHandler(BaseHTTPRequestHandler):
    """Local JSON API. Bind 127.0.0.1 only."""

    server_version = f"AttendanceAPI/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in _CORS_HEADERS.items():
            self.send_header(key, value)
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise ValueError("payload too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self._json(200, {"status": "ok", "version": __version__})
            return
        if path == "/doctor":
            try:
                self._json(200, doctor_report())
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
            return
        try:
            db = Database()
            qs = parse_qs(parsed.query)
            if path == "/stats":
                self._json(200, db.stats())
                return
            if path == "/students":
                rows = db.list_students()
                students = [
                    {
                        "id": int(r["id"]),
                        "roll_number": r["roll_number"],
                        "name": r["name"],
                        "email": r["email"] or "",
                        "department": r["department"] or "",
                        "section": r["section"] or "",
                        "active": int(r["active"] if "active" in r.keys() else 1),
                    }
                    for r in rows
                ]
                self._json(200, {"count": len(students), "students": students})
                return
            if path == "/roster":
                raw_min = (qs.get("min") or ["8"])[0]
                try:
                    min_samples = int(raw_min)
                except (TypeError, ValueError):
                    self._json(400, {"error": "invalid min"})
                    return
                rows = db.enrollment_roster(min_samples=min_samples)
                self._json(
                    200,
                    {
                        "count": len(rows),
                        "min_samples": min_samples,
                        "students": rows,
                    },
                )
                return
            if path == "/attendance":
                date = (qs.get("date") or [None])[0]
                if date:
                    date = date.strip()
                    try:
                        datetime.strptime(date, "%Y-%m-%d")
                    except ValueError:
                        self._json(
                            400,
                            {"error": "invalid date; use YYYY-MM-DD"},
                        )
                        return
                df = db.get_attendance(date=date)
                records = _df_records(df)
                self._json(
                    200,
                    {
                        "date": date,
                        "count": len(records),
                        "records": records,
                    },
                )
                return
            if path == "/holidays":
                rows = db.list_holidays()
                holidays = [{"date": r["date"], "name": r["name"] or ""} for r in rows]
                self._json(200, {"count": len(holidays), "holidays": holidays})
                return
            if path == "/alerts":
                today = datetime.now().strftime("%Y-%m-%d")
                start = (qs.get("from") or [None])[0] or (
                    datetime.now() - timedelta(days=6)
                ).strftime("%Y-%m-%d")
                end = (qs.get("to") or [None])[0] or today
                try:
                    threshold = float((qs.get("threshold") or ["75"])[0])
                except ValueError:
                    self._json(400, {"error": "invalid threshold"})
                    return
                self._json(
                    200,
                    {
                        "from": start,
                        "to": end,
                        "threshold": threshold,
                        "at_risk": db.at_risk_students(start, end, threshold=threshold),
                        "consecutive": db.consecutive_absences(as_of=end, min_days=3),
                    },
                )
                return
            if path == "/streaks":
                as_of = (qs.get("as_of") or [None])[0] or datetime.now().strftime(
                    "%Y-%m-%d"
                )
                as_of = as_of.strip()
                section = (qs.get("section") or [None])[0]
                try:
                    datetime.strptime(as_of, "%Y-%m-%d")
                    rows = db.streaks_report(
                        as_of=as_of,
                        section=(section.strip() if section else None),
                    )
                except ValueError as exc:
                    message = str(exc)
                    if "does not match format" in message or "unconverted data" in message:
                        message = "invalid date; use YYYY-MM-DD"
                    self._json(400, {"error": message})
                    return
                self._json(
                    200,
                    {
                        "as_of": as_of,
                        "section": (section or "").strip(),
                        "count": len(rows),
                        "streaks": rows,
                    },
                )
                return
            if path in {"/calendar", "/calendar.ics"}:
                start = (qs.get("from") or [None])[0]
                end = (qs.get("to") or [None])[0]
                section = (qs.get("section") or [None])[0]
                if not start or not end:
                    self._json(
                        400,
                        {"error": "from and to are required (YYYY-MM-DD)"},
                    )
                    return
                start = start.strip()
                end = end.strip()
                section_s = section.strip() if section else None
                try:
                    datetime.strptime(start, "%Y-%m-%d")
                    datetime.strptime(end, "%Y-%m-%d")
                    if path == "/calendar.ics":
                        ics = db.export_attendance_ics(
                            start, end, section=section_s
                        )
                        body = ics.encode("utf-8")
                        self._send(
                            200,
                            body,
                            "text/calendar; charset=utf-8",
                            {
                                "Content-Disposition": 'attachment; filename="attendance.ics"',
                            },
                        )
                        return
                    grid = db.calendar_grid(start, end, section=section_s)
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                self._json(200, grid)
                return
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for key, value in _CORS_HEADERS.items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_BODY_BYTES:
            self._json(413, {"error": "payload too large"})
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            code = 413 if "too large" in str(exc) else 400
            self._json(code, {"error": str(exc)})
            return
        roll = str(payload.get("roll") or payload.get("roll_number") or "").strip()
        pin = payload.get("pin")
        pin_s = str(pin).strip() if pin is not None and str(pin).strip() != "" else None

        if path == "/pin":
            if not roll:
                self._json(400, {"error": "roll is required"})
                return
            if pin_s is None:
                self._json(400, {"error": "pin is required"})
                return
            try:
                db = Database()
                db.set_pin(roll, pin_s)
            except ValueError as exc:
                code = 404 if "no student" in str(exc) else 400
                self._json(code, {"error": str(exc)})
                return
            self._json(200, {"ok": True, "roll": roll})
            return

        if path != "/mark":
            self._json(404, {"error": "not found"})
            return
        if not roll:
            self._json(400, {"error": "roll is required"})
            return
        try:
            db = Database()
            student = db.get_student_by_roll(roll)
            if student is None:
                self._json(404, {"error": f"no student with roll '{roll}'"})
                return
            at = None
            if payload.get("at"):
                at = datetime.fromisoformat(str(payload["at"]))
            note = payload.get("note")
            touch = bool(payload.get("touch") or payload.get("out"))
            if pin_s:
                inserted = db.mark_with_pin(
                    roll,
                    pin_s,
                    status=payload.get("status"),
                    at=at,
                    period=payload.get("period"),
                    source="pin",
                    note=note,
                    touch=touch,
                )
            else:
                inserted = db.mark_attendance(
                    student_id=int(student["id"]),
                    status=payload.get("status"),
                    at=at,
                    period=payload.get("period"),
                    source="api",
                    note=note,
                    touch=touch,
                )
            self._json(
                200,
                {
                    "ok": True,
                    "inserted": inserted,
                    "roll": roll,
                    "name": student["name"],
                    "source": "pin" if pin_s else "api",
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": str(exc)})


def make_server(
    host: str = "127.0.0.1", port: int = 8768
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("API binds local-only (127.0.0.1)")
    return ThreadingHTTPServer((host, int(port)), AttendanceAPIHandler)


def serve_forever(host: str = "127.0.0.1", port: int = 8768) -> None:
    httpd = make_server(host, port)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

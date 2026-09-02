"""Faces — detect, redact, and take attendance from one CLI."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

USAGE = """\
Faces — detect, redact, and take attendance. Local only.

Usage:
  faces redact [options]              Privacy redaction (YuNet / Haar)
  faces attendance [command]          Recognition + SQLite attendance
  faces --help

Redact:
  faces redact portrait.jpg --output out.png --headless
  faces redact --input clip.mp4 --output out.mp4 --headless --redaction pixelate
  python face_detection.py …          (same engine, original flags)

Attendance:
  faces attendance                    Desktop GUI
  faces attendance students add --roll 1 --name Ada
  faces attendance mark --roll 1
  faces attendance report digest --date today
  faces attendance serve --port 8768
  python -m attendance.cli …          (same engine)
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, rest = args[0], args[1:]
    if cmd in ("redact", "detect", "privacy"):
        from face_detection import main as redact_main

        return int(redact_main(rest))
    if cmd in ("attendance", "attend"):
        from attendance.cli import main as attendance_main

        return int(attendance_main(rest))
    print(f"Unknown command: {cmd}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

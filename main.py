"""Backward-compatible attendance CLI entry (`from main import main`)."""

from attendance.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

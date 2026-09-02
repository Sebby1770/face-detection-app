"""Unified Faces CLI dispatcher."""

from faces import main


def test_help() -> None:
    assert main(["--help"]) == 0


def test_unknown_command() -> None:
    assert main(["nope"]) == 2


def test_redact_help() -> None:
    try:
        code = main(["redact", "--help"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 0


def test_attendance_help() -> None:
    try:
        code = main(["attendance", "--help"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 0

from pathlib import Path

import pytest

from face_detection import parse_args


def test_privacy_defaults_are_conservative():
    args = parse_args([])

    assert args.redaction == "solid"
    assert args.padding == 0.25
    assert args.hold_frames == 2
    assert not args.no_redaction
    assert not args.overlays


def test_headless_processing_requires_an_output(tmp_path: Path):
    input_path = tmp_path / "input.png"
    input_path.touch()

    with pytest.raises(SystemExit):
        parse_args(["--input", str(input_path), "--headless"])


def test_input_and_output_must_be_different(tmp_path: Path):
    input_path = tmp_path / "input.png"
    input_path.touch()

    with pytest.raises(SystemExit):
        parse_args(
            ["--input", str(input_path), "--output", str(input_path), "--headless"]
        )


def test_existing_output_requires_explicit_overwrite(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    input_path.touch()
    output_path.touch()

    with pytest.raises(SystemExit):
        parse_args(
            ["--input", str(input_path), "--output", str(output_path), "--headless"]
        )

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--headless",
            "--overwrite",
        ]
    )
    assert args.overwrite


@pytest.mark.parametrize("option", ["--score-threshold", "--nms-threshold", "--padding"])
def test_probability_options_are_bounded(option):
    with pytest.raises(SystemExit):
        parse_args([option, "1.1"])


def test_hex_color_is_converted_to_opencv_bgr():
    args = parse_args(["--solid-color", "#123456"])

    assert args.solid_color == (0x56, 0x34, 0x12)

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
    assert args.mask == "box"
    assert args.feather == 0
    assert args.min_size == 0
    assert args.redact_ids is None
    assert args.keep_ids is None
    assert not args.review
    assert args.pose is None
    assert not args.identify


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


def test_parse_redact_and_keep_ids():
    args = parse_args(["--redact-ids", "1,3", "--keep-ids", "2"])

    assert args.redact_ids == [1, 3]
    assert args.keep_ids == [2]


def test_invalid_redact_ids_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--redact-ids", "1,abc"])


def test_positional_source_is_alias_of_input(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    input_path.touch()

    args = parse_args([str(input_path), "--output", str(output_path), "--headless"])

    assert args.input == input_path
    assert args.output == output_path


def test_positional_and_input_must_agree(tmp_path: Path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.touch()
    second.touch()

    with pytest.raises(SystemExit):
        parse_args([str(first), "--input", str(second), "--headless", "--output", str(tmp_path / "out.png")])


def test_pixelate_shortcut_selects_redaction_mode():
    args = parse_args(["--pixelate"])

    assert args.redaction == "pixelate"


def test_directory_input_implies_headless_and_requires_directory_output(tmp_path: Path):
    input_dir = tmp_path / "inbox"
    output_dir = tmp_path / "redacted"
    input_dir.mkdir()

    args = parse_args(["--input", str(input_dir), "--output", str(output_dir)])

    assert args.batch
    assert args.headless
    assert args.input == input_dir
    assert args.output == output_dir

    existing_file = tmp_path / "out.png"
    existing_file.touch()
    with pytest.raises(SystemExit):
        parse_args(["--input", str(input_dir), "--output", str(existing_file)])


def test_mask_and_feather_flags():
    args = parse_args(["--mask", "ellipse", "--feather", "8"])

    assert args.mask == "ellipse"
    assert args.feather == 8


def test_min_size_flag():
    args = parse_args(["--min-size", "30"])

    assert args.min_size == 30


def test_negative_min_size_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--min-size", "-1"])


def test_identify_is_rejected_and_mentions_crops(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--identify"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "crops" in err

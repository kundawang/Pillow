from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from PIL import GifImagePlugin, Image, ImageSequence

from .helper import skip_unless_feature

if TYPE_CHECKING:
    from pathlib import Path

RGB_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (0, 0, 0),
)
COLOR_MODES = (
    "CMYK",
    "HSV",
    "RGBX",
    "YCbCr",
    pytest.param("LAB", marks=skip_unless_feature("littlecms2")),
)
ALPHA_MODES = ("PA", "RGBa")


def _block_image(mode: str, frame: int = 0) -> Image.Image:
    alpha = mode in ALPHA_MODES
    im = Image.new("RGBA" if alpha else "RGB", (len(RGB_COLORS), 1))
    for x in range(len(RGB_COLORS)):
        color = RGB_COLORS[(x + 2 * frame) % len(RGB_COLORS)]
        if alpha:
            im.putpixel((x, 0), (*color, 0 if x == 0 else 255))
        else:
            im.putpixel((x, 0), color)
    return im.convert(mode)


def _assert_matching_blocks(actual: Image.Image, expected: Image.Image) -> None:
    expected_rgba = expected.convert("RGBA")
    actual_rgba = actual.convert("RGBA")
    assert actual_rgba.size == expected_rgba.size

    for x in range(expected_rgba.width):
        expected_pixel = expected_rgba.getpixel((x, 0))
        actual_pixel = actual_rgba.getpixel((x, 0))
        assert isinstance(expected_pixel, tuple)
        assert isinstance(actual_pixel, tuple)
        if expected_pixel[3] == 0:
            assert actual_pixel[3] == 0
        else:
            assert actual_pixel[3] == 255
            for expected_value, actual_value in zip(
                expected_pixel[:3], actual_pixel[:3]
            ):
                assert abs(expected_value - actual_value) <= 2


@pytest.mark.parametrize("mode", COLOR_MODES + ALPHA_MODES)
def test_save_modes_with_fixed_blocks(tmp_path: Path, mode: str) -> None:
    out = tmp_path / "blocks.gif"
    im = _block_image(mode)
    im.save(out)

    with Image.open(out) as reread:
        assert isinstance(reread, GifImagePlugin.GifImageFile)
        assert reread.n_frames == 1
        _assert_matching_blocks(reread, im)


@pytest.mark.parametrize("mode", COLOR_MODES + ALPHA_MODES)
def test_save_modes_multiple_frames(tmp_path: Path, mode: str) -> None:
    frames = [_block_image(mode, frame) for frame in range(3)]
    durations = [100, 200, 300]
    out = tmp_path / "blocks.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
    )

    with Image.open(out) as reread:
        assert isinstance(reread, GifImagePlugin.GifImageFile)
        assert reread.n_frames == len(frames)
        for frame, expected, duration in zip(
            ImageSequence.Iterator(reread), frames, durations
        ):
            assert frame.info["duration"] == duration
            _assert_matching_blocks(frame, expected)


@pytest.mark.parametrize("mode", COLOR_MODES + ALPHA_MODES)
def test_save_modes_with_palette(tmp_path: Path, mode: str) -> None:
    im = _block_image(mode)
    rgba = im.convert("RGBA")
    palette: list[int] = []
    for x in range(rgba.width):
        pixel = rgba.getpixel((x, 0))
        assert isinstance(pixel, tuple)
        palette.extend(pixel[:3])
    palette.extend([0] * (768 - len(palette)))

    out = tmp_path / "blocks.gif"
    im.save(out, palette=bytes(palette))

    with Image.open(out) as reread:
        assert isinstance(reread, GifImagePlugin.GifImageFile)
        assert reread.n_frames == 1
        _assert_matching_blocks(reread, im)


def test_unsupported_mode_fails_explicitly(tmp_path: Path) -> None:
    out = tmp_path / "unsupported.gif"
    with pytest.raises(ValueError, match="conversion from La to L not supported"):
        Image.new("La", (1, 1)).save(out)

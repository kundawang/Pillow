from __future__ import annotations

import struct

import pytest

from PIL import Image

from .helper import hopper

I16_MODES = ("I;16", "I;16L", "I;16B")


def test_sanity() -> None:
    bbox = hopper().getbbox()
    assert isinstance(bbox, tuple)


def test_bbox() -> None:
    def check(im: Image.Image, fill_color: int | tuple[int, ...]) -> None:
        assert im.getbbox() is None

        im.paste(fill_color, (10, 25, 90, 75))
        assert im.getbbox() == (10, 25, 90, 75)

        im.paste(fill_color, (25, 10, 75, 90))
        assert im.getbbox() == (10, 10, 90, 90)

        im.paste(fill_color, (-10, -10, 110, 110))
        assert im.getbbox() == (0, 0, 100, 100)

    # 8-bit and 16-bit integer modes
    for mode in ("1", "L", "I", "P", *I16_MODES):
        im = Image.new(mode, (100, 100), 0)
        check(im, 255)

    # 32-bit mode
    im = Image.new("RGB", (100, 100), 0)
    check(im, 255)

    for mode in ("RGBA", "RGBa"):
        for rgba_color in ((0, 0, 0, 0), (127, 127, 127, 0), (255, 255, 255, 0)):
            im = Image.new(mode, (100, 100), rgba_color)
            check(im, (255, 255, 255, 255))

    for mode in ("La", "LA", "PA"):
        for color in ((0, 0), (127, 0), (255, 0)):
            im = Image.new(mode, (100, 100), color)
            check(im, (255, 255))


@pytest.mark.parametrize("mode", ("L", "RGB", "I", "F", "I;16", "I;16L", "I;16B"))
@pytest.mark.parametrize(
    "box",
    (
        (0, 0, 1, 1),  # single pixel, top-left
        (99, 0, 100, 1),  # single pixel, top-right
        (0, 99, 1, 100),  # single pixel, bottom-left
        (99, 99, 100, 100),  # single pixel, bottom-right
        (0, 0, 100, 1),  # full top row
        (0, 99, 100, 100),  # full bottom row
        (0, 0, 1, 100),  # full left column
        (99, 0, 100, 100),  # full right column
        (40, 40, 60, 60),  # centred block
    ),
)
def test_bbox_edges(mode: str, box: tuple[int, int, int, int]) -> None:
    im = Image.new(mode, (100, 100), 0)
    bands = Image.getmodebands(mode)
    im.paste((255,) * bands, box)
    assert im.getbbox() == box


def _i16_image(mode: str, rows: list[list[int]]) -> Image.Image:
    byteorder = ">" if mode == "I;16B" else "<"
    data = b"".join(
        struct.pack(byteorder + "H", value) for row in rows for value in row
    )
    return Image.frombytes(mode, (len(rows[0]), len(rows)), data)


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize(
    "rows, expected",
    (
        ([[0, 0], [0, 1]], (1, 1, 2, 2)),  # low byte only, bottom-right
        ([[0, 0], [0, 255]], (1, 1, 2, 2)),  # low byte only
        ([[0, 0], [0, 256]], (1, 1, 2, 2)),  # high byte only
        ([[0, 0], [0, 65280]], (1, 1, 2, 2)),  # high byte only
        ([[0, 0], [0, 65535]], (1, 1, 2, 2)),  # both bytes
        (
            [
                [1, 0, 256],
                [0, 0, 0],
                [255, 0, 65280],
            ],
            (0, 0, 3, 3),  # all four corners have one non-zero byte
        ),
        (
            [
                [0, 256, 0],
                [1, 0, 65280],
                [0, 255, 0],
            ],
            (0, 0, 3, 3),  # non-zero pixels touch all four edges
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 256, 1, 0],
                [0, 255, 65280, 0],
                [0, 0, 0, 0],
            ],
            (1, 1, 3, 3),  # central two-by-two block
        ),
    ),
)
def test_bbox_i16_small_matrices(
    mode: str,
    rows: list[list[int]],
    expected: tuple[int, int, int, int],
) -> None:
    bbox = _i16_image(mode, rows).getbbox()
    assert bbox == expected


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("value", (1, 255, 256, 65280, 65535))
@pytest.mark.parametrize("position", ((2, 2), (0, 0), (4, 0), (0, 4), (4, 4)))
def test_bbox_i16_single_points(
    mode: str, value: int, position: tuple[int, int]
) -> None:
    im = Image.new(mode, (5, 5))
    im.putpixel(position, value)
    x, y = position
    assert im.getbbox() == (x, y, x + 1, y + 1)


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("size", ((0, 0), (0, 1), (1, 0)))
def test_bbox_i16_empty(mode: str, size: tuple[int, int]) -> None:
    assert Image.new(mode, size).getbbox() is None


@pytest.mark.parametrize("mode", I16_MODES)
def test_bbox_i16_all_zero(mode: str) -> None:
    assert Image.new(mode, (4, 3)).getbbox() is None


@pytest.mark.parametrize("mode", I16_MODES)
def test_bbox_i16_cropped(mode: str) -> None:
    im = _i16_image(
        mode,
        [
            [0, 0, 0, 0, 0],
            [0, 1, 256, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 256, 65535, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )

    cases = (
        ((1, 1, 3, 4), (0, 0, 2, 3)),
        ((2, 1, 3, 4), (0, 0, 1, 3)),
        ((0, 0, 2, 2), (1, 1, 2, 2)),
        ((2, 2, 5, 5), (0, 1, 1, 2)),
        ((3, 3, 5, 5), None),
    )
    for crop_box, expected in cases:
        assert im.crop(crop_box).getbbox() == expected


@pytest.mark.parametrize("mode", ("RGBA", "RGBa", "La", "LA", "PA"))
def test_bbox_alpha_only_false(mode: str) -> None:
    im = Image.new(mode, (100, 100))
    assert im.getbbox(alpha_only=False) is None

    fill_color = [1] * Image.getmodebands(mode)
    fill_color[-1] = 0
    im.paste(tuple(fill_color), (25, 25, 75, 75))
    assert im.getbbox(alpha_only=False) == (25, 25, 75, 75)

    assert im.getbbox() is None

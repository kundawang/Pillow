from __future__ import annotations

import struct

import pytest

from PIL import Image, ImageDraw

SIZE = 8
INKS = (0, 256, 65535)
I16_MODES = ("I;16", "I;16L", "I;16B", "I;16N")
CASES = (
    "point",
    "line_width_1",
    "line_width_3",
    "rectangle_fill",
    "rectangle_outline_width_1",
    "rectangle_outline_width_3",
    "polygon_fill",
    "polygon_outline_width_1",
    "polygon_outline_width_4",
    "ellipse_fill",
    "ellipse_outline_width_2",
    "bitmap_mode_1_mask",
    "bitmap_l_mask",
)


def seeded_image(mode: str) -> Image.Image:
    image = Image.new(mode, (SIZE, SIZE))
    pixels = image.load()
    values = (0, 256, 65535)
    for y in range(SIZE):
        for x in range(SIZE):
            pixels[x, y] = values[(x + y) % len(values)]
    return image


def draw_case(mode: str, case: str, ink: int) -> Image.Image:
    image = seeded_image(mode)
    draw = ImageDraw.Draw(image)

    if case == "point":
        draw.point((4, 4), fill=ink)
    elif case == "line_width_1":
        draw.line((0, 2, 7, 5), fill=ink, width=1)
    elif case == "line_width_3":
        draw.line((0, 6, 7, 1), fill=ink, width=3)
    elif case == "rectangle_fill":
        draw.rectangle((1, 1, 6, 6), fill=ink)
    elif case == "rectangle_outline_width_1":
        draw.rectangle((0, 0, 7, 7), outline=ink, width=1)
    elif case == "rectangle_outline_width_3":
        draw.rectangle((0, 0, 7, 7), outline=ink, width=3)
    elif case == "polygon_fill":
        draw.polygon([(1, 6), (3, 1), (6, 6)], fill=ink)
    elif case == "polygon_outline_width_1":
        draw.polygon([(1, 6), (3, 1), (6, 6)], outline=ink)
    elif case == "polygon_outline_width_4":
        # Widths greater than one are rasterized through a mode "1" mask.
        draw.polygon([(0, 0), (7, 0), (7, 7), (0, 7)], outline=ink, width=4)
    elif case == "ellipse_fill":
        draw.ellipse((1, 1, 6, 6), fill=ink)
    elif case == "ellipse_outline_width_2":
        draw.ellipse((0, 0, 7, 7), outline=ink, width=2)
    elif case == "bitmap_mode_1_mask":
        mask = Image.new("1", (4, 4), 0)
        ImageDraw.Draw(mask).rectangle((0, 0, 3, 3), fill=1)
        draw.bitmap((2, 2), mask, fill=ink)
    elif case == "bitmap_l_mask":
        mask = Image.new("L", (4, 4), 255)
        draw.bitmap((2, 2), mask, fill=ink)
    else:
        msg = f"unknown drawing case: {case}"
        raise AssertionError(msg)

    return image


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("ink", INKS)
@pytest.mark.parametrize("case", CASES)
def test_i16_drawing_matches_i_mode(mode: str, ink: int, case: str) -> None:
    reference = draw_case("I", case, ink)
    expected = [reference.getpixel((x, y)) for y in range(SIZE) for x in range(SIZE)]

    image = draw_case(mode, case, ink)
    actual = [image.getpixel((x, y)) for y in range(SIZE) for x in range(SIZE)]

    assert actual == expected
    if mode == "I;16B":
        assert image.mode == "I;16B"
        assert image.tobytes() == b"".join(struct.pack(">H", value) for value in actual)


@pytest.mark.parametrize("mode", (*I16_MODES, "L"))
@pytest.mark.parametrize("color", (1.5, (1, 2, 3)))
def test_invalid_i16_color_errors_are_unchanged(
    mode: str, color: float | tuple[int, int, int]
) -> None:
    draw = ImageDraw.Draw(Image.new(mode, (2, 2)))
    with pytest.raises(TypeError, match="color must be int or single-element tuple"):
        draw.point((0, 0), fill=color)

from __future__ import annotations

import math
import struct
import sys
from typing import Literal, cast

import pytest

from PIL import Image, ImageFilter

sample = Image.new("L", (7, 5))
sample.putdata(sum([
    [210, 50,  20,  10,  220, 230, 80],
    [190, 210, 20,  180, 170, 40,  110],
    [120, 210, 250, 60,  220, 0,   220],
    [220, 40,  230, 80,  130, 250, 40],
    [250, 0,   80,  30,  60,  20,  110],
], []))  # fmt: skip

I16_MODES = ("I;16", "I;16L", "I;16B", "I;16N")


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _box_blur_weights(radius: float) -> tuple[int, int, int]:
    # Mirror the fixed point weights computed in ImagingHorizontalBoxBlur(),
    # including its single precision arithmetic.
    integer_radius = int(radius)
    ww = int(_float32(_float32(float(1 << 24)) / _float32(radius * 2.0 + 1.0)))
    fw = ((1 << 24) - (integer_radius * 2 + 1) * ww) // 2
    return integer_radius, ww, fw


def _box_blur_line_reference(values: list[int], radius: float) -> list[int]:
    """Reference box blur of one line, averaging with clamped edge extension."""
    integer_radius, ww, fw = _box_blur_weights(radius)
    last = len(values) - 1
    result = []
    for x in range(len(values)):
        acc = sum(
            values[min(max(i, 0), last)]
            for i in range(x - integer_radius, x + integer_radius + 1)
        )
        left = values[min(max(x - integer_radius - 1, 0), last)]
        right = values[min(max(x + integer_radius + 1, 0), last)]
        bulk = acc * ww + (left + right) * fw
        result.append((bulk + (1 << 23)) >> 24)
    return result


def _transpose(rows: list[list[int]]) -> list[list[int]]:
    return [list(column) for column in zip(*rows)]


def _box_blur_reference(
    rows: list[list[int]], xradius: float, yradius: float, passes: int = 1
) -> list[list[int]]:
    """Reference for ImagingBoxBlur(): n horizontal passes in each direction."""
    result = [list(row) for row in rows]
    if xradius == 0 and yradius == 0:
        return result
    if xradius != 0:
        for _ in range(passes):
            result = [_box_blur_line_reference(row, xradius) for row in result]
    if yradius != 0:
        transposed = _transpose(result)
        for _ in range(passes):
            transposed = [_box_blur_line_reference(row, yradius) for row in transposed]
        result = _transpose(transposed)
    return result


def _gaussian_blur_radius(radius: float, passes: int) -> float:
    # Mirror _gaussian_blur_radius() in BoxBlur.c.
    sigma2 = _float32(_float32(radius * radius) / _float32(passes))
    length = _float32(math.sqrt(_float32(12.0 * sigma2 + 1.0)))
    integer_part = _float32(math.floor(_float32(_float32(length - 1.0) / 2.0)))
    l_term = _float32(integer_part * _float32(integer_part + 1))
    numerator = _float32(
        _float32(2 * integer_part + 1) * _float32(l_term - _float32(3 * sigma2))
    )
    denominator = _float32(
        6 * _float32(sigma2 - _float32(_float32(integer_part + 1) ** 2))
    )
    return _float32(integer_part + _float32(numerator / denominator))


def _gaussian_blur_reference(
    rows: list[list[int]], xradius: float, yradius: float, passes: int = 3
) -> list[list[int]]:
    return _box_blur_reference(
        rows,
        _gaussian_blur_radius(xradius, passes),
        _gaussian_blur_radius(yradius, passes),
        passes,
    )


def _image_rows(im: Image.Image) -> list[list[int]]:
    data = cast("tuple[int, ...]", im.get_flattened_data())
    return [list(data[y * im.width : (y + 1) * im.width]) for y in range(im.height)]


def _image_from_rows(rows: list[list[int]], mode: str) -> Image.Image:
    im = Image.new(mode, (len(rows[0]), len(rows)))
    im.putdata([value for row in rows for value in row])
    return im


def test_imageops_box_blur() -> None:
    i = sample.filter(ImageFilter.BoxBlur(1))
    assert i.mode == sample.mode
    assert i.size == sample.size
    assert isinstance(i, Image.Image)


def box_blur(image: Image.Image, radius: float = 1, n: int = 1) -> Image.Image:
    return image._new(image.im.box_blur((radius, radius), n))


def assert_image(im: Image.Image, data: list[list[int]], delta: int = 0) -> None:
    it = iter(im.get_flattened_data())
    for data_row in data:
        im_row = []
        for _ in range(im.width):
            im_v = next(it)
            assert isinstance(im_v, (int, float))
            im_row.append(im_v)
        if any(abs(data_v - im_v) > delta for data_v, im_v in zip(data_row, im_row)):
            assert im_row == data_row
    with pytest.raises(StopIteration):
        next(it)


def assert_blur(
    im: Image.Image,
    radius: float,
    data: list[list[int]],
    passes: int = 1,
    delta: int = 0,
) -> None:
    # check grayscale image
    assert_image(box_blur(im, radius, passes), data, delta)
    rgba = Image.merge("RGBA", (im, im, im, im))
    for band in box_blur(rgba, radius, passes).split():
        assert_image(band, data, delta)


def test_color_modes() -> None:
    with pytest.raises(ValueError):
        box_blur(sample.convert("1"))
    with pytest.raises(ValueError):
        box_blur(sample.convert("P"))
    box_blur(sample.convert("L"))
    box_blur(sample.convert("LA"))
    box_blur(sample.convert("LA").convert("La"))
    with pytest.raises(ValueError):
        box_blur(sample.convert("I"))
    with pytest.raises(ValueError):
        box_blur(sample.convert("F"))
    box_blur(sample.convert("RGB"))
    box_blur(sample.convert("RGBA"))
    box_blur(sample.convert("RGBA").convert("RGBa"))
    box_blur(sample.convert("CMYK"))
    with pytest.raises(ValueError):
        box_blur(sample.convert("YCbCr"))


@pytest.mark.parametrize("size", ((0, 1), (1, 0)))
def test_zero_dimension(size: tuple[int, int]) -> None:
    assert box_blur(Image.new("L", size)).size == size


def test_radius_0() -> None:
    assert_blur(
        sample,
        0,
        [
            [210, 50,  20,  10,  220, 230, 80],
            [190, 210, 20,  180, 170, 40,  110],
            [120, 210, 250, 60,  220, 0,   220],
            [220, 40,  230, 80,  130, 250, 40],
            [250, 0,   80,  30,  60,  20,  110],
        ],  # fmt: skip
    )


def test_radius_0_02() -> None:
    assert_blur(
        sample,
        0.02,
        [
            [206, 55,  20,  17,  215, 223, 83],
            [189, 203, 31,  171, 169, 46,  110],
            [125, 206, 241, 69,  210, 13,  210],
            [215, 49,  221, 82,  131, 235, 48],
            [244, 7,   80,  32,  60,  27,  107],
        ],  # fmt: skip
        delta=2,
    )


def test_radius_0_05() -> None:
    assert_blur(
        sample,
        0.05,
        [
            [202, 62,  22,  27,  209, 215, 88],
            [188, 194, 44,  161, 168, 56,  111],
            [131, 201, 229, 81,  198, 31,  198],
            [209, 62,  209, 86,  133, 216, 59],
            [237, 17,  80,  36,  60,  35,  103],
        ],  # fmt: skip
        delta=2,
    )


def test_radius_0_1() -> None:
    assert_blur(
        sample,
        0.1,
        [
            [196, 72,  24,  40,  200, 203, 93],
            [187, 183, 62,  148, 166, 68,  111],
            [139, 193, 213, 96,  182, 54,  182],
            [201, 78,  193, 91,  133, 191, 73],
            [227, 31,  80,  42,  61,  47,  99],
        ],  # fmt: skip
        delta=1,
    )


def test_radius_0_5() -> None:
    assert_blur(
        sample,
        0.5,
        [
            [176, 101, 46,  83,  163, 165, 111],
            [176, 149, 108, 122, 144, 120, 117],
            [164, 171, 159, 141, 134, 119, 129],
            [170, 136, 133, 114, 116, 124, 109],
            [184, 95,  72,  70,  69,  81,  89],
        ],  # fmt: skip
        delta=1,
    )


def test_radius_1() -> None:
    assert_blur(
        sample,
        1,
        [
            [170, 109, 63,  97,  146, 153, 116],
            [168, 142, 112, 128, 126, 143, 121],
            [169, 166, 142, 149, 126, 131, 114],
            [159, 156, 109, 127, 94,  117, 112],
            [164, 128, 63,  87,  76,  89,  90],
        ],  # fmt: skip
        delta=1,
    )


def test_radius_1_5() -> None:
    assert_blur(
        sample,
        1.5,
        [
            [155, 120, 105, 112, 124, 137, 130],
            [160, 136, 124, 125, 127, 134, 130],
            [166, 147, 130, 125, 120, 121, 119],
            [168, 145, 119, 109, 103, 105, 110],
            [168, 134, 96,  85,  85,  89,  97],
        ],  # fmt: skip
        delta=1,
    )


def test_radius_bigger_then_half() -> None:
    assert_blur(
        sample,
        3,
        [
            [144, 145, 142, 128, 114, 115, 117],
            [148, 145, 137, 122, 109, 111, 112],
            [152, 145, 131, 117, 103, 107, 108],
            [156, 144, 126, 111, 97,  102, 103],
            [160, 144, 121, 106, 92,  98,  99],
        ],  # fmt: skip
        delta=1,
    )


def test_radius_bigger_then_width() -> None:
    assert_blur(
        sample,
        10,
        [
            [158, 153, 147, 141, 135, 129, 123],
            [159, 153, 147, 141, 136, 130, 124],
            [159, 154, 148, 142, 136, 130, 124],
            [160, 154, 148, 142, 137, 131, 125],
            [160, 155, 149, 143, 137, 131, 125],
        ],
        delta=0,
    )


def test_extreme_large_radius() -> None:
    assert_blur(
        sample,
        600,
        [
            [162, 162, 162, 162, 162, 162, 162],
            [162, 162, 162, 162, 162, 162, 162],
            [162, 162, 162, 162, 162, 162, 162],
            [162, 162, 162, 162, 162, 162, 162],
            [162, 162, 162, 162, 162, 162, 162],
        ],
        delta=1,
    )


def test_two_passes() -> None:
    assert_blur(
        sample,
        1,
        [
            [153, 123, 102, 109, 132, 135, 129],
            [159, 138, 123, 121, 133, 131, 126],
            [162, 147, 136, 124, 127, 121, 121],
            [159, 140, 125, 108, 111, 106, 108],
            [154, 126, 105, 87,  94,  93,  97],
        ],  # fmt: skip
        passes=2,
        delta=1,
    )


def test_three_passes() -> None:
    assert_blur(
        sample,
        1,
        [
            [146, 131, 116, 118, 126, 131, 130],
            [151, 138, 125, 123, 126, 128, 127],
            [154, 143, 129, 123, 120, 120, 119],
            [152, 139, 122, 113, 108, 108, 108],
            [148, 132, 112, 102, 97,  99,  100],
        ],  # fmt: skip
        passes=3,
        delta=1,
    )


@pytest.mark.parametrize("mode", I16_MODES)
def test_i16_matches_l(mode: str) -> None:
    # I;16 source data in the 0-255 range must blur exactly like L data.
    for radius in (0, 0.02, 0.5, 1, 1.5, 3, 600):
        for passes in (1, 2, 3):
            reference = box_blur(sample, radius, passes)
            result = box_blur(sample.convert(mode), radius, passes)
            assert result.mode == mode
            assert _image_rows(result) == _image_rows(reference)


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize(
    "xradius, yradius, passes",
    (
        (1, 1, 1),
        (0, 0, 1),
        (0, 2, 1),
        (2, 0, 1),
        (0.5, 1.5, 1),
        (2, 5, 1),
        (1, 1, 2),
        (1, 1, 3),
        (3, 3, 2),
        (10, 10, 1),
        (600, 600, 1),
        (600, 1, 3),
    ),
)
def test_i16_reference(mode: str, xradius: float, yradius: float, passes: int) -> None:
    rows = [
        [0, 1, 255, 256, 1000],
        [32767, 32768, 65534, 65535, 0],
        [40000, 12345, 65535, 1, 999],
    ]
    im = _image_from_rows(rows, mode)
    result = im._new(im.im.box_blur((xradius, yradius), passes))
    assert _image_rows(result) == _box_blur_reference(rows, xradius, yradius, passes)


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize(
    "xradius, yradius, passes",
    ((1, 1, 3), (0, 2, 3), (2, 0, 3), (2, 5, 3), (0, 0, 3), (5, 1, 4)),
)
def test_i16_gaussian_reference(
    mode: str, xradius: float, yradius: float, passes: int
) -> None:
    rows = [
        [0, 1, 255, 256, 1000],
        [32767, 32768, 65534, 65535, 0],
        [40000, 12345, 65535, 1, 999],
    ]
    im = _image_from_rows(rows, mode)
    result = im._new(im.im.gaussian_blur((xradius, yradius), passes))
    assert _image_rows(result) == _gaussian_blur_reference(
        rows, xradius, yradius, passes
    )


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("value", (0, 65535))
def test_i16_extremes(mode: str, value: int) -> None:
    im = Image.new(mode, sample.size, value)
    for radius in (0, 1, 3, 600):
        for blur in (ImageFilter.BoxBlur(radius), ImageFilter.GaussianBlur(radius)):
            result = im.filter(blur)
            assert result.mode == mode
            assert result.tobytes() == im.tobytes()


@pytest.mark.parametrize("mode", I16_MODES)
def test_i16_byte_order(mode: str) -> None:
    # Values straddle byte boundaries, so a byte swap would change the result.
    values = [0, 1, 1000, 65534, 65535]
    im = _image_from_rows([values], mode)
    result = im._new(im.im.box_blur((1, 0), 1))
    expected = _box_blur_reference([values], 1, 0)[0]
    assert [result.getpixel((x, 0)) for x in range(5)] == expected
    assert result.tobytes() == b"".join(
        value.to_bytes(2, _byte_order(mode)) for value in expected
    )


def _byte_order(mode: str) -> Literal["big", "little"]:
    if mode == "I;16B":
        return "big"
    if mode == "I;16N":
        return sys.byteorder
    return "little"

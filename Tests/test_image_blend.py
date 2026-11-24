from __future__ import annotations

import ctypes
import sys

import pytest

from PIL import Image

I16_MODES = ("I;16", "I;16L", "I;16B", "I;16N")

# Two fixed grayscale images covering the low end, mid-range and the
# 16-bit boundaries (0 and 65535).
VALUES1 = (0, 1, 100, 1000, 32768, 60000, 65534, 65535)
VALUES2 = (65535, 0, 200, 2000, 32767, 5000, 1, 0)


def _f32(value: float) -> float:
    # Match the C implementation, which computes in 32-bit float.
    return ctypes.c_float(value).value


def _expected_pixel(value1: int, value2: int, alpha: float) -> int:
    alpha32 = _f32(alpha)
    temp = _f32(value1 + _f32(alpha32 * _f32(value2 - value1)))
    if alpha < 0.0 or alpha > 1.0:
        # Extrapolation: clip to the 16-bit output range.
        if temp <= 0.0:
            return 0
        if temp >= 65535.0:
            return 65535
    return int(temp)


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("alpha", (0.0, 0.5, 1.0, -1.0, 2.0, 10.0))
def test_blend_i16(mode: str, alpha: float) -> None:
    image1 = Image.new(mode, (len(VALUES1), 1))
    image1.putdata(VALUES1)
    image2 = Image.new(mode, (len(VALUES2), 1))
    image2.putdata(VALUES2)

    blended = Image.blend(image1, image2, alpha)

    # Mode, size and byte order are preserved.
    assert blended.mode == mode
    assert blended.size == image1.size

    for x, (value1, value2) in enumerate(zip(VALUES1, VALUES2)):
        assert blended.getpixel((x, 0)) == _expected_pixel(value1, value2, alpha)


@pytest.mark.parametrize("mode", I16_MODES)
def test_blend_i16_endpoints(mode: str) -> None:
    image1 = Image.new(mode, (len(VALUES1), 1))
    image1.putdata(VALUES1)
    image2 = Image.new(mode, (len(VALUES2), 1))
    image2.putdata(VALUES2)

    # alpha 0.0 and 1.0 return copies, byte for byte.
    assert Image.blend(image1, image2, 0.0).tobytes() == image1.tobytes()
    assert Image.blend(image1, image2, 1.0).tobytes() == image2.tobytes()


@pytest.mark.parametrize("mode", I16_MODES)
def test_blend_i16_byte_order(mode: str) -> None:
    image1 = Image.new(mode, (1, 1))
    image1.putpixel((0, 0), 0x1234)
    image2 = Image.new(mode, (1, 1))
    image2.putpixel((0, 0), 0x1234)

    raw = Image.blend(image1, image2, 0.5).tobytes()
    if mode in ("I;16", "I;16L"):
        assert raw == b"\x34\x12"
    elif mode == "I;16B":
        assert raw == b"\x12\x34"
    else:  # I;16N follows the native byte order
        assert raw == (b"\x34\x12" if sys.byteorder == "little" else b"\x12\x34")


@pytest.mark.parametrize("mode", I16_MODES)
def test_blend_i16_overflow(mode: str) -> None:
    # Extrapolation past the 16-bit range is clipped at both ends.
    black = Image.new(mode, (2, 1))
    black.putdata((0, 65535))
    white = Image.new(mode, (2, 1))
    white.putdata((65535, 0))

    blended = Image.blend(black, white, 2.0)
    assert tuple(blended.getpixel((x, 0)) for x in range(2)) == (65535, 0)

    blended = Image.blend(black, white, -1.0)
    assert tuple(blended.getpixel((x, 0)) for x in range(2)) == (0, 65535)


def test_blend_i16_mismatch() -> None:
    # Different sizes
    with pytest.raises(ValueError):
        Image.blend(Image.new("I;16", (2, 2)), Image.new("I;16", (3, 3)), 0.5)

    # Different I;16 modes (different byte order)
    with pytest.raises(ValueError):
        Image.blend(Image.new("I;16", (2, 2)), Image.new("I;16B", (2, 2)), 0.5)

    # Different pixel types
    with pytest.raises(ValueError):
        Image.blend(Image.new("L", (2, 2)), Image.new("I;16", (2, 2)), 0.5)


def test_blend_8bit_unchanged() -> None:
    image1 = Image.new("L", (3, 1))
    image1.putdata((0, 100, 255))
    image2 = Image.new("L", (3, 1))
    image2.putdata((255, 200, 0))

    def values(image: Image.Image) -> tuple[int, ...]:
        return tuple(image.getpixel((x, 0)) for x in range(3))

    assert values(Image.blend(image1, image2, 0.0)) == (0, 100, 255)
    assert values(Image.blend(image1, image2, 0.5)) == (127, 150, 127)
    assert values(Image.blend(image1, image2, 1.0)) == (255, 200, 0)
    # Extrapolation is clipped to [0, 255].
    assert values(Image.blend(image1, image2, 2.0)) == (255, 255, 0)
    assert values(Image.blend(image1, image2, -1.0)) == (0, 0, 255)


def test_blend_8bit_cross_mode_unchanged() -> None:
    # Two 8-bit modes with matching bands/size were blendable before; the
    # new I;16 mode check must not change the 8-bit behavior.
    image1 = Image.new("RGBA", (1, 1), (0, 0, 0, 255))
    image2 = Image.new("RGBa", (1, 1), (255, 255, 255, 255))

    blended = Image.blend(image1, image2, 0.5)
    assert blended.mode == "RGBA"
    assert blended.getpixel((0, 0)) == (127, 127, 127, 255)


def test_blend_float_unsupported() -> None:
    # F mode was never supported; keep rejecting it.
    with pytest.raises(ValueError):
        Image.blend(Image.new("F", (2, 1)), Image.new("F", (2, 1)), 0.5)

from __future__ import annotations

import pytest

from PIL import Image

from .helper import assert_image_equal, hopper


def test_sanity() -> None:
    im = hopper()

    with pytest.raises(ValueError):
        im.point(list(range(256)))
    im.point(list(range(256)) * 3)
    im.point(lambda x: x)
    im.point(lambda x: x * 1.2)

    im = im.convert("I")
    with pytest.raises(ValueError):
        im.point(list(range(256)))
    im.point(lambda x: x * 1)
    im.point(lambda x: x + 1)
    im.point(lambda x: x - 1)
    im.point(lambda x: x * 1 + 1)
    im.point(lambda x: 0.1 + 0.2 * x)
    im.point(lambda x: -x)
    im.point(lambda x: x - 0.5)
    im.point(lambda x: 1 - x / 2)
    im.point(lambda x: (2 + x) / 3)
    im.point(lambda x: 0.5)
    im.point(lambda x: x / 1)
    im.point(lambda x: x + x)
    with pytest.raises(TypeError):
        im.point(lambda x: x * x)
    with pytest.raises(TypeError):
        im.point(lambda x: x / x)
    with pytest.raises(TypeError):
        im.point(lambda x: 1 / x)
    with pytest.raises(TypeError):
        im.point(lambda x: x // 2)


@pytest.mark.parametrize("mode", ("LA", "RGBA", "CMYK"))
def test_multiband(mode: str) -> None:
    im = hopper(mode)
    single_band_lut = [(x + 1) % 256 for x in range(256)]
    lut = single_band_lut * Image.getmodebands(mode)
    out = im.point(lut)

    expected = Image.merge(
        mode, tuple(band.point(single_band_lut) for band in im.split())
    )
    assert_image_equal(out, expected)


@pytest.mark.parametrize("mode", ("LA", "PA", "RGB", "RGBA"))
def test_l_to_multiband(mode: str) -> None:
    # every input value, including the 0 and 255 extremes
    im = Image.frombytes("L", (16, 16), bytes(range(256)))

    bands = Image.getmodebands(mode)
    band_luts = [
        [(v * (b + 2) + 13 * b) % 256 for v in range(256)] for b in range(bands)
    ]
    lut = [v for band_lut in band_luts for v in band_lut]

    out = im.point(lut, mode)

    # reference: apply each 256-entry segment as a separate grayscale mapping
    expected = Image.merge(mode, [im.point(band_lut) for band_lut in band_luts])
    assert_image_equal(out, expected)


@pytest.mark.parametrize("mode", ("LA", "PA", "RGB", "RGBA"))
def test_l_to_multiband_identity(mode: str) -> None:
    im = hopper("L")
    lut = list(range(256)) * Image.getmodebands(mode)

    out = im.point(lut, mode)

    expected = Image.merge(mode, [im] * Image.getmodebands(mode))
    assert_image_equal(out, expected)


@pytest.mark.parametrize("mode", ("LA", "PA", "RGB", "RGBA"))
def test_l_to_multiband_wrong_length(mode: str) -> None:
    im = Image.new("L", (1, 1))
    lut = list(range(256)) * Image.getmodebands(mode)

    with pytest.raises(ValueError):
        im.point(lut[:-1], mode)
    with pytest.raises(ValueError):
        im.point(lut + [0], mode)


def test_l_to_multiband_callable() -> None:
    im = hopper("L")
    out = im.point(lambda x: 255 - x, "RGB")

    expected = Image.merge("RGB", [im.point(lambda x: 255 - x)] * 3)
    assert_image_equal(out, expected)


def test_p_to_pa_palette() -> None:
    im = hopper("P")
    out = im.point(list(range(256)) * 2, "PA")

    assert out.mode == "PA"
    assert out.getpalette() == im.getpalette()


def test_explicit_same_mode() -> None:
    # passing the current mode explicitly must not change the mapping
    im = hopper("RGB")
    lut = [(x + 1) % 256 for x in range(256)] * 3

    assert_image_equal(im.point(lut, "RGB"), im.point(lut))


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_16bit(mode: str) -> None:
    im = Image.new(mode, (1, 1), 0x1234)
    im.point(lambda x: x).getpixel((0, 0)) == 0x1234


def test_16bit_lut() -> None:
    """Tests for 16 bit -> 8 bit lut for converting I->L images
    see https://github.com/python-pillow/Pillow/issues/440
    """
    im = hopper("I")
    im.point(list(range(256)) * 256, "L")


def test_f_lut() -> None:
    """Tests for floating point lut of 8bit gray image"""
    im = hopper("L")
    lut = [0.5 * float(x) for x in range(256)]

    out = im.point(lut, "F")

    int_lut = [x // 2 for x in range(256)]
    assert_image_equal(out.convert("L"), im.point(int_lut, "L"))


def test_f_mode() -> None:
    im = hopper("F")
    with pytest.raises(ValueError):
        im.point([])


def test_overstated_length() -> None:
    # shouldn't segfault
    # see https://github.com/python-pillow/Pillow/issues/9892

    class OverstatedLengthSequence:
        def __len__(self) -> int:
            return 256

        def __getitem__(self, index: int) -> int:
            if index >= 8:
                raise IndexError
            return index

    im = Image.new("L", (4, 4))
    with pytest.raises(ValueError):
        im.point(OverstatedLengthSequence(), "F")  # type: ignore[arg-type]


def test_raising_length() -> None:
    class RaisingLengthSequence:
        def __len__(self) -> int:
            raise RuntimeError

        def __getitem__(self, index: int) -> float:
            return index

    im = Image.new("L", (1, 1))
    with pytest.raises(RuntimeError):
        im.point(RaisingLengthSequence(), "F")  # type: ignore[arg-type]

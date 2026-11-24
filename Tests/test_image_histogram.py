from __future__ import annotations

import pytest

from PIL import Image

from .helper import hopper


def test_histogram() -> None:
    def histogram(mode: str) -> tuple[int, int, int]:
        h = hopper(mode).histogram()
        return len(h), min(h), max(h)

    assert histogram("1") == (256, 0, 10994)
    assert histogram("L") == (256, 0, 662)
    assert histogram("LA") == (512, 0, 16384)
    assert histogram("La") == (512, 0, 16384)
    for mode in ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
        assert histogram(mode) == (256, 0, 662)
    assert histogram("P") == (256, 0, 1551)
    assert histogram("PA") == (512, 0, 16384)
    assert histogram("RGB") == (768, 4, 675)
    assert histogram("RGBA") == (1024, 0, 16384)
    assert histogram("CMYK") == (1024, 0, 16384)
    assert histogram("YCbCr") == (768, 0, 1908)


def test_histogram_masked() -> None:
    mask = Image.new("1", (128, 128), 0)
    crop_box = (0, 0, 128, 64)
    mask.paste(1, crop_box)

    def histogram(mode: str) -> tuple[int, int, int]:
        im = hopper(mode)
        h = im.histogram(mask)
        assert h == im.crop(crop_box).histogram()
        return len(h), min(h), max(h)

    assert histogram("1") == (256, 0, 5024)
    assert histogram("L") == (256, 0, 178)
    assert histogram("LA") == (512, 0, 8192)
    assert histogram("La") == (512, 0, 8192)
    assert histogram("P") == (256, 0, 797)
    assert histogram("PA") == (512, 0, 8192)
    assert histogram("RGB") == (768, 1, 174)
    assert histogram("RGBA") == (1024, 0, 8192)
    assert histogram("CMYK") == (1024, 0, 8192)
    assert histogram("YCbCr") == (768, 0, 1009)


@pytest.mark.parametrize("mode", ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"))
def test_histogram_masked_unsupported_mode(mode: str) -> None:
    mask = Image.new("1", (128, 128))
    with pytest.raises(ValueError, match="image has wrong mode"):
        hopper(mode).histogram(mask)


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_histogram_i16(mode: str) -> None:
    im = Image.new(mode, (2, 1))
    im.putpixel((1, 0), 300)
    h = im.histogram()
    assert h[0] == h[255] == 1
    assert sum(h) == 2

    h = im.histogram(extrema=(0, 510))
    assert h[0] == h[150] == 1
    assert sum(h) == 2


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_histogram_i16_fixed_values(mode: str) -> None:
    # Fixed pixels covering the full 16-bit range and an 8-bit boundary
    values = (0, 255, 256, 65535)
    base = Image.new("I", (len(values), 1))
    base.putdata(values)
    im = base.convert(mode)

    assert list(im.getdata()) == list(values)
    assert im.getextrema() == (0, 65535)

    h = im.histogram()
    assert len(h) == 256
    # 0, 256 and 65535 land in the first/last edge bins when scaling the
    # full 0..65535 range onto 256 bins; 255 maps to the last bin
    assert h[0] == 3
    assert h[255] == 1
    assert sum(h) == len(values)
    assert all(count == 0 for index, count in enumerate(h) if index not in (0, 255))


@pytest.mark.parametrize("mode", ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"))
def test_histogram_single_value(mode: str) -> None:
    im = Image.new(mode, (2, 2), 3)
    assert im.histogram() == [4] + [0] * 255
    assert im.histogram(extrema=(3, 3)) == [4] + [0] * 255
    assert im.histogram(extrema=(2, 2)) == [0] * 256
    assert im.histogram(extrema=(4, 2)) == [0] * 256


def test_histogram_zero_size() -> None:
    assert Image.new("L", (0, 0)).histogram() == [0] * 256
    for mode in ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
        with pytest.raises(ValueError, match="min/max not given"):
            Image.new(mode, (0, 0)).histogram()


def test_histogram_mask_size_mismatch() -> None:
    mask = Image.new("1", (1, 1))
    with pytest.raises(ValueError, match="images do not match"):
        hopper("L").histogram(mask)


@pytest.mark.parametrize(
    "mode", ("LA", "La", "I", "F", "P", "PA", "RGB", "RGBA", "CMYK", "YCbCr")
)
def test_histogram_bad_mask_mode(mode: str) -> None:
    mask = Image.new(mode, (128, 128))
    with pytest.raises(ValueError, match="bad transparency mask"):
        hopper("L").histogram(mask)

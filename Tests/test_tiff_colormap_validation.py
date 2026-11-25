"""Tests for TIFF ColorMap length validation against BitsPerSample.

A ColorMap stores three planes (R, G, B), each containing
``1 << BitsPerSample`` SHORT values, so the flat tag must contain exactly
``3 * (1 << BitsPerSample)`` items. Both the pure-Python and the libtiff
write paths enforce this before writing the tag.
"""

from __future__ import annotations

import io

import pytest

from PIL import Image, TiffImagePlugin, features

COLORMAP = TiffImagePlugin.COLORMAP

# Length validation happens in Python before the libtiff encoder is created,
# so error cases run even when Pillow was built without libtiff. The
# successful libtiff write/reopen cases need the native encoder.
LIBTIFF_WRITES = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not features.check("libtiff"), reason="libtiff support required"
        ),
    ),
]


def _image_for_bits(bits: int) -> Image.Image:
    return Image.new("I;16" if bits == 16 else "L", (2, 2))


@pytest.mark.parametrize("libtiff", [False, True])
@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("delta", [-1, 1])
def test_colormap_off_by_one_is_rejected(
    monkeypatch: pytest.MonkeyPatch, libtiff: bool, bits: int, delta: int
) -> None:
    monkeypatch.setattr(TiffImagePlugin, "WRITE_LIBTIFF", libtiff)
    expected = 3 * (1 << bits)

    im = _image_for_bits(bits)
    with pytest.raises(ValueError, match=f"Requiring {expected} items for Colormap"):
        im.save(
            io.BytesIO(),
            format="TIFF",
            tiffinfo={COLORMAP: (0,) * (expected + delta)},
        )


@pytest.mark.parametrize("libtiff", [False, True])
@pytest.mark.parametrize("bits", [8, 16])
def test_colormap_bad_length_is_rejected(
    monkeypatch: pytest.MonkeyPatch, libtiff: bool, bits: int
) -> None:
    monkeypatch.setattr(TiffImagePlugin, "WRITE_LIBTIFF", libtiff)
    expected = 3 * (1 << bits)

    im = _image_for_bits(bits)
    with pytest.raises(ValueError, match=f"Requiring {expected} items for Colormap"):
        im.save(
            io.BytesIO(),
            format="TIFF",
            tiffinfo={COLORMAP: (0,)},
        )


@pytest.mark.parametrize("libtiff", LIBTIFF_WRITES)
@pytest.mark.parametrize("bits", [8, 16])
def test_colormap_correct_length_roundtrips_count_and_planes(
    monkeypatch: pytest.MonkeyPatch, libtiff: bool, bits: int
) -> None:
    monkeypatch.setattr(TiffImagePlugin, "WRITE_LIBTIFF", libtiff)
    plane = 1 << bits
    expected = plane * 3

    # Sentinels at both ends of each plane catch a wrong plane stride/order.
    colormap = [0] * expected
    sentinels = {
        0: 11,
        plane - 1: 22,
        plane: 33,
        2 * plane - 1: 44,
        2 * plane: 55,
        expected - 1: 65535 if bits == 16 else 66,
    }
    for index, value in sentinels.items():
        colormap[index] = value

    out = io.BytesIO()
    _image_for_bits(bits).save(out, format="TIFF", tiffinfo={COLORMAP: tuple(colormap)})

    with Image.open(out) as reloaded:
        assert isinstance(reloaded, TiffImagePlugin.TiffImageFile)
        tag = reloaded.tag_v2[COLORMAP]
        assert len(tag) == expected
        assert list(tag) == colormap


@pytest.mark.parametrize("libtiff", LIBTIFF_WRITES)
def test_palette_image_colors_roundtrip(
    monkeypatch: pytest.MonkeyPatch, libtiff: bool
) -> None:
    monkeypatch.setattr(TiffImagePlugin, "WRITE_LIBTIFF", libtiff)

    palette: list[int] = []
    for i in range(256):
        palette += [(i * 7) % 256, (i * 13) % 256, (255 - i * 3) % 256]

    im = Image.new("P", (16, 16))
    im.putpalette(palette)
    px = im.load()
    assert px is not None
    for y in range(16):
        for x in range(16):
            px[x, y] = (x * 16 + y) % 256

    out = io.BytesIO()
    im.save(out, format="TIFF")

    with Image.open(out) as reloaded:
        assert isinstance(reloaded, TiffImagePlugin.TiffImageFile)
        assert len(reloaded.tag_v2[COLORMAP]) == 768
        rgb = reloaded.convert("RGB")
        rp = rgb.load()
        assert rp is not None
        for y in range(16):
            for x in range(16):
                index = (x * 16 + y) % 256
                expected_rgb = tuple(palette[3 * index : 3 * index + 3])
                assert rp[x, y][:3] == expected_rgb


@pytest.mark.parametrize("libtiff", LIBTIFF_WRITES)
def test_big_endian_16bit_colormap_roundtrip(
    monkeypatch: pytest.MonkeyPatch, libtiff: bool
) -> None:
    monkeypatch.setattr(TiffImagePlugin, "WRITE_LIBTIFF", libtiff)
    plane = 1 << 16
    colormap = [0] * (3 * plane)
    colormap[0] = 101
    colormap[plane - 1] = 202
    colormap[plane] = 303
    colormap[2 * plane - 1] = 404
    colormap[2 * plane] = 505
    colormap[-1] = 606

    out = io.BytesIO()
    Image.new("I;16B", (1, 1)).save(
        out, format="TIFF", tiffinfo={COLORMAP: tuple(colormap)}
    )

    # The pure-Python writer honors the MM prefix; libtiff writes native
    # byte order. Either way the tag values must round-trip unchanged.
    if not libtiff:
        assert out.getvalue()[:2] == b"MM"

    with Image.open(out) as reloaded:
        tag = reloaded.tag_v2[COLORMAP]
        assert len(tag) == 3 * plane
        assert list(tag) == colormap


@pytest.mark.parametrize("libtiff", LIBTIFF_WRITES)
@pytest.mark.parametrize("mode", ["L", "I;16"])
def test_grayscale_without_colormap_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, libtiff: bool, mode: str
) -> None:
    monkeypatch.setattr(TiffImagePlugin, "WRITE_LIBTIFF", libtiff)

    im = Image.new(mode, (8, 4), 128 if mode == "L" else 12345)
    out = io.BytesIO()
    im.save(out, format="TIFF")

    with Image.open(out) as reloaded:
        assert COLORMAP not in reloaded.tag_v2
        assert reloaded.size == im.size
        for y in range(im.height):
            for x in range(im.width):
                assert reloaded.getpixel((x, y)) == im.getpixel((x, y))

"""Tests for writing nested EXIF/TIFF sub-IFDs without libtiff.

These tests save to in-memory TIFF streams and reopen them, checking the
nested dictionaries, image pixels and ordinary tags individually.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from PIL import Image, TiffImagePlugin

from .helper import assert_image_equal

EXIF_IFD = 34665
GPS_IFD = 34853
INTEROP_IFD = 40965


def _save_reopen(
    im: Image.Image, metadata: Any, *, via: str = "tiffinfo", big_tiff: bool = False
) -> Image.Image:
    stream = io.BytesIO()
    kwargs: dict[str, Any] = {"big_tiff": big_tiff}
    kwargs[via] = metadata
    im.save(stream, format="TIFF", **kwargs)
    stream.seek(0)
    reloaded = Image.open(stream)
    reloaded.load()
    # Keep the backing stream alive for the lifetime of the checks.
    reloaded._test_stream = stream  # type: ignore[attr-defined]
    return reloaded


@pytest.mark.parametrize("big_tiff", [False, True])
@pytest.mark.parametrize(
    "mode, prefix",
    [("L", b"II"), ("I;16B", b"MM")],
)
def test_nested_dictionaries_roundtrip(
    mode: str, prefix: bytes, big_tiff: bool
) -> None:
    im = Image.new(mode, (13, 11))
    exif_ifd = {
        270: "e" * 120,  # long ASCII, forces out-of-line data
        36864: b"0231",
        315: "a" * 57,  # odd length payload
        INTEROP_IFD: {1: "R98", 2: b"0100", 270: "i" * 120},
    }
    gps_ifd = {1: "N", 3: "E", 18: "g" * 120}
    info = {
        EXIF_IFD: exif_ifd,
        GPS_IFD: gps_ifd,
        305: "s" * 90,
        270: "d" * 160,
    }

    reloaded = _save_reopen(im, info, big_tiff=big_tiff)

    assert reloaded.tag_v2.prefix == prefix
    assert reloaded.tag_v2._bigtiff is big_tiff
    assert reloaded.tobytes() == im.tobytes()

    # Ordinary metadata coexisting with the sub-IFDs.
    assert reloaded.tag_v2[305] == info[305]
    assert reloaded.tag_v2[270] == info[270]

    exif = reloaded.getexif()
    read_exif = exif.get_ifd(EXIF_IFD)
    for tag, value in exif_ifd.items():
        if tag == INTEROP_IFD:
            # The pointer tag stays an offset in the Exif IFD.
            assert isinstance(read_exif[tag], int)
        else:
            assert read_exif[tag] == value

    read_gps = exif.get_ifd(GPS_IFD)
    for tag, value in gps_ifd.items():
        assert read_gps[tag] == value

    read_interop = exif.get_ifd(INTEROP_IFD)
    for tag, value in exif_ifd[INTEROP_IFD].items():
        assert read_interop[tag] == value


@pytest.mark.parametrize("big_tiff", [False, True])
def test_multiple_ifds_and_long_tags_do_not_overlap(big_tiff: bool) -> None:
    im = Image.new("L", (9, 9))
    info = {
        270: "D" * 500,
        305: "S" * 700,
        700: b"X" * 1000,
        EXIF_IFD: {270: "E" * 900, INTEROP_IFD: {270: "I" * 400}},
        GPS_IFD: {18: "G" * 800},
        50001: "Z" * 1200,
    }

    reloaded = _save_reopen(im, info, big_tiff=big_tiff)

    assert reloaded.tobytes() == im.tobytes()
    assert reloaded.tag_v2[270] == "D" * 500
    assert reloaded.tag_v2[305] == "S" * 700
    assert reloaded.tag_v2[700] == b"X" * 1000
    assert reloaded.tag_v2[50001] == "Z" * 1200

    exif = reloaded.getexif()
    assert exif.get_ifd(EXIF_IFD)[270] == "E" * 900
    assert exif.get_ifd(INTEROP_IFD)[270] == "I" * 400
    assert exif.get_ifd(GPS_IFD)[18] == "G" * 800


def test_exif_keyword_accepts_nested_dictionary() -> None:
    im = Image.new("L", (4, 4))
    exif = {
        EXIF_IFD: {36864: b"0231", INTEROP_IFD: {1: "R98"}},
        GPS_IFD: {1: "N"},
    }

    reloaded = _save_reopen(im, exif, via="exif")

    ex = reloaded.getexif()
    assert ex.get_ifd(EXIF_IFD)[36864] == b"0231"
    assert ex.get_ifd(INTEROP_IFD)[1] == "R98"
    assert ex.get_ifd(GPS_IFD)[1] == "N"


def test_exif_object_get_ifd_is_written() -> None:
    # Exif IFDs lazily created with get_ifd() must be serialized even though
    # no top-level pointer key was assigned explicitly.
    im = Image.new("L", (1, 1))
    exif = im.getexif()
    exif[271] = "Canon"
    gps = exif.get_ifd(GPS_IFD)
    gps[18] = "WGS-84"
    exif_ifd = exif.get_ifd(EXIF_IFD)
    exif_ifd[36864] = b"0231"
    exif_ifd[INTEROP_IFD] = {1: "R98", 2: b"0100"}

    reloaded = _save_reopen(im, exif, via="exif")

    ex = reloaded.getexif()
    assert ex[271] == "Canon"
    assert ex.get_ifd(GPS_IFD) == {18: "WGS-84"}
    assert ex.get_ifd(EXIF_IFD)[36864] == b"0231"
    assert ex.get_ifd(INTEROP_IFD) == {1: "R98", 2: b"0100"}


def test_exif_bytes_with_nested_ifd() -> None:
    source = Image.Exif()
    source.get_ifd(EXIF_IFD)[36864] = b"0231"
    source.get_ifd(EXIF_IFD)[INTEROP_IFD] = {1: "R98"}

    im = Image.new("L", (2, 2))
    reloaded = _save_reopen(im, source.tobytes(), via="exif")

    ex = reloaded.getexif()
    assert ex.get_ifd(EXIF_IFD)[36864] == b"0231"
    assert ex.get_ifd(INTEROP_IFD)[1] == "R98"


def test_unknown_tags_remain_supported() -> None:
    im = Image.new("L", (4, 4))
    info = {59999: 123456, 60000: b"\x01\x02\x03", 60001: "hello"}

    reloaded = _save_reopen(im, info)

    assert reloaded.tag_v2[59999] == 123456
    assert reloaded.tag_v2[60000] == b"\x01\x02\x03"
    assert reloaded.tag_v2[60001] == "hello"


def test_unknown_tag_dictionary_is_written_as_ifd() -> None:
    im = Image.new("L", (4, 4))
    stream = io.BytesIO()
    im.save(stream, format="TIFF", tiffinfo={59999: {60001: "nested-value"}})
    raw = stream.getvalue()
    stream.seek(0)
    reloaded = Image.open(stream)
    reloaded.load()

    pointer = reloaded.tag_v2[59999]
    assert isinstance(pointer, int)

    header = raw[:8]
    if 43 in header[2:4]:
        header += raw[8:16]
    sub_ifd = TiffImagePlugin.ImageFileDirectory_v2(header, group=59999)
    sub_stream = io.BytesIO(raw)
    sub_stream.seek(pointer)
    sub_ifd.load(sub_stream)
    assert sub_ifd[60001] == "nested-value"


@pytest.mark.parametrize(
    "bad_value",
    [
        {EXIF_IFD: "not-a-dictionary"},
        {EXIF_IFD: None},
        {EXIF_IFD: b"1234"},
        {GPS_IFD: ["N"]},
        {EXIF_IFD: {INTEROP_IFD: "not-a-dictionary"}},
        {EXIF_IFD: {"string-key": 1}},
        {EXIF_IFD: {70000: 1}},
    ],
)
def test_invalid_nested_values_fail(bad_value: dict[int, Any]) -> None:
    im = Image.new("L", (4, 4))
    stream = io.BytesIO()
    with pytest.raises((TypeError, ValueError)):
        im.save(stream, format="TIFF", tiffinfo=bad_value)


def test_tiff_without_subifd_still_saves(tmp_path: Any) -> None:
    im = Image.new("RGB", (8, 8), (10, 20, 30))
    out = tmp_path / "plain.tif"
    im.save(out, format="TIFF")
    with Image.open(out) as reloaded:
        assert_image_equal(reloaded, im)

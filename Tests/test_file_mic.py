from __future__ import annotations

from io import BytesIO

import olefile
import pytest

from PIL import Image, ImagePalette
from PIL._binary import o32le as o32

from .helper import (
    assert_image_equal,
    assert_image_similar,
    hopper,
    skip_unless_feature,
)

MicImagePlugin = pytest.importorskip(
    "PIL.MicImagePlugin", reason="olefile not installed"
)
pytestmark = skip_unless_feature("libtiff")
TEST_FILE = "Tests/images/hopper.mic"


def test_sanity() -> None:
    with Image.open(TEST_FILE) as im:
        im.load()
        assert im.mode == "RGBA"
        assert im.size == (128, 128)
        assert im.format == "MIC"

        # Adjust for the gamma of 2.2 encoded into the file
        lut = ImagePalette.make_gamma_lut(1 / 2.2)
        im1 = Image.merge("RGBA", [chan.point(lut) for chan in im.split()])

        im2 = hopper("RGBA")
        assert_image_similar(im1, im2, 10)


def test_n_frames() -> None:
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, MicImagePlugin.MicImageFile)
        assert im.n_frames == 1


def test_is_animated() -> None:
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, MicImagePlugin.MicImageFile)
        assert not im.is_animated


def test_tell() -> None:
    with Image.open(TEST_FILE) as im:
        assert im.tell() == 0


def test_seek() -> None:
    with Image.open(TEST_FILE) as im:
        im.seek(0)
        assert im.tell() == 0

        with pytest.raises(EOFError):
            im.seek(99)
        assert im.tell() == 0


def _make_two_frame_mic(im: Image.Image) -> None:
    assert isinstance(im, MicImagePlugin.MicImageFile)
    im._n_frames = 2
    im.images = [im.images[0], im.images[0]]


def test_seek_multiple_frames() -> None:
    with Image.open(TEST_FILE) as im, Image.open(TEST_FILE) as expected:
        _make_two_frame_mic(im)
        assert im.n_frames == 2

        im.seek(1)
        assert im.tell() == 1
        assert im.size == (128, 128)
        assert_image_equal(im, expected)

        # Repeated seeks remain consistent
        im.seek(0)
        assert im.tell() == 0
        assert_image_equal(im, expected)
        im.seek(1)
        assert im.tell() == 1
        assert_image_equal(im, expected)


def test_seek_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    # The first frame is 128x128; enlarge the second frame beyond twice the budget
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 128 * 128)
    with Image.open(TEST_FILE) as im:
        _make_two_frame_mic(im)

        assert isinstance(im, MicImagePlugin.MicImageFile)
        with im.ole.openstream(im.images[0]) as fp:
            tiff_data = fp.read()
        tiff_data = tiff_data[:35132] + o32(256) + tiff_data[35136:]
        tiff_data = tiff_data[:35144] + o32(256) + tiff_data[35148:]
        with open(TEST_FILE, "rb") as fp:
            b = BytesIO(fp.read())
        im.ole = olefile.OleFileIO(b, write_mode=True)
        im.ole.write_stream(im.images[0], tiff_data)

        with pytest.raises(Image.DecompressionBombError):
            im.seek(1)
        # A failed seek must not update the frame state or allocate a buffer
        assert im.tell() == 0
        assert im._im is None


def test_close() -> None:
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, MicImagePlugin.MicImageFile)
    assert im.ole.fp.closed

    im = Image.open(TEST_FILE)
    assert isinstance(im, MicImagePlugin.MicImageFile)
    im.close()
    assert im.ole.fp.closed


def test_invalid_file() -> None:
    # Test an invalid OLE file
    invalid_file = "Tests/images/flower.jpg"
    with pytest.raises(SyntaxError):
        MicImagePlugin.MicImageFile(invalid_file)

    # Test a valid OLE file, but not a MIC file
    ole_file = "Tests/images/test-ole-file.doc"
    with pytest.raises(SyntaxError):
        MicImagePlugin.MicImageFile(ole_file)

from __future__ import annotations

import warnings
from io import BytesIO

import pytest

from PIL import DcxImagePlugin, Image
from PIL._binary import o16le as o16
from PIL._binary import o32le as o32

from .helper import assert_image_equal, hopper, is_pypy

# Created with ImageMagick: convert hopper.ppm hopper.dcx
TEST_FILE = "Tests/images/hopper.dcx"


def test_sanity() -> None:
    # Arrange

    # Act
    with Image.open(TEST_FILE) as im:
        # Assert
        assert im.size == (128, 128)
        assert isinstance(im, DcxImagePlugin.DcxImageFile)
        orig = hopper()
        assert_image_equal(im, orig)


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
def test_unclosed_file() -> None:
    def open_test_image() -> None:
        im = Image.open(TEST_FILE)
        im.load()

    with pytest.warns(ResourceWarning):
        open_test_image()


def test_closed_file() -> None:
    with warnings.catch_warnings(action="error"):
        im = Image.open(TEST_FILE)
        im.load()
        im.close()


def test_context_manager() -> None:
    with warnings.catch_warnings(action="error"):
        with Image.open(TEST_FILE) as im:
            im.load()


def test_invalid_file() -> None:
    with open("Tests/images/flower.jpg", "rb") as fp:
        with pytest.raises(SyntaxError):
            DcxImagePlugin.DcxImageFile(fp)


def test_tell() -> None:
    # Arrange
    with Image.open(TEST_FILE) as im:
        # Act
        frame = im.tell()

        # Assert
        assert frame == 0


def test_n_frames() -> None:
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, DcxImagePlugin.DcxImageFile)
        assert im.n_frames == 1
        assert not im.is_animated


def test_eoferror() -> None:
    with Image.open(TEST_FILE) as im:
        assert isinstance(im, DcxImagePlugin.DcxImageFile)
        n_frames = im.n_frames

        # Test seeking past the last frame
        with pytest.raises(EOFError):
            im.seek(n_frames)
        assert im.tell() < n_frames

        # Test that seeking to the last frame does not raise an error
        im.seek(n_frames - 1)


def test_seek_too_far() -> None:
    # Arrange
    with Image.open(TEST_FILE) as im:
        frame = 999  # too big on purpose

    # Act / Assert
    with pytest.raises(EOFError):
        im.seek(frame)


def _make_dcx(frames: list[bytes]) -> BytesIO:
    header = o32(DcxImagePlugin.MAGIC)
    offset = 4 + 4 * (len(frames) + 1)
    for frame in frames:
        header += o32(offset)
        offset += len(frame)
    return BytesIO(header + o32(0) + b"".join(frames))


def test_seek_multiple_frames() -> None:
    with open("Tests/images/pil184.pcx", "rb") as fp:
        frame = fp.read()
    with (
        Image.open("Tests/images/pil184.pcx") as expected,
        Image.open(_make_dcx([frame, frame])) as im,
    ):
        assert im.n_frames == 2
        assert im.is_animated

        assert im.tell() == 0
        assert_image_equal(im, expected)

        im.seek(1)
        assert im.tell() == 1
        assert_image_equal(im, expected)

        # Repeated seeks remain consistent
        im.seek(0)
        assert im.tell() == 0
        assert_image_equal(im, expected)
        im.seek(1)
        assert im.tell() == 1
        assert_image_equal(im, expected)


def test_seek_decompression_bomb_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    with open("Tests/images/pil184.pcx", "rb") as fp:
        first_frame = fp.read()
    # The first frame is 447x144; make the second frame exceed the budget
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 447 * 144)
    second_frame = first_frame[:8] + o16(300) + o16(300) + first_frame[12:]
    with Image.open(_make_dcx([first_frame, second_frame])) as im:
        with pytest.warns(Image.DecompressionBombWarning):
            im.seek(1)
        assert im.tell() == 1


def test_seek_decompression_bomb_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with open("Tests/images/pil184.pcx", "rb") as fp:
        first_frame = fp.read()
    # The first frame is 447x144; make the second frame exceed twice the budget
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 447 * 144)
    second_frame = first_frame[:8] + o16(400) + o16(400) + first_frame[12:]
    with Image.open(_make_dcx([first_frame, second_frame])) as im:
        with pytest.raises(Image.DecompressionBombError):
            im.seek(1)
        # A failed seek must not update the frame state or allocate a buffer
        assert im.tell() == 0
        assert im._im is None

        # Seeking again fails the same way, leaving the frame state untouched
        with pytest.raises(Image.DecompressionBombError):
            im.seek(1)
        assert im.tell() == 0
        assert im._im is None

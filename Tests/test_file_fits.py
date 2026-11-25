from __future__ import annotations

import gzip
import struct
from io import BytesIO

import pytest

from PIL import FitsImagePlugin, Image

from .helper import assert_image_equal, assert_image_equal_tofile, hopper

TEST_FILE = "Tests/images/hopper.fits"


def fits_header(number_of_bits: int, size: tuple[int, int], naxis: int = 2) -> bytes:
    cards = [
        b"SIMPLE  = T",
        f"BITPIX  = {number_of_bits}".encode(),
        f"NAXIS   = {naxis}".encode(),
    ]
    if naxis >= 1:
        cards.append(f"NAXIS1  = {size[0]}".encode())
    if naxis >= 2:
        cards.append(f"NAXIS2  = {size[1]}".encode())
    cards.append(b"END")
    header = b"".join(card.ljust(80, b" ") for card in cards)
    return header.ljust((len(header) + 2879) // 2880 * 2880, b" ")


def fits_data(
    number_of_bits: int, size: tuple[int, int], data: bytes, naxis: int = 2
) -> BytesIO:
    return BytesIO(fits_header(number_of_bits, size, naxis) + data)


def test_open() -> None:
    # Act
    with Image.open(TEST_FILE) as im:
        # Assert
        assert im.format == "FITS"
        assert im.size == (128, 128)
        assert im.mode == "L"

        assert_image_equal(im, hopper("L"))


def test_gzip1() -> None:
    with Image.open("Tests/images/m13_gzip.fits") as im:
        assert im.getpixel((0, 0)) == 111

        assert_image_equal_tofile(im, "Tests/images/m13.fits")


def gzip_fits(number_of_bits: int, values: list[float]) -> BytesIO:
    # A gzip-compressed FITS image stores each pixel as a big-endian
    # four-byte group, right-justified within the group.
    if number_of_bits == 8:
        payload = b"".join(b"\0\0\0" + bytes([int(v)]) for v in values)
    elif number_of_bits == -32:
        payload = b"".join(struct.pack(">f", v) for v in values)
    else:
        payload = b"".join(struct.pack(">I", int(v) & 0xFFFFFFFF) for v in values)

    primary = b"".join(
        card.ljust(80, b" ")
        for card in [b"SIMPLE  = T", b"BITPIX  = 8", b"NAXIS   = 0", b"END"]
    ).ljust(2880, b" ")
    cards = [
        b"XTENSION= 'BINTABLE'",
        b"BITPIX  = 8",
        b"NAXIS   = 2",
        b"NAXIS1  = 8",
        b"NAXIS2  = 4",
        b"PCOUNT  = 100",
        b"GCOUNT  = 1",
        b"TFIELDS = 1",
        b"ZIMAGE  = T",
        b"ZCMPTYPE= 'GZIP_1  '",
        f"ZBITPIX = {number_of_bits}".encode(),
        b"ZNAXIS  = 2",
        b"ZNAXIS1 = 2",
        b"ZNAXIS2 = 2",
        b"END",
    ]
    extension = b"".join(card.ljust(80, b" ") for card in cards).ljust(2880, b" ")
    # The compressed data follows a binary table. FITS pads every data unit
    # to a multiple of 2880 bytes, so the file can be parsed to its end.
    table = b"\0" * 32 + gzip.compress(payload)
    padded = table.ljust((len(table) + 2879) // 2880 * 2880, b"\0")
    return BytesIO(primary + extension + padded)


def test_gzip_big_endian() -> None:
    with Image.open(gzip_fits(16, [0x0102, 0x1234, 0xFFFE, 0x8000])) as im:
        assert im.mode == "I;16"
        assert [im.getpixel((x, y)) for y in (1, 0) for x in (0, 1)] == [
            0x0102,
            0x1234,
            0xFFFE,
            0x8000,
        ]


def test_gzip_float() -> None:
    with Image.open(gzip_fits(-32, [1.5, -2.25, 3.0, 4.75])) as im:
        assert im.mode == "F"
        assert [im.getpixel((x, y)) for y in (1, 0) for x in (0, 1)] == pytest.approx(
            [1.5, -2.25, 3.0, 4.75]
        )


def test_gzip_decode() -> None:
    with open("Tests/images/m13_gzip.fits", "rb") as fp:
        data = bytearray(fp.read())

    offset = 11040
    corrupted = bytearray(data)
    corrupted[offset + 10] = 0
    with Image.open(BytesIO(corrupted)) as im:
        with pytest.raises(OSError, match="Corrupt FITS gzip data"):
            im.load()

    with Image.open(BytesIO(bytes(data[: offset + 2]))) as im:
        with pytest.raises(OSError, match="Truncated FITS gzip data"):
            im.load()


def test_big_endian() -> None:
    # The high and low bytes of each value differ, so reading them in the
    # wrong order would produce visibly different pixels. FITS rows are
    # presented bottom-up, as for 8-bit data.
    data = struct.pack(">4H", 0x0102, 0x1234, 0xFFFE, 0x8000)
    with Image.open(fits_data(16, (2, 2), data)) as im:
        assert im.mode == "I;16"
        assert im.size == (2, 2)
        assert [im.getpixel((x, y)) for y in range(2) for x in range(2)] == [
            0xFFFE,
            0x8000,
            0x0102,
            0x1234,
        ]


def test_signed_32_bit() -> None:
    data = struct.pack(">4i", 0x01020304, 0x7F000001, -1, -0x80000000)
    with Image.open(fits_data(32, (2, 2), data)) as im:
        assert im.mode == "I"
        assert [im.getpixel((x, y)) for y in range(2) for x in range(2)] == [
            -1,
            -0x80000000,
            0x01020304,
            0x7F000001,
        ]


@pytest.mark.parametrize("number_of_bits", (-32, -64))
def test_float(number_of_bits: int) -> None:
    values = [1.5, -2.25, 3.0, 4.75]
    code = "f" if number_of_bits == -32 else "d"
    data = struct.pack(f">{len(values)}{code}", *values)
    with Image.open(fits_data(number_of_bits, (2, 2), data)) as im:
        assert im.mode == "F"
        assert [
            im.getpixel((x, y)) for y in range(2) for x in range(2)
        ] == pytest.approx([3.0, 4.75, 1.5, -2.25])


@pytest.mark.parametrize("number_of_bits", (24, 128, 0, -16))
def test_unsupported_number_of_bits(number_of_bits: int) -> None:
    with pytest.raises(OSError, match="Unsupported number of bits"):
        Image.open(fits_data(number_of_bits, (2, 2), b"\0" * 16))


def test_truncated_data() -> None:
    # A 16-bit image whose data unit stops partway through should not
    # silently decode the header padding as pixels.
    full = fits_header(16, (4, 4)) + struct.pack(">16H", *range(16))
    with Image.open(BytesIO(full[: 2880 + 16])) as im:
        with pytest.raises(OSError):
            im.load()


def test_invalid_file() -> None:
    # Arrange
    invalid_file = "Tests/images/flower.jpg"

    # Act / Assert
    with pytest.raises(SyntaxError):
        FitsImagePlugin.FitsImageFile(invalid_file)


def test_truncated_fits() -> None:
    # No END to headers
    image_data = b"SIMPLE  =                    T" + b" " * 50 + b"TRUNCATE"
    with pytest.raises(OSError):
        FitsImagePlugin.FitsImageFile(BytesIO(image_data))


def test_naxis_zero() -> None:
    # This test image has been manually hexedited
    # to set the number of data axes to zero
    with pytest.raises(ValueError):
        with Image.open("Tests/images/hopper_naxis_zero.fits"):
            pass


def test_comment() -> None:
    image_data = b"SIMPLE  =                    T / comment string"
    with pytest.raises(OSError):
        FitsImagePlugin.FitsImageFile(BytesIO(image_data))

from __future__ import annotations

import pytest

from PIL import Image


def pack_bit_planes(
    width: int, planes: int, rows: list[list[int]]
) -> tuple[bytes, int]:
    """Pack palette indices into bit-plane layout.

    Each row holds "planes" consecutive, byte-padded plane runs, with the
    pixels of a plane stored MSB-first.
    """
    plane_stride = (width + 7) // 8
    out = bytearray()
    for row in rows:
        for plane in range(planes):
            run = bytearray(plane_stride)
            for x, value in enumerate(row):
                if value & (1 << plane):
                    run[x // 8] |= 0x80 >> (x % 8)
            out += run
    return bytes(out), planes * plane_stride


def packbits_literal(data: bytes) -> bytes:
    """Encode data as PackBits literal runs."""
    out = bytearray()
    for i in range(0, len(data), 128):
        chunk = data[i : i + 128]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
@pytest.mark.parametrize("width", (1, 7, 8, 9, 16, 17))
@pytest.mark.parametrize("decoder", ("raw", "hex", "packbits"))
def test_bit_plane_geometry(
    rawmode: str, planes: int, width: int, decoder: str
) -> None:
    height = 2
    max_value = (1 << planes) - 1
    rows = [[(x * 3 + y) & max_value for x in range(width)] for y in range(height)]
    packed, line_size = pack_bit_planes(width, planes, rows)

    if decoder == "raw":
        payload = packed
        im = Image.frombuffer("P", (width, height), payload, "raw", rawmode, 0, 1)
    elif decoder == "hex":
        payload = packed.hex().encode("ascii")
        im = Image.frombytes("P", (width, height), payload, "hex", rawmode)
    else:
        payload = packbits_literal(packed)
        im = Image.frombytes("P", (width, height), payload, "packbits", rawmode)

    px = im.load()
    assert line_size == planes * ((width + 7) // 8)
    for y in range(height):
        for x in range(width):
            assert px[x, y] == rows[y][x]


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
@pytest.mark.parametrize("width", (8, 9, 17))
def test_bit_plane_rows_do_not_bleed(rawmode: str, planes: int, width: int) -> None:
    # Two rows with distinct values: the padded tail of a row must never be
    # read as the start of the next row.
    rows = [
        [0 for _ in range(width)],
        [(1 << planes) - 1 for _ in range(width)],
    ]
    packed, _ = pack_bit_planes(width, planes, rows)
    im = Image.frombuffer("P", (width, 2), packed, "raw", rawmode, 0, 1)
    px = im.load()
    assert [px[x, 0] for x in range(width)] == rows[0]
    assert [px[x, 1] for x in range(width)] == rows[1]


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
def test_bit_plane_consumed_length(rawmode: str, planes: int) -> None:
    width, height = 9, 2
    rows = [[x & ((1 << planes) - 1) for x in range(width)] for _ in range(height)]
    packed, line_size = pack_bit_planes(width, planes, rows)

    # The exact geometric length decodes successfully.
    assert len(packed) == line_size * height
    im = Image.frombuffer("P", (width, height), packed, "raw", rawmode, 0, 1)
    px = im.load()
    for y in range(height):
        for x in range(width):
            assert px[x, y] == rows[y][x]

    # A trailing byte is padding and must not be consumed as image data.
    im = Image.frombuffer("P", (width, height), packed + b"\xff", "raw", rawmode, 0, 1)
    px = im.load()
    for y in range(height):
        for x in range(width):
            assert px[x, y] == rows[y][x]


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
@pytest.mark.parametrize("width", (8, 9, 17))
def test_bit_plane_truncated_raw(rawmode: str, planes: int, width: int) -> None:
    height = 2
    line_size = planes * ((width + 7) // 8)
    total = line_size * height
    for length in range(total):
        with pytest.raises(ValueError, match="not enough image data"):
            Image.frombuffer(
                "P", (width, height), b"\x00" * length, "raw", rawmode, 0, 1
            )


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
@pytest.mark.parametrize("width", (8, 9, 17))
def test_bit_plane_truncated_hex(rawmode: str, planes: int, width: int) -> None:
    height = 1
    line_size = planes * ((width + 7) // 8)
    for length in range(line_size):
        payload = (b"\x00" * length).hex().encode("ascii")
        with pytest.raises((ValueError, OSError)):
            Image.frombytes("P", (width, height), payload, "hex", rawmode)


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
@pytest.mark.parametrize("width", (8, 9, 17))
def test_bit_plane_truncated_packbits(rawmode: str, planes: int, width: int) -> None:
    height = 1
    line_size = planes * ((width + 7) // 8)

    # A run token promising more bytes than the stream contains.
    with pytest.raises((ValueError, OSError)):
        Image.frombytes("P", (width, height), b"\xf9", "packbits", rawmode)

    # A literal run that fills only part of the line before the stream ends.
    partial = bytes([line_size - 2]) + b"\x00" * (line_size - 1)
    with pytest.raises((ValueError, OSError)):
        Image.frombytes("P", (width, height), partial, "packbits", rawmode)


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
def test_bit_plane_packbits_spans_lines(rawmode: str, planes: int) -> None:
    width, height = 9, 2
    line_size = planes * ((width + 7) // 8)
    total = line_size * height

    # One literal token covering both lines must produce the same pixels as
    # the raw decoder gets from the same bytes.
    payload = bytes(range(total))
    encoded = bytes([total - 1]) + payload
    im = Image.frombytes("P", (width, height), encoded, "packbits", rawmode)
    reference = Image.frombuffer("P", (width, height), payload, "raw", rawmode, 0, 1)
    px, ref = im.load(), reference.load()
    for y in range(height):
        for x in range(width):
            assert px[x, y] == ref[x, y]

    # A single run longer than one line must also span the boundary.
    run = bytes([257 - total, 0xAA])
    im = Image.frombytes("P", (width, height), run, "packbits", rawmode)
    reference = Image.frombuffer(
        "P", (width, height), b"\xaa" * total, "raw", rawmode, 0, 1
    )
    px, ref = im.load(), reference.load()
    for y in range(height):
        for x in range(width):
            assert px[x, y] == ref[x, y]


@pytest.mark.parametrize("rawmode, planes", (("P;2L", 2), ("P;4L", 4)))
def test_bit_plane_stride(rawmode: str, planes: int) -> None:
    width, height = 9, 2
    rows = [[x & 3 for x in range(width)], [(x + 1) & 3 for x in range(width)]]
    packed, line_size = pack_bit_planes(width, planes, rows)
    stride = line_size + 2
    buffered = b"".join(
        packed[y * line_size : (y + 1) * line_size] + b"\x00\x00" for y in range(height)
    )
    im = Image.frombuffer("P", (width, height), buffered, "raw", rawmode, stride, 1)
    px = im.load()
    for y in range(height):
        for x in range(width):
            assert px[x, y] == rows[y][x]

    with pytest.raises(ValueError):
        Image.frombuffer(
            "P", (width, 1), b"\x00" * 10, "raw", rawmode, line_size - 1, 1
        )


@pytest.mark.parametrize("rawmode", ("P;2L", "P;4L"))
def test_bit_plane_zero_size(rawmode: str) -> None:
    im = Image.new("P", (0, 0))
    im.frombytes(b"", "raw", rawmode, 0, 1)


def test_packed_modes_unchanged() -> None:
    im = Image.frombuffer("P", (4, 1), b"\xe4", "raw", "P;2", 0, 1)
    assert [im.getpixel((x, 0)) for x in range(4)] == [3, 2, 1, 0]

    im = Image.frombuffer("P", (4, 1), b"\x2e\xf0", "raw", "P;4", 0, 1)
    assert [im.getpixel((x, 0)) for x in range(4)] == [2, 14, 15, 0]

    im = Image.frombuffer("P", (3, 1), b"\x01\x02\x03", "raw", "P", 0, 1)
    assert [im.getpixel((x, 0)) for x in range(3)] == [1, 2, 3]

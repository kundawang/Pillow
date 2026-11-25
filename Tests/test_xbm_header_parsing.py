from __future__ import annotations

from io import BytesIO

import pytest

from PIL import Image, UnidentifiedImageError, XbmImagePlugin

from .helper import assert_image_equal, timeout_unless_slower_valgrind

# 8x1 bitmap, single byte 0xFF -> every pixel set.
SIMPLE = (
    b"#define simple_width 8\n"
    b"#define simple_height 1\n"
    b"static char simple_bits[] = {\n"
    b"0xff\n"
    b"};\n"
)


def _open(data: bytes) -> Image.Image:
    im = XbmImagePlugin.XbmImageFile(BytesIO(data))
    im.load()
    return im


def test_simple() -> None:
    im = _open(SIMPLE)
    assert im.format == "XBM"
    assert im.mode == "1"
    assert im.size == (8, 1)
    assert im.tobytes() == b"\xff"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_line_endings(newline: bytes) -> None:
    data = SIMPLE.replace(b"\n", newline)
    im = _open(data)
    assert im.size == (8, 1)
    assert im.tobytes() == b"\xff"


@pytest.mark.parametrize("length", [1, 75, 200, 5000])
def test_long_identifier(length: int) -> None:
    name = b"A" * length
    data = (
        b"#define "
        + name
        + b"_width 8\n"
        + b"#define "
        + name
        + b"_height 1\n"
        + b"static char "
        + name
        + b"_bits[] = {\n0xff\n};\n"
    )
    im = _open(data)
    assert im.size == (8, 1)
    assert im.tobytes() == b"\xff"


def test_tabs_and_hash_spacing() -> None:
    data = (
        b"#define\tshape_width\t8\n"
        b"#  define shape_height 1\n"
        b"static\tchar\tshape_bits[] = {0xff};\n"
    )
    im = Image.open(BytesIO(data))
    im.load()
    assert im.size == (8, 1)
    assert im.tobytes() == b"\xff"


@pytest.mark.parametrize(
    "comment",
    [
        b"/* comment */\n",
        b"/*\nThis is\n a comment\n*/\n",
        b"/* " + b"A" * 500 + b" */\n",
        b"/* one *//* two */\n",
        b"// line comment\n",
    ],
)
def test_leading_comments(comment: bytes) -> None:
    with Image.open(BytesIO(comment + SIMPLE)) as im:
        assert im.format == "XBM"
        assert im.size == (8, 1)
        assert im.tobytes() == b"\xff"


def test_inline_comments() -> None:
    data = (
        b"#define x_width 8 /* width */\n"
        b"#define x_height 1 // height\n"
        b"static char x_bits[] = { /* data */ 0xff };\n"
    )
    im = _open(data)
    assert im.tobytes() == b"\xff"


def test_unterminated_comment() -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"/* never ends\n" + SIMPLE))


@pytest.mark.parametrize(
    "header",
    [
        b"#define x_height 1\nstatic char x_bits[] = {0xff};",
        b"#define x_width 8\nstatic char x_bits[] = {0xff};",
        b"#define x_width 8\n#define x_height 1\n",
        b"#define x_width missing\n#define x_height 1\nx_bits[]",
        b"#define x_width\n#define x_height 1\nx_bits[]",
        b"#define x_width -8\n#define x_height 1\nx_bits[]",
        b"not an xbm file at all\n",
    ],
)
def test_missing_or_malformed(header: bytes) -> None:
    with pytest.raises(SyntaxError):
        XbmImagePlugin.XbmImageFile(BytesIO(header))


@pytest.mark.parametrize(
    "header",
    [
        b"#define x_width 8\n#define x_width 9\n#define x_height 1\nx_bits[]",
        b"#define x_width 8\n#define x_height 1\n#define x_height 2\nx_bits[]",
        b"#define a_width 8\n#define b_width 9\n#define x_height 1\nx_bits[]",
        (
            b"#define x_width 8\n#define x_height 1\n"
            b"#define x_x_hot 0\n#define x_x_hot 1\n"
            b"#define x_y_hot 0\nx_bits[]"
        ),
    ],
)
def test_conflicting_definitions(header: bytes) -> None:
    with pytest.raises(SyntaxError):
        XbmImagePlugin.XbmImageFile(BytesIO(header))


def test_identical_duplicates_allowed() -> None:
    data = (
        b"#define x_width 8\n#define x_width 8\n"
        b"#define x_height 1\n#define x_height 1\nx_bits[]"
    )
    im = XbmImagePlugin.XbmImageFile(BytesIO(data))
    assert im.size == (8, 1)


def test_arbitrary_lines_before_bits_tolerated() -> None:
    data = (
        b"#define x_width 8\n"
        b"#define x_height 1\n"
        b"/* a comment */\n"
        b"static int unrelated = 1;\n"
        b"static char x_bits[] = {0xff};\n"
    )
    im = _open(data)
    assert im.tobytes() == b"\xff"


def test_hotspot() -> None:
    data = (
        b"#define x_width 8\n#define x_height 1\n"
        b"#define x_x_hot 3\n#define x_y_hot 5\nx_bits[]"
    )
    im = XbmImagePlugin.XbmImageFile(BytesIO(data))
    assert im.info["hotspot"] == (3, 5)


def test_pixel_data_unchanged_hopper() -> None:
    with Image.open("Tests/images/hopper.xbm") as im:
        im.load()
        assert im.size == (128, 128)
        hopper_bytes = im.tobytes()

    with Image.open("Tests/images/hopper_underscore.xbm") as underscore:
        underscore.load()
        assert underscore.size == (128, 128)
        assert underscore.tobytes() == hopper_bytes


def test_crlf_pixels_match_lf() -> None:
    lf = _open(SIMPLE)
    crlf = _open(SIMPLE.replace(b"\n", b"\r\n"))
    assert_image_equal(lf, crlf)


@timeout_unless_slower_valgrind(1)
def test_redos() -> None:
    # The historical catastrophic-backtracking header.
    redos = b""
    for prop in (b"width", b"height", b"x_hot", b"y_hot"):
        redos += b"#define" + b" " * 60 + b"_" + prop + b" 1\r"
    redos += b"\r" * 60 + b"_bits" + b"A" * 512

    with pytest.raises(SyntaxError):
        XbmImagePlugin.XbmImageFile(BytesIO(redos))


@pytest.mark.parametrize("length", [2000, 20000])
def test_long_malformed_header_is_linear(length: int) -> None:
    # Long runs of ambiguous tokens / whitespace / comments must be rejected
    # quickly rather than triggering super-linear backtracking.
    headers = [
        b"#define" + b" " * length + b"_width 1\r",
        b"#define " + b"a_width 1 \t" * length + b"\n#define x_height 1\n",
        b"/* " + b"A " * length,
        b"#define " + b"A" * length,
    ]
    for header in headers:
        with pytest.raises(SyntaxError):
            XbmImagePlugin.XbmImageFile(BytesIO(header))


def test_header_size_bound() -> None:
    # A header without a _bits[] marker is rejected once the bounded prefix is
    # exhausted, even if it is larger than the bound.
    header = b"#define x_width 8\n#define x_height 1\n" + b" " * (
        XbmImagePlugin.MAX_HEADER_BYTES + 100
    )
    with pytest.raises(SyntaxError):
        XbmImagePlugin.XbmImageFile(BytesIO(header))


def test_accept() -> None:
    assert XbmImagePlugin._accept(b"#define x_width 8")
    assert XbmImagePlugin._accept(b"   #define x_width")
    assert XbmImagePlugin._accept(b"/* XBM */")
    assert XbmImagePlugin._accept(b"// XBM")
    assert not XbmImagePlugin._accept(b"\x89PNG\r\n\x1a\n")
    assert not XbmImagePlugin._accept(b"BM6\x00\x00")

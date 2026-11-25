#
# The Python Imaging Library.
# $Id$
#
# XBM File handling
#
# History:
# 1995-09-08 fl   Created
# 1996-11-01 fl   Added save support
# 1997-07-07 fl   Made header parser more tolerant
# 1997-07-22 fl   Fixed yet another parser bug
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.4)
# 2001-05-13 fl   Added hotspot handling (based on code from Bernhard Herzog)
# 2004-02-24 fl   Allow some whitespace before first #define
#
# Copyright (c) 1997-2004 by Secret Labs AB
# Copyright (c) 1996-1997 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image, ImageFile

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import IO

# Only this many leading bytes are inspected for the width/height/hotspot
# defines and the ``_bits[]`` marker.  The bound keeps header parsing linear
# (and memory bounded) even for headers made out of very long identifiers,
# whitespace, comments or malformed macros.
MAX_HEADER_BYTES = 1024 * 1024

# Introduces the hexadecimal bitmap data; the tile starts right after it.
_BITS_MARKER = b"_bits[]"

_WIDTH_SUFFIX = b"_width"
_HEIGHT_SUFFIX = b"_height"
_X_HOT_SUFFIX = b"_x_hot"
_Y_HOT_SUFFIX = b"_y_hot"


def _accept(prefix: bytes) -> bool:
    stripped = prefix.lstrip()
    if not stripped:
        # The prefix is empty or only whitespace; the magic is inconclusive,
        # so let the bounded header parser decide.
        return True
    if stripped.startswith((b"/*", b"//")):
        return True
    if stripped.startswith(b"#"):
        # Accept both "#define" and C's "#  define" variant.
        rest = stripped[1:].lstrip()
        return not rest or rest.startswith(b"define")
    return False


def _bad_header() -> SyntaxError:
    # Keep the historical error message; callers only rely on the type.
    return SyntaxError("not a XBM file")


def _decimal(token: bytes) -> int:
    # XBM dimensions are plain decimal integers.  Check the characters in
    # linear time before calling int(); this also rejects signs, whitespace
    # and underscores.  An over-long digit run is reported as a bad header
    # instead of escaping Python's integer conversion limit.
    if not token or any(c < ord("0") or c > ord("9") for c in token):
        raise _bad_header()
    try:
        return int(token)
    except ValueError:
        raise _bad_header() from None


def _consistent(value: int | None, new_value: int) -> int:
    if value is not None and value != new_value:
        # Two definitions claim different values; the header is ambiguous.
        raise _bad_header()
    return new_value


def _mask_comments(text: bytearray) -> None:
    """Replace C/C++ comment contents with spaces, in place, in O(len(text)).

    Newline bytes are preserved and line-ending characters have already been
    normalized to LF, so line/offset arithmetic stays valid.
    """
    in_block_comment = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_block_comment:
            if c == ord("*") and i + 1 < n and text[i + 1] == ord("/"):
                text[i] = text[i + 1] = ord(" ")
                in_block_comment = False
                i += 2
            else:
                if c != ord("\n"):
                    text[i] = ord(" ")
                i += 1
        elif c == ord("/") and i + 1 < n and text[i + 1] == ord("*"):
            text[i] = text[i + 1] = ord(" ")
            in_block_comment = True
            i += 2
        elif c == ord("/") and i + 1 < n and text[i + 1] == ord("/"):
            text[i] = text[i + 1] = ord(" ")
            i += 2
            while i < n and text[i] != ord("\n"):
                text[i] = ord(" ")
                i += 1
        else:
            i += 1


def _define_name(line: bytes) -> tuple[bytes, int] | None:
    """Return the macro name and the index of its first value token."""
    tokens = line.split()
    if len(tokens) >= 2 and tokens[0] == b"#define":
        return tokens[1], 2
    # C also permits whitespace between '#' and 'define'.
    if len(tokens) >= 3 and tokens[0] == b"#" and tokens[1] == b"define":
        return tokens[2], 3
    return None


def _parse_header(data: bytes) -> tuple[int, int, tuple[int, int] | None, int]:
    """
    Identify width, height, hotspot and pixel-data offset in an XBM header.

    The input must be at most :data:`MAX_HEADER_BYTES` long.  Returns
    ``(width, height, hotspot, offset)`` where ``offset`` is relative to
    ``data`` and points immediately after the ``_bits[]`` marker.

    Raises :py:class:`SyntaxError` if required definitions are missing or
    conflicting.  The running time is linear in ``len(data)``.
    """
    # Normalize CR/CRLF to LF with a same-length replacement so offsets are
    # unchanged (CRLF becomes a harmless extra blank line).
    text = bytearray(data.replace(b"\r", b"\n"))
    _mask_comments(text)

    width: int | None = None
    height: int | None = None
    x_hot: int | None = None
    y_hot: int | None = None
    offset: int | None = None

    n = len(text)
    line_start = 0
    while line_start <= n:
        line_end = text.find(b"\n", line_start)
        if line_end < 0:
            line_end = n

        marker = text.find(_BITS_MARKER, line_start, line_end)
        if marker >= 0:
            offset = marker + len(_BITS_MARKER)
            break

        line = bytes(text[line_start:line_end]).strip()
        if line:
            if line.startswith(b"#"):
                define = _define_name(line)
                if define is not None:
                    name, value_index = define
                    is_xbm_define = name.endswith(
                        (
                            _WIDTH_SUFFIX,
                            _HEIGHT_SUFFIX,
                            _X_HOT_SUFFIX,
                            _Y_HOT_SUFFIX,
                        )
                    )
                    if is_xbm_define:
                        tokens = line.split()
                        if len(tokens) <= value_index:
                            raise _bad_header()
                        # The value is the final token; comments have already
                        # been masked away.  Any trailing non-numeric token is
                        # therefore a malformed definition.
                        value = _decimal(tokens[-1])
                        if name.endswith(_WIDTH_SUFFIX):
                            width = _consistent(width, value)
                        elif name.endswith(_HEIGHT_SUFFIX):
                            height = _consistent(height, value)
                        elif name.endswith(_X_HOT_SUFFIX):
                            x_hot = _consistent(x_hot, value)
                        else:
                            y_hot = _consistent(y_hot, value)
                # Any other line is tolerated: historical XBM headers may
                # contain arbitrary declarations between the defines and the
                # _bits[] marker, which is searched for on every line.

        if line_end == n:
            break
        line_start = line_end + 1

    if width is None or height is None or offset is None:
        raise _bad_header()

    hotspot = None
    if x_hot is not None and y_hot is not None:
        hotspot = x_hot, y_hot
    return width, height, hotspot, offset


##
# Image plugin for X11 bitmaps.


class XbmImageFile(ImageFile.ImageFile):
    format = "XBM"
    format_description = "X11 Bitmap"

    def _open(self) -> None:
        assert self.fp is not None

        header_start = self.fp.tell()
        data = self.fp.read(MAX_HEADER_BYTES)

        width, height, hotspot, offset = _parse_header(data)

        if hotspot is not None:
            self.info["hotspot"] = hotspot

        self._mode = "1"
        self._size = width, height

        self.tile = [ImageFile._Tile("xbm", (0, 0) + self.size, header_start + offset)]


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode != "1":
        msg = f"cannot write mode {im.mode} as XBM"
        raise OSError(msg)

    fp.write(f"#define im_width {im.size[0]}\n".encode("ascii"))
    fp.write(f"#define im_height {im.size[1]}\n".encode("ascii"))

    hotspot = im.encoderinfo.get("hotspot")
    if hotspot:
        fp.write(f"#define im_x_hot {hotspot[0]}\n".encode("ascii"))
        fp.write(f"#define im_y_hot {hotspot[1]}\n".encode("ascii"))

    fp.write(b"static char im_bits[] = {\n")

    ImageFile._save(im, fp, [ImageFile._Tile("xbm", (0, 0) + im.size)])

    fp.write(b"};\n")


Image.register_open(XbmImageFile.format, XbmImageFile, _accept)
Image.register_save(XbmImageFile.format, _save)

Image.register_extension(XbmImageFile.format, ".xbm")

Image.register_mime(XbmImageFile.format, "image/xbm")

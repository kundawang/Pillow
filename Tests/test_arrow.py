from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap

import pytest

from PIL import Image

from .helper import assert_image_equal, hopper, is_pypy

TYPE_CHECKING = False

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "mode, dest_modes",
    (
        ("L", ["I", "F", "LA", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "HSV"]),
        ("I", ["L", "F"]),  # Technically I;32 can work for any 4x8bit storage.
        ("F", ["I", "L", "LA", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "HSV"]),
        ("LA", ["L", "F"]),
        ("RGB", ["L", "F"]),
        ("RGBA", ["L", "F"]),
        ("RGBX", ["L", "F"]),
        ("CMYK", ["L", "F"]),
        ("YCbCr", ["L", "F"]),
        ("HSV", ["L", "F"]),
    ),
)
def test_invalid_array_type(mode: str, dest_modes: list[str]) -> None:
    img = hopper(mode)
    for dest_mode in dest_modes:
        with pytest.raises(ValueError):
            Image.fromarrow(img, dest_mode, img.size)


def test_invalid_array_size() -> None:
    img = hopper("RGB")

    assert img.size != (10, 10)
    with pytest.raises(ValueError):
        Image.fromarrow(img, "RGB", (10, 10))


def test_release_schema() -> None:
    # these should not error out, valgrind should be clean
    img = hopper("L")
    schema = img.__arrow_c_schema__()
    del schema


def test_release_array() -> None:
    # these should not error out, valgrind should be clean
    img = hopper("L")
    array, schema = img.__arrow_c_array__()
    del array
    del schema


def test_readonly() -> None:
    img = hopper("L")
    reloaded = Image.fromarrow(img, img.mode, img.size)
    assert reloaded.readonly == 1
    reloaded._readonly = 0
    assert reloaded.readonly == 1


def test_multiblock_l_image() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in single channel mode
    size = (4096, 2 * block_size // 4096)
    img = Image.new("L", size, 128)

    with pytest.raises(ValueError):
        schema, arr = img.__arrow_c_array__()


def test_multiblock_rgba_image() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in 4 channel mode
    size = (4096, (block_size // 4096) // 2)
    img = Image.new("RGBA", size, (128, 127, 126, 125))

    with pytest.raises(ValueError):
        schema, arr = img.__arrow_c_array__()


def test_multiblock_l_schema() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in single channel mode
    size = (4096, 2 * block_size // 4096)
    img = Image.new("L", size, 128)

    with pytest.raises(ValueError):
        img.__arrow_c_schema__()


def test_multiblock_rgba_schema() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in 4 channel mode
    size = (4096, (block_size // 4096) // 2)
    img = Image.new("RGBA", size, (128, 127, 126, 125))

    with pytest.raises(ValueError):
        img.__arrow_c_schema__()


@pytest.mark.usefixtures("enable_block_allocator")
def test_singleblock_l_image() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in 4 channel mode
    size = (4096, 2 * (block_size // 4096))
    img = Image.new("L", size, 128)
    assert img.im.isblock()

    schema, arr = img.__arrow_c_array__()
    assert schema
    assert arr


@pytest.mark.usefixtures("enable_block_allocator")
def test_singleblock_rgba_image() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in 4 channel mode
    size = (4096, (block_size // 4096) // 2)
    img = Image.new("RGBA", size, (128, 127, 126, 125))
    assert img.im.isblock()

    schema, arr = img.__arrow_c_array__()
    assert schema
    assert arr


@pytest.mark.usefixtures("enable_block_allocator")
def test_singleblock_l_schema() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in single channel mode
    size = (4096, 2 * block_size // 4096)
    img = Image.new("L", size, 128)
    assert img.im.isblock()

    schema = img.__arrow_c_schema__()
    assert schema


@pytest.mark.usefixtures("enable_block_allocator")
def test_singleblock_rgba_schema() -> None:
    block_size = Image.core.get_block_size()

    # check a 2 block image in 4 channel mode
    size = (4096, (block_size // 4096) // 2)
    img = Image.new("RGBA", size, (128, 127, 126, 125))
    assert img.im.isblock()

    schema = img.__arrow_c_schema__()
    assert schema


@pytest.mark.usefixtures("aligned_arena")
@pytest.mark.parametrize("mode", ("L", "RGBA"))
def test_aligned_rows_not_contiguous(mode: str) -> None:
    # Rows are padded to the alignment, so they aren't back to back
    img = hopper(mode).crop((0, 0, 3, 3))

    with pytest.raises(ValueError, match="not contiguous"):
        img.__arrow_c_array__()
    with pytest.raises(ValueError, match="not contiguous"):
        img.__arrow_c_schema__()


@pytest.mark.usefixtures("aligned_arena")
@pytest.mark.parametrize("mode, width", (("L", 64), ("RGBA", 16)))
def test_aligned_rows_contiguous(mode: str, width: int) -> None:
    # Rows are exactly the alignment, so there is no padding,
    # but the first row may start after the start of the block
    for height in range(1, 9):
        img = hopper(mode).crop((0, 0, width, height))

        reloaded = Image.fromarrow(img, img.mode, img.size)
        assert_image_equal(img, reloaded)


@pytest.mark.parametrize("mode", ("L", "RGBA"))
def test_mapped_image(mode: str) -> None:
    expected = hopper(mode).crop((0, 0, 5, 3))
    img = Image.frombuffer(mode, expected.size, expected.tobytes(), "raw", mode, 0, 1)

    reloaded = Image.fromarrow(img, img.mode, img.size)
    assert_image_equal(reloaded, expected)


@pytest.mark.parametrize(
    "mode, ext",
    (("L", "tif"), ("RGBA", "tif"), ("L", "ppm")),
)
def test_mapped_file(tmp_path: Path, mode: str, ext: str) -> None:
    expected = hopper(mode)
    filename = tmp_path / f"temp.{ext}"
    expected.save(filename)

    with Image.open(filename) as img:
        img.load()
        assert img.map is not None  # memory-mapped
        reloaded = Image.fromarrow(img, img.mode, img.size)
        assert_image_equal(reloaded, expected)


@pytest.mark.parametrize("stride, ystep", ((4, 1), (0, -1)))
def test_mapped_image_not_contiguous(stride: int, ystep: int) -> None:
    img = Image.frombuffer("L", (3, 3), bytes(range(12)), "raw", "L", stride, ystep)

    with pytest.raises(ValueError, match="not contiguous"):
        img.__arrow_c_array__()


@pytest.mark.parametrize("use_block_allocator", (0, 1))
@pytest.mark.parametrize("size", ((0, 0), (0, 3), (3, 0)))
@pytest.mark.parametrize("mode", ("L", "RGBA"))
def test_empty_image(
    mode: str, size: tuple[int, int], use_block_allocator: int
) -> None:
    Image.core.set_use_block_allocator(use_block_allocator)
    try:
        img = Image.new(mode, size)
    finally:
        Image.core.set_use_block_allocator(0)

    schema, array = img.__arrow_c_array__()
    assert schema
    assert array


@pytest.mark.parametrize("mode", ("L", "RGBA"))
def test_frombuffer_image(mode: str) -> None:
    # Trigger from GHSA-4fj2-qr5f-54hc
    # Fixed, no error from GHSA-654x-cwwv-5j9c
    size = (4, 4)
    buffer = bytes(size[0] * size[1] * Image.getmodebands(mode))
    img = Image.frombuffer(mode, size, buffer, "raw", mode, 0, 1)

    img2 = Image.fromarrow(img, mode, img.size)
    assert img2.tobytes()


def test_mapped_image_pgm(tmp_path: Path) -> None:
    # Trigger from GHSA-4fj2-qr5f-54hc
    # Fixed, no error from GHSA-654x-cwwv-5j9c
    path = tmp_path / "temp.pgm"
    hopper("L").save(path)

    with Image.open(path) as img:
        img.load()
        assert img.map is not None

        img2 = Image.fromarrow(img, img.mode, img.size)
        assert img2.tobytes()


class ArrowSchema(ctypes.Structure):
    pass


ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class ArrowArray(ctypes.Structure):
    pass


ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class MalformedArrow:
    def __init__(self, schema: ArrowSchema, array: ArrowArray) -> None:
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

        self.schema = schema
        self.array = array
        self.schema_capsule = capsule_new(
            ctypes.addressof(schema), b"arrow_schema", None
        )
        self.array_capsule = capsule_new(ctypes.addressof(array), b"arrow_array", None)

    def __arrow_c_array__(self, *args: object) -> tuple[object, object]:
        return self.schema_capsule, self.array_capsule


def _single_channel_array(
    length: int = 4, buffer_size: int = 4
) -> tuple[ArrowSchema, ArrowArray, list[object]]:
    # Keep a reference to every ctypes object so nothing is freed early
    keepalive: list[object] = []
    data = ctypes.create_string_buffer(buffer_size)
    buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
    keepalive += [data, buffers]
    schema = ArrowSchema(format=b"C")
    array = ArrowArray(length=length, n_buffers=2, buffers=buffers)
    return schema, array, keepalive


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
@pytest.mark.parametrize(
    "field, value",
    (("format", None), ("buffers", None), ("n_children", 1), ("offset", -1)),
)
def test_malformed(field: str, value: object) -> None:
    schema, array, _keepalive = _single_channel_array()
    setattr(schema if field == "format" else array, field, value)

    with pytest.raises(ValueError, match="Arrow "):
        Image.fromarrow(MalformedArrow(schema, array), "L", (2, 2))


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
@pytest.mark.parametrize(
    "field, value, match",
    (
        ("length", -1, "negative length"),
        ("length", 8, "Invalid Arrow array mode or size mismatch"),
        ("null_count", -2, "invalid null count"),
        ("null_count", 5, "invalid null count"),
        ("null_count", 1, "missing null bitmap"),
        ("offset", -(2**62), "negative offset"),
    ),
)
def test_malformed_fields(field: str, value: int, match: str) -> None:
    schema, array, _keepalive = _single_channel_array()
    setattr(array, field, value)

    with pytest.raises(ValueError, match=match):
        Image.fromarrow(MalformedArrow(schema, array), "L", (2, 2))


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
def test_malformed_offset_overflow() -> None:
    schema = ArrowSchema(format=b"i")
    data = ctypes.create_string_buffer(16)
    buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
    array = ArrowArray(length=4, n_buffers=2, offset=2**62, buffers=buffers)

    with pytest.raises(ValueError, match="offset too large"):
        Image.fromarrow(MalformedArrow(schema, array), "I", (2, 2))


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
def test_malformed_schema_child() -> None:
    children = (ctypes.POINTER(ArrowSchema) * 1)()
    schema = ArrowSchema(format=b"+w:4", n_children=1, children=children)
    array = ArrowArray(length=4)

    with pytest.raises(ValueError, match="Invalid Arrow array mode or size mismatch"):
        Image.fromarrow(MalformedArrow(schema, array), "RGBA", (2, 2))


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
def test_malformed_array_child() -> None:
    child_schema = ArrowSchema(format=b"C")
    schema_children = (ctypes.POINTER(ArrowSchema) * 1)(ctypes.pointer(child_schema))
    schema = ArrowSchema(format=b"+w:4", n_children=1, children=schema_children)
    # the child pointer itself is NULL
    array_children = (ctypes.POINTER(ArrowArray) * 1)()
    array = ArrowArray(length=4, n_children=1, children=array_children)

    with pytest.raises(ValueError, match="Invalid Arrow array mode or size mismatch"):
        Image.fromarrow(MalformedArrow(schema, array), "RGBA", (2, 2))


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
def test_malformed_array_child_buffers() -> None:
    child_schema = ArrowSchema(format=b"C")
    schema_children = (ctypes.POINTER(ArrowSchema) * 1)(ctypes.pointer(child_schema))
    schema = ArrowSchema(format=b"+w:4", n_children=1, children=schema_children)
    # child array with no buffers at all
    child = ArrowArray(length=16)
    array_children = (ctypes.POINTER(ArrowArray) * 1)(ctypes.pointer(child))
    array = ArrowArray(length=4, n_children=1, children=array_children)

    with pytest.raises(ValueError, match="exactly 2 buffers required"):
        Image.fromarrow(MalformedArrow(schema, array), "RGBA", (2, 2))


class _FakeBuffer:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakeType:
    def __init__(self, bit_width: int | None) -> None:
        self.bit_width = bit_width


class _FakeProducer:
    """A producer that exposes pyarrow-style buffer information
    alongside the arrow_c_array interface, so that the declared
    layout can be checked against the actual buffer capacity."""

    def __init__(
        self,
        img: Image.Image,
        length: int,
        data_size: int,
        null_count: int = 0,
        validity_size: int | None = None,
        bit_width: int | None = 8,
        values: _FakeProducer | None = None,
    ) -> None:
        self._capsules = img.__arrow_c_array__()
        self.offset = 0
        self.null_count = null_count
        self.type = _FakeType(bit_width)
        self._length = length
        self._buffers = [
            _FakeBuffer(validity_size) if validity_size is not None else None,
            _FakeBuffer(data_size),
        ]
        if values is not None:
            self.values = values

    def __len__(self) -> int:
        return self._length

    def buffers(self) -> list[_FakeBuffer | None]:
        return self._buffers

    def __arrow_c_array__(self, *args: object) -> tuple[object, object]:
        return self._capsules


def test_buffer_capacity() -> None:
    img = Image.new("L", (10, 10))

    # declared length fits the data buffer
    producer = _FakeProducer(img, length=100, data_size=100)
    reloaded = Image.fromarrow(producer, "L", (10, 10))
    assert_image_equal(img, reloaded)

    # data buffer smaller than the declared length
    producer = _FakeProducer(img, length=100, data_size=50)
    with pytest.raises(ValueError, match="data buffer smaller"):
        Image.fromarrow(producer, "L", (10, 10))


def test_null_bitmap_capacity() -> None:
    img = Image.new("L", (10, 10))

    # null bitmap must cover offset + length bits
    producer = _FakeProducer(img, length=100, data_size=100, null_count=1)
    with pytest.raises(ValueError, match="null bitmap smaller"):
        Image.fromarrow(producer, "L", (10, 10))

    producer = _FakeProducer(
        img, length=100, data_size=100, null_count=1, validity_size=13
    )
    reloaded = Image.fromarrow(producer, "L", (10, 10))
    assert_image_equal(img, reloaded)


def test_nested_buffer_capacity() -> None:
    img = Image.new("RGBA", (10, 10))

    # fixed-size-list style producer: capacity is checked on the child
    child = _FakeProducer(img, length=400, data_size=100)
    producer = _FakeProducer(img, length=100, data_size=0, bit_width=None, values=child)
    with pytest.raises(ValueError, match="data buffer smaller"):
        Image.fromarrow(producer, "RGBA", (10, 10))


SUBPROCESS_PREAMBLE = """
import ctypes


class ArrowSchema(ctypes.Structure):
    pass


ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class ArrowArray(ctypes.Structure):
    pass


ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class MalformedArrow:
    def __init__(self, schema, array):
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

        self.schema = schema
        self.array = array
        self.schema_capsule = capsule_new(
            ctypes.addressof(schema), b"arrow_schema", None
        )
        self.array_capsule = capsule_new(ctypes.addressof(array), b"arrow_array", None)

    def __arrow_c_array__(self, *args):
        return self.schema_capsule, self.array_capsule
"""

SUBPROCESS_CASES = {
    # data buffer pointer is NULL
    "short_buffer_null_data": """
        data = ctypes.create_string_buffer(4)
        buffers = (ctypes.c_void_p * 2)(None, None)
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=4, n_buffers=2, buffers=buffers)
        expect_value_error("buffers")
    """,
    # buffers pointer itself is NULL
    "short_buffer_no_buffers": """
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=4, n_buffers=2)
        expect_value_error("buffers")
    """,
    # declared length does not match the image dimensions
    "forged_length": """
        data = ctypes.create_string_buffer(8)
        buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=8, n_buffers=2, buffers=buffers)
        expect_value_error("size mismatch")
    """,
    "negative_length": """
        data = ctypes.create_string_buffer(4)
        buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=-1, n_buffers=2, buffers=buffers)
        expect_value_error("negative length")
    """,
    "invalid_null_count": """
        data = ctypes.create_string_buffer(4)
        buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=4, null_count=5, n_buffers=2, buffers=buffers)
        expect_value_error("invalid null count")
    """,
    "missing_null_bitmap": """
        data = ctypes.create_string_buffer(4)
        buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=4, null_count=1, n_buffers=2, buffers=buffers)
        expect_value_error("null bitmap")
    """,
    # n_children set but no child arrays: dereferenced NULL before the fix
    "missing_child": """
        data = ctypes.create_string_buffer(4)
        buffers = (ctypes.c_void_p * 2)(None, ctypes.addressof(data))
        schema = ArrowSchema(format=b"C")
        array = ArrowArray(length=4, n_buffers=2, n_children=1, buffers=buffers)
        expect_value_error("child")
    """,
    # fixed-size-list schema with a NULL child schema
    "null_schema_child": """
        children = (ctypes.POINTER(ArrowSchema) * 1)()
        schema = ArrowSchema(format=b"+w:4", n_children=1, children=children)
        array = ArrowArray(length=4)
        expect_value_error("size mismatch", mode="RGBA")
    """,
    # fixed-size-list array with a NULL child array
    "null_array_child": """
        child_schema = ArrowSchema(format=b"C")
        schema_children = (ctypes.POINTER(ArrowSchema) * 1)(
            ctypes.pointer(child_schema)
        )
        schema = ArrowSchema(format=b"+w:4", n_children=1, children=schema_children)
        array_children = (ctypes.POINTER(ArrowArray) * 1)()
        array = ArrowArray(length=4, n_children=1, children=array_children)
        expect_value_error("size mismatch", mode="RGBA")
    """,
}


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
@pytest.mark.parametrize("case", SUBPROCESS_CASES)
def test_malformed_subprocess(case: str) -> None:
    # A malformed Arrow structure must raise ValueError, not crash the
    # process, so run the import in an isolated subprocess and check the
    # exit code.
    script = SUBPROCESS_PREAMBLE + """
from PIL import Image


def expect_value_error(match, mode="L"):
    try:
        Image.fromarrow(MalformedArrow(schema, array), mode, (2, 2))
    except ValueError as e:
        assert match in str(e), str(e)
    else:
        raise AssertionError("no exception raised")
""" + textwrap.dedent(SUBPROCESS_CASES[case]) + '\nprint("OK")\n'
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


@pytest.mark.skipif(is_pypy(), reason="Requires CPython")
def test_valid_arrays_subprocess() -> None:
    # Zero-copy images keep their pixels, read-only state and shared
    # storage semantics across the arrow round trip.
    script = """
from PIL import Image

img = Image.new("L", (4, 3))
img.putdata(list(range(12)))
reloaded = Image.fromarrow(img, "L", (4, 3))
assert reloaded.readonly == 1
assert list(reloaded.getdata()) == list(range(12))

rgba = Image.new("RGBA", (3, 2))
pixels = [(i, i + 1, i + 2, i + 3) for i in range(0, 24, 4)]
rgba.putdata(pixels)
reloaded = Image.fromarrow(rgba, "RGBA", (3, 2))
assert reloaded.readonly == 1
assert list(reloaded.getdata()) == pixels

# zero-copy: the reloaded image shares storage with the original
assert reloaded.tobytes() == rgba.tobytes()

print("OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout

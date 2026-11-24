from __future__ import annotations

import math
import struct

import pytest

from PIL import Image, ImageFilter

from .helper import assert_image_equal, hopper

MODES = (
    "L",
    "LA",
    "La",
    "I",
    "I;16",
    "I;16B",
    "I;16L",
    "I;16N",
    "RGB",
    "RGBA",
    "RGBa",
    "RGBX",
    "CMYK",
)


@pytest.mark.parametrize(
    "filter_to_apply",
    (
        ImageFilter.BLUR,
        ImageFilter.CONTOUR,
        ImageFilter.DETAIL,
        ImageFilter.EDGE_ENHANCE,
        ImageFilter.EDGE_ENHANCE_MORE,
        ImageFilter.EMBOSS,
        ImageFilter.FIND_EDGES,
        ImageFilter.SMOOTH,
        ImageFilter.SMOOTH_MORE,
        ImageFilter.SHARPEN,
        ImageFilter.MaxFilter,
        ImageFilter.MedianFilter,
        ImageFilter.MinFilter,
        ImageFilter.ModeFilter,
        ImageFilter.GaussianBlur,
        ImageFilter.GaussianBlur(0),
        ImageFilter.GaussianBlur(5),
        ImageFilter.GaussianBlur((2, 5)),
        ImageFilter.BoxBlur(0),
        ImageFilter.BoxBlur(5),
        ImageFilter.BoxBlur((2, 5)),
        ImageFilter.UnsharpMask,
        ImageFilter.UnsharpMask(10),
    ),
)
@pytest.mark.parametrize("mode", MODES)
def test_sanity(
    filter_to_apply: ImageFilter.Filter | type[ImageFilter.Filter], mode: str
) -> None:
    im = hopper(mode)
    if mode[0] != "I" or (
        callable(filter_to_apply)
        and issubclass(filter_to_apply, ImageFilter.BuiltinFilter)
    ):
        out = im.filter(filter_to_apply)
        assert out.mode == im.mode
        assert out.size == im.size


@pytest.mark.parametrize("mode", MODES)
def test_sanity_error(mode: str) -> None:
    im = hopper(mode)
    with pytest.raises(TypeError):
        im.filter("hello")  # type: ignore[arg-type]


def test_noop_on_small_images() -> None:
    # If image is smaller than the kernel size, return it as-is.
    kernel_size: tuple[int, int] = ImageFilter.SMOOTH_MORE.filterargs.size
    kernel_w, kernel_h = kernel_size
    for w in range(1, kernel_w):
        for h in range(1, kernel_h):
            im = hopper("RGB").resize((w, h))
            # Precondition for the below equality test:
            # filter is larger or equal to image.
            assert im.size < kernel_size
            assert_image_equal(im.filter(ImageFilter.SMOOTH_MORE), im)


@pytest.mark.parametrize(
    "mode, expected",
    (
        ("1", (4, 0)),
        ("L", (4, 0)),
        ("P", (4, 0)),
        ("RGB", ((4, 0, 0), (0, 0, 0))),
    ),
)
def test_modefilter(
    mode: str,
    expected: tuple[int, int] | tuple[tuple[int, int, int], tuple[int, int, int]],
) -> None:
    im = Image.new(mode, (3, 3), None)
    im.putdata(list(range(9)))
    # image is:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    mod = im.filter(ImageFilter.ModeFilter).getpixel((1, 1))
    im.putdata([0, 0, 1, 2, 5, 1, 5, 2, 0])  # mode=0
    mod2 = im.filter(ImageFilter.ModeFilter).getpixel((1, 1))
    assert (mod, mod2) == expected


@pytest.mark.parametrize(
    "mode, expected",
    (
        ("1", (0, 4, 8)),
        ("L", (0, 4, 8)),
        ("RGB", ((0, 0, 0), (4, 0, 0), (8, 0, 0))),
        ("I", (0, 4, 8)),
        ("F", (0.0, 4.0, 8.0)),
    ),
)
def test_rankfilter(
    mode: str,
    expected: (
        tuple[float, float, float]
        | tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    ),
) -> None:
    im = Image.new(mode, (3, 3), None)
    im.putdata(list(range(9)))
    # image is:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    minimum = im.filter(ImageFilter.MinFilter).getpixel((1, 1))
    med = im.filter(ImageFilter.MedianFilter).getpixel((1, 1))
    maximum = im.filter(ImageFilter.MaxFilter).getpixel((1, 1))
    assert (minimum, med, maximum) == expected


I16_RANK_MODES = ("I;16", "I;16L", "I;16B", "I;16N")

# Include values whose two bytes sort in the opposite order to their numeric
# order, such as 0x00FF (255) and 0x0100 (256).
I16_RANK_MATRIX = [
    [0x0000, 0x00FF, 0x0100, 0x1234, 0xFFFF],
    [0x8000, 0x7FFF, 0x0001, 0x2345, 0xABCD],
    [0x1000, 0x0100, 0x0010, 0xF0F0, 0x0F0F],
    [0xABCD, 0xDCBA, 0xCAFE, 0xBABE, 0x1357],
    [0x2468, 0x8642, 0x9999, 0xAAAA, 0x5555],
]

RANK_MATRIX = [
    [17, 4, 22, 9, 13],
    [8, 24, 1, 15, 6],
    [20, 11, 7, 25, 3],
    [2, 18, 14, 0, 23],
    [12, 5, 19, 10, 16],
]


def rank_reference(matrix: list[list[int]], size: int, rank: int) -> list[list[int]]:
    """Return rank-filtered values using repeated-edge boundary extension."""
    margin = size // 2
    height = len(matrix)
    width = len(matrix[0])
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            window = []
            for window_y in range(y - margin, y + margin + 1):
                source_y = min(max(window_y, 0), height - 1)
                for window_x in range(x - margin, x + margin + 1):
                    source_x = min(max(window_x, 0), width - 1)
                    window.append(matrix[source_y][source_x])
            row.append(sorted(window)[rank])
        result.append(row)
    return result


@pytest.mark.parametrize("mode", I16_RANK_MODES)
@pytest.mark.parametrize("size", (1, 3, 5))
def test_rankfilter_i16_reference(mode: str, size: int) -> None:
    endian = {
        "I;16": "<",
        "I;16L": "<",
        "I;16B": ">",
        "I;16N": "=",
    }[mode]
    width = len(I16_RANK_MATRIX[0])
    height = len(I16_RANK_MATRIX)
    values = [value for row in I16_RANK_MATRIX for value in row]
    data = struct.pack(f"{endian}{len(values)}H", *values)
    im = Image.frombytes(mode, (width, height), data)

    filters_and_ranks = (
        (ImageFilter.MinFilter(size), 0),
        (ImageFilter.MedianFilter(size), size * size // 2),
        (ImageFilter.MaxFilter(size), size * size - 1),
    )
    for rank_filter, rank in filters_and_ranks:
        result = im.filter(rank_filter)
        expected = rank_reference(I16_RANK_MATRIX, size, rank)

        assert result.mode == mode
        assert result.size == im.size
        for y, row in enumerate(expected):
            for x, value in enumerate(row):
                assert result.getpixel((x, y)) == value, (
                    mode,
                    rank_filter.name,
                    size,
                    (x, y),
                )


@pytest.mark.parametrize("mode", ("L", "I", "F"))
@pytest.mark.parametrize("size", (1, 3, 5))
def test_rankfilter_integer_and_float_reference(mode: str, size: int) -> None:
    width = len(RANK_MATRIX[0])
    height = len(RANK_MATRIX)
    im = Image.new(mode, (width, height))
    im.putdata([value for row in RANK_MATRIX for value in row])

    filters_and_ranks = (
        (ImageFilter.MinFilter(size), 0),
        (ImageFilter.MedianFilter(size), size * size // 2),
        (ImageFilter.MaxFilter(size), size * size - 1),
    )
    for rank_filter, rank in filters_and_ranks:
        result = im.filter(rank_filter)
        expected = rank_reference(RANK_MATRIX, size, rank)

        assert result.mode == mode
        for y, row in enumerate(expected):
            for x, value in enumerate(row):
                assert result.getpixel((x, y)) == value


@pytest.mark.parametrize("mode", ("L", *I16_RANK_MODES))
@pytest.mark.parametrize(
    "attribute, value, message",
    (
        ("size", 2, "bad filter size"),
        ("rank", 9, "bad rank value"),
    ),
)
def test_rankfilter_invalid_arguments_match_8bit_path(
    mode: str, attribute: str, value: int, message: str
) -> None:
    im = Image.new(mode, (4, 4), 1000)
    rank_filter = ImageFilter.RankFilter(3, 0)
    setattr(rank_filter, attribute, value)

    with pytest.raises(ValueError, match=message):
        im.filter(rank_filter)


@pytest.mark.parametrize("mode", I16_RANK_MODES)
@pytest.mark.parametrize("size", ((0, 1), (1, 0), (0, 0)))
def test_rankfilter_i16_zero_dimension(mode: str, size: tuple[int, int]) -> None:
    im = Image.new(mode, size)
    filters = (
        ImageFilter.MinFilter(1),
        ImageFilter.MedianFilter(1),
        ImageFilter.MaxFilter(1),
        ImageFilter.RankFilter(3, 1),
    )
    for rank_filter in filters:
        result = im.filter(rank_filter)
        assert result.mode == mode
        assert result.size == size


@pytest.mark.parametrize(
    "filter", (ImageFilter.MinFilter, ImageFilter.MedianFilter, ImageFilter.MaxFilter)
)
def test_rankfilter_error(filter: ImageFilter.RankFilter) -> None:
    with pytest.raises(ValueError):
        im = Image.new("P", (3, 3), None)
        im.putdata(list(range(9)))
        # image is:
        #   0 1 2
        #   3 4 5
        #   6 7 8
        im.filter(filter).getpixel((1, 1))


@pytest.mark.parametrize(
    "filter",
    (
        ImageFilter.MinFilter(1),
        ImageFilter.MedianFilter(1),
        ImageFilter.MaxFilter(1),
        ImageFilter.RankFilter(3, 1),
    ),
)
@pytest.mark.parametrize("size", ((0, 1), (1, 0), (0, 0)))
def test_rankfilter_zero_dimension(
    filter: ImageFilter.RankFilter, size: tuple[int, int]
) -> None:
    im = Image.new("L", size)
    assert im.filter(filter).size == size


def test_rankfilter_properties() -> None:
    rankfilter = ImageFilter.RankFilter(3, 2)

    assert rankfilter.size == 3
    assert rankfilter.rank == 2

    with pytest.raises(ValueError, match="bad filter size"):
        ImageFilter.RankFilter(2, 1)
    with pytest.raises(ValueError, match="bad filter size"):
        ImageFilter.MaxFilter(2)
    with pytest.raises(ValueError, match="bad filter size"):
        ImageFilter.MedianFilter(2)
    with pytest.raises(ValueError, match="bad filter size"):
        ImageFilter.MinFilter(2)

    with pytest.raises(ValueError, match="filter size too large"):
        ImageFilter.RankFilter(23171, 1)
    im = Image.new("1", (1, 1))
    with pytest.raises(ValueError, match="filter size too large"):
        im.im.expand(23171)

    with pytest.raises(ValueError, match="bad rank value"):
        ImageFilter.RankFilter(1, 1)


def test_rankfilter_overflow() -> None:
    # Large margins used to overflow the ImagingExpand overflow guard itself (SIGFPE),
    # by mutating RankFilter.size after construction, bypassing __init__'s validation.
    im = Image.new("L", (16, 16))
    rankfilter = ImageFilter.RankFilter(3, 0)

    for size in (2**31, 2**32 - 1):  # margins of 2**30 and INT_MAX
        rankfilter.size = size
        with pytest.raises(ValueError, match="filter size too large"):
            im.filter(rankfilter)


@pytest.mark.parametrize("mode", ("P", "PA"))
def test_builtinfilter_palette(mode: str) -> None:
    builtin_filter = ImageFilter.BuiltinFilter()

    im = hopper(mode)
    with pytest.raises(ValueError, match="cannot filter palette images"):
        builtin_filter.filter(im.im)


def test_kernel_not_enough_coefficients() -> None:
    with pytest.raises(ValueError):
        ImageFilter.Kernel((3, 3), (0, 0))


EMBOSS_MATRIX = {
    3: (
        -1, -1,  0,
        -1,  0,  1,
         0,  1,  1,
    ),
    5: (
        -1, -1, -1, -1,  0,
        -1, -1, -1,  0,  1,
        -1, -1,  0,  1,  1,
        -1,  0,  1,  1,  1,
         0,  1,  1,  1,  1,
    ),
}  # fmt: skip


@pytest.mark.parametrize("size", (3, 5))
def test_consistency(size: int) -> None:
    kernel = ImageFilter.Kernel((size, size), EMBOSS_MATRIX[size], 0.3)
    with Image.open("Tests/images/hopper.bmp") as source:
        with Image.open(f"Tests/images/hopper_emboss_{size}x{size}.bmp") as reference:
            assert_image_equal(source.filter(kernel), reference)


@pytest.mark.parametrize("size", (3, 5))
@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_consistency_i16(size: int, mode: str) -> None:
    kernel = ImageFilter.Kernel((size, size), EMBOSS_MATRIX[size], 0.3)
    reference = hopper("I").filter(kernel)
    result = hopper(mode).filter(kernel)
    # I;16* images cannot have negative values, but I images can
    # Convert the reference image before comparing so that both are clipped
    assert_image_equal(result, reference.convert(mode))


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_consistency_i16_high_byte(mode: str) -> None:
    # Exercise filters with a 16bpc image that has content in the high byte, too.
    im = Image.new(mode, (8, 8), 1000)
    # Ensure repeated smoothing retains the exact same color value,
    # rather than drifting due to rounding errors.
    for _ in range(5):
        im = im.filter(ImageFilter.SMOOTH)
        assert im.getpixel((4, 4)) == 1000


def test_i_negative_values() -> None:
    # I mode is signed, so filtering must keep negative pixel values
    im = Image.new("I", (8, 8), -1000)
    assert im.filter(ImageFilter.SMOOTH).getpixel((4, 4)) == -1000


@pytest.mark.parametrize(
    "filter_to_apply",
    (
        ImageFilter.GaussianBlur,
        ImageFilter.GaussianBlur(0),
        ImageFilter.GaussianBlur(5),
        ImageFilter.GaussianBlur((2, 5)),
        ImageFilter.BoxBlur(0),
        ImageFilter.BoxBlur(5),
        ImageFilter.BoxBlur((2, 5)),
    ),
)
@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_blur_i16(
    filter_to_apply: ImageFilter.Filter | type[ImageFilter.Filter], mode: str
) -> None:
    im = hopper("L")
    reference = im.filter(filter_to_apply)
    result = im.convert(mode).filter(filter_to_apply)
    assert result.mode == mode
    assert_image_equal(result.convert("L"), reference)


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_unsharp_mask_i16(mode: str) -> None:
    with pytest.raises(ValueError, match="image has wrong mode"):
        hopper(mode).filter(ImageFilter.UnsharpMask)


@pytest.mark.parametrize("size", (-1, math.nan, math.inf, 2**31))
def test_invalid_box_blur_filter(size: int) -> None:
    for radius in (size, (size, size), (size, 1), (1, size)):
        with pytest.raises(ValueError, match="radius"):
            ImageFilter.BoxBlur(radius)

        im = hopper()
        box_blur_filter = ImageFilter.BoxBlur(2)
        box_blur_filter.radius = radius
        with pytest.raises(ValueError, match="radius"):
            im.filter(box_blur_filter)


@pytest.mark.parametrize("radius", (math.nan, math.inf, 2**31))
def test_invalid_gaussian_blur_filter(radius: int) -> None:
    im = hopper()
    with pytest.raises(ValueError, match="radius"):
        im.filter(ImageFilter.GaussianBlur(radius))


def test_rankfilter_size_1() -> None:
    im = Image.new("L", (3, 3), 128)

    # Size 1 should not crash (margin is 0)
    assert im.filter(ImageFilter.MinFilter(1)).getpixel((1, 1)) == 128
    assert im.filter(ImageFilter.MaxFilter(1)).getpixel((1, 1)) == 128
    assert im.filter(ImageFilter.MedianFilter(1)).getpixel((1, 1)) == 128
    assert im.filter(ImageFilter.RankFilter(1, 0)).getpixel((1, 1)) == 128

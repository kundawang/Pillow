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
    kernel_size: tuple[int, int] = ImageFilter.SMOOTH_MORE.filterargs[0]
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


def test_builtinfilter_p() -> None:
    builtin_filter = ImageFilter.BuiltinFilter()

    with pytest.raises(ValueError):
        builtin_filter.filter(hopper("P").im)


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
    assert_image_equal(result.convert("I"), reference)


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_consistency_i16_high_byte(mode: str) -> None:
    # Exercise filters with a 16bpc image that has content in the high byte, too.
    im = Image.new(mode, (8, 8), 1000)
    # Ensure repeated smoothing retains the exact same color value,
    # rather than drifting due to rounding errors.
    for _ in range(5):
        im = im.filter(ImageFilter.SMOOTH)
        assert im.getpixel((4, 4)) == 1000


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


def _naive_rank(
    data: list[int], width: int, height: int, size: int, rank: int
) -> list[int]:
    """Reference rank filter on the output-sized image.

    ``Image.filter`` first expands the border by ``size // 2`` using edge
    replication and then filters, so every output pixel is the ``rank``-th
    smallest value of a full window of the expanded image.
    """
    margin = size // 2

    def pixel(x: int, y: int) -> int:
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        return data[y * width + x]

    out = []
    for y in range(height):
        for x in range(width):
            window = [
                pixel(x + dx - margin, y + dy - margin)
                for dy in range(size)
                for dx in range(size)
            ]
            window.sort()
            out.append(window[rank])
    return out


@pytest.mark.parametrize("mode", ("L", "I"))
@pytest.mark.parametrize("size", (1, 3, 5, 7))
def test_rankfilter_extremes_match_naive(mode: str, size: int) -> None:
    # Small matrix with duplicated and extreme values, covering window edges,
    # the special-cased min/max ranks, and a different odd window size.
    width = height = 11
    patterns = {
        "L": (0, 255, 0, 255, 128, 3, 3, 200),
        "I": (-(2**31), 2**31 - 1, 0, 7, -7),
    }
    pattern = patterns[mode]
    values = [pattern[n % len(pattern)] for n in range(width * height)]
    im = Image.new(mode, (width, height))
    im.putdata(values)

    for filter_class, rank in (
        (ImageFilter.MinFilter, 0),
        (ImageFilter.MaxFilter, size * size - 1),
        (ImageFilter.MedianFilter, size * size // 2),
    ):
        expected = _naive_rank(values, width, height, size, rank)
        result = list(im.filter(filter_class(size)).get_flattened_data())
        assert result == expected


@pytest.mark.parametrize("size", (1, 3, 5, 9))
def test_rankfilter_min_max_uniform_image(size: int) -> None:
    im = Image.new("L", (16, 16), 77)
    for filter_class in (ImageFilter.MinFilter, ImageFilter.MaxFilter):
        assert im.filter(filter_class(size)).getextrema() == (77, 77)


def test_rankfilter_float_nan_and_signed_zero_unchanged() -> None:
    # Windows containing NaN or signed zero are not ordered by the comparison
    # operators.  F mode keeps using rank selection, so the exact bits returned
    # for such windows must not change.
    im = Image.frombytes(
        "F", (3, 3), struct.pack("<9f", 5.0, -0.0, 0.0, 9, 9, 9, 9, 9, 9)
    )
    assert (
        math.copysign(1.0, im.filter(ImageFilter.MinFilter(3)).getpixel((1, 1))) == -1.0
    )

    nan = float("nan")
    im = Image.frombytes(
        "F", (3, 3), struct.pack("<9f", nan, 3.0, 2.0, 1, 4, 5, 0, 8, 7)
    )
    minimum = im.filter(ImageFilter.MinFilter(3))
    maximum = im.filter(ImageFilter.MaxFilter(3))
    assert minimum.getpixel((1, 1)) == 0.0
    assert math.isnan(maximum.getpixel((0, 0)))
    assert math.isnan(maximum.getpixel((1, 0)))
    assert maximum.getpixel((1, 1)) == 8.0

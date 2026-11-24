from __future__ import annotations

import pytest

from PIL import Image

from .helper import PillowLeakTestCase, assert_image_equal, hopper


@pytest.mark.parametrize(
    "mode", ("1", "L", "LA", "I", "F", "P", "RGB", "RGBA", "CMYK", "YCbCr")
)
def test_merge_roundtrip(mode: str) -> None:
    # A successful merge must reproduce the source image exactly
    im = hopper(mode)
    merged = Image.merge(mode, im.split())
    assert merged.mode == im.mode
    assert_image_equal(im, merged)


def test_merge_alpha() -> None:
    # The alpha band must survive a split/merge roundtrip unchanged
    im = hopper("RGBA")
    merged = Image.merge("RGBA", im.split())
    assert_image_equal(im.getchannel("A"), merged.getchannel("A"))


def test_merge_single_band() -> None:
    # Merging a single band returns a copy of that band
    for mode in ("L", "I", "F", "P"):
        im = hopper(mode)
        merged = Image.merge(mode, (im,))
        assert merged.mode == mode
        assert_image_equal(im, merged)


def test_merge_wrong_number_of_bands() -> None:
    band = hopper("L")
    with pytest.raises(ValueError, match="wrong number of bands"):
        Image.merge("RGB", (band, band))
    with pytest.raises(ValueError, match="wrong number of bands"):
        Image.merge("RGB", (band, band, band, band))


def test_merge_size_mismatch() -> None:
    band = Image.new("L", (10, 10))
    other = Image.new("L", (11, 10))
    with pytest.raises(ValueError, match="size mismatch"):
        Image.merge("RGB", (band, other, band))


def test_merge_mode_mismatch() -> None:
    band = hopper("L")
    other = hopper("I")
    # bands[1:] are validated against the output mode by the Python layer
    with pytest.raises(ValueError, match="mode mismatch"):
        Image.merge("RGB", (band, other, band))
    # bands[0] reaches the C layer, which rejects multi-band storage
    with pytest.raises(ValueError, match="image has wrong mode"):
        Image.merge("RGB", (other, band, band))


def test_merge_single_band_mode_mismatch() -> None:
    # The output image is allocated before the band copy fails,
    # exercising the partial-allocation failure path in ImagingMerge
    with pytest.raises(ValueError, match="images do not match"):
        Image.merge("I", (hopper("L"),))
    with pytest.raises(ValueError, match="images do not match"):
        Image.merge("L", (hopper("I"),))


class TestMergeLeak(PillowLeakTestCase):
    # Small images, many iterations: any per-call leak in the C failure
    # paths must stay well under this bound.
    iterations = 5000
    mem_limit = 512  # kB

    def test_leak_merge_wrong_number_of_bands(self) -> None:
        band = Image.new("L", (25, 25))

        def core() -> None:
            with pytest.raises(ValueError, match="wrong number of bands"):
                Image.merge("RGBA", (band, band))

        self._test_leak(core)

    def test_leak_merge_size_mismatch(self) -> None:
        band = Image.new("L", (25, 25))
        other = Image.new("L", (26, 25))

        def core() -> None:
            with pytest.raises(ValueError, match="size mismatch"):
                Image.merge("RGB", (band, other, band))

        self._test_leak(core)

    def test_leak_merge_mode_mismatch(self) -> None:
        band = Image.new("L", (25, 25))
        other = Image.new("I", (25, 25))

        def core() -> None:
            with pytest.raises(ValueError, match="image has wrong mode"):
                Image.merge("RGB", (other, band, band))

        self._test_leak(core)

    def test_leak_merge_single_band_mode_mismatch(self) -> None:
        # Partial-allocation path: imOut is allocated, then the copy fails
        band = Image.new("L", (25, 25))

        def core() -> None:
            with pytest.raises(ValueError, match="images do not match"):
                Image.merge("I", (band,))

        self._test_leak(core)

    def test_leak_merge_success(self) -> None:
        # Control: a fixed successful merge must not grow memory either
        bands = tuple(Image.new("L", (25, 25)) for _ in range(4))

        def core() -> None:
            Image.merge("RGBA", bands)

        self._test_leak(core)

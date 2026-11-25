"""
Tests for resize functionality.
"""

from __future__ import annotations

import sys
from itertools import permutations

import pytest

from PIL import Image, ImageFile

from .helper import (
    assert_image_equal,
    assert_image_equal_tofile,
    assert_image_similar,
    hopper,
    skip_unless_feature,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class TestImagingCoreResize:
    def resize(
        self, im: Image.Image, size: tuple[int, int], f: Image.Resampling
    ) -> Image.Image:
        # Image class independent version of resize.
        im.load()
        return im._new(im.im.resize(size, f))

    @pytest.mark.parametrize(
        "mode", ("1", "P", "L", "I", "F", "RGB", "RGBA", "CMYK", "YCbCr", "I;16")
    )
    def test_nearest_mode(self, mode: str) -> None:
        im = hopper(mode)
        r = self.resize(im, (15, 12), Image.Resampling.NEAREST)
        assert r.mode == mode
        assert r.size == (15, 12)
        assert r.im.bands == im.im.bands

    def test_convolution_modes(self) -> None:
        with pytest.raises(ValueError):
            self.resize(hopper("1"), (15, 12), Image.Resampling.BILINEAR)
        with pytest.raises(ValueError):
            self.resize(hopper("P"), (15, 12), Image.Resampling.BILINEAR)
        for mode in [
            "L",
            "I",
            "I;16",
            "I;16L",
            "I;16B",
            "I;16N",
            "F",
            "RGB",
            "RGBA",
            "CMYK",
            "YCbCr",
        ]:
            im = hopper(mode)
            r = self.resize(im, (15, 12), Image.Resampling.BILINEAR)
            assert r.mode == mode
            assert r.size == (15, 12)
            assert r.im.bands == im.im.bands

    @pytest.mark.parametrize(
        "resample",
        (
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
            Image.Resampling.MKS2013,
            Image.Resampling.MKS2021,
        ),
    )
    def test_i16_resample_endianness(self, resample: Image.Resampling) -> None:
        # Resampling must interpret the two bytes of each pixel as a single
        # 16-bit value, so all four I;16 modes holding the same numeric data
        # must produce numerically identical output. Use values whose high and
        # low bytes differ to make a byte swap obvious.
        values = [
            0x0000,
            0x00FF,
            0xFF00,
            0xFFFF,
            0x1234,
            0x4321,
            0xABCD,
            0xDCBA,
            0x0102,
            0x0304,
            0x0506,
            0x0708,
            0xF0F0,
            0x0F0F,
            0xAAAA,
            0x5555,
        ]

        def make(mode: str) -> Image.Image:
            if mode == "I;16B":
                raw = b"".join(bytes((v >> 8, v & 0xFF)) for v in values)
            elif mode == "I;16N" and sys.byteorder == "big":
                raw = b"".join(bytes((v >> 8, v & 0xFF)) for v in values)
            else:
                raw = b"".join(bytes((v & 0xFF, v >> 8)) for v in values)
            return Image.frombytes(mode, (4, 4), raw)

        reference = self.resize(make("I;16L"), (7, 5), resample).get_flattened_data()
        for mode in ("I;16", "I;16B", "I;16N"):
            resized = self.resize(make(mode), (7, 5), resample)
            assert resized.mode == mode
            assert resized.size == (7, 5)
            assert resized.get_flattened_data() == reference

    def test_i16_resample_byte_order(self) -> None:
        # The resampled output must keep each mode's on-disk byte order.
        values = [0x1234, 0xABCD, 0x00FF, 0xFF00]
        little_raw = b"".join(bytes((v & 0xFF, v >> 8)) for v in values)
        big_raw = b"".join(bytes((v >> 8, v & 0xFF)) for v in values)
        native_raw = little_raw if sys.byteorder == "little" else big_raw
        for mode, raw in (
            ("I;16", little_raw),
            ("I;16L", little_raw),
            ("I;16B", big_raw),
            ("I;16N", native_raw),
        ):
            resized = self.resize(
                Image.frombytes(mode, (2, 2), raw), (4, 4), Image.Resampling.BILINEAR
            )
            assert resized.mode == mode
            # Numeric corner value is preserved by the edge weights.
            assert resized.getpixel((0, 0)) == 0x1234
            # And the two bytes are stored in the mode's declared order.
            assert resized.tobytes()[0:2] == raw[0:2]

    @pytest.mark.parametrize(
        "resample",
        (
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ),
    )
    def test_i16_resample_extrema(self, resample: Image.Resampling) -> None:
        # Uniform 0 and 65535 images must keep their values through resampling,
        # and every result has to stay within the 16-bit range.
        for value in (0, 65535):
            for mode in ("I;16", "I;16L", "I;16B", "I;16N"):
                im = Image.new(mode, (4, 4), value)
                resized = self.resize(im, (7, 5), resample)
                assert resized.mode == mode
                assert set(resized.get_flattened_data()) == {value}, (
                    mode,
                    resample,
                    value,
                )

    @pytest.mark.parametrize(
        "resample",
        (
            Image.Resampling.NEAREST,
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
            Image.Resampling.MKS2013,
            Image.Resampling.MKS2021,
        ),
    )
    def test_reduce_filters(self, resample: Image.Resampling) -> None:
        r = self.resize(hopper("RGB"), (15, 12), resample)
        assert r.mode == "RGB"
        assert r.size == (15, 12)

    @pytest.mark.parametrize(
        "resample",
        (
            Image.Resampling.NEAREST,
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
            Image.Resampling.MKS2013,
            Image.Resampling.MKS2021,
        ),
    )
    def test_enlarge_filters(self, resample: Image.Resampling) -> None:
        r = self.resize(hopper("RGB"), (212, 195), resample)
        assert r.mode == "RGB"
        assert r.size == (212, 195)

    @pytest.mark.parametrize(
        "resample",
        (
            Image.Resampling.NEAREST,
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
            Image.Resampling.MKS2013,
            Image.Resampling.MKS2021,
        ),
    )
    @pytest.mark.parametrize(
        "mode, channels_set",
        (
            ("RGB", ("blank", "filled", "dirty")),
            ("RGBA", ("blank", "blank", "filled", "dirty")),
            ("LA", ("filled", "dirty")),
        ),
    )
    def test_endianness(
        self, resample: Image.Resampling, mode: str, channels_set: tuple[str, ...]
    ) -> None:
        # Make an image with one colored pixel, in one channel.
        # When resized, that channel should be the same as a GS image.
        # Other channels should be unaffected.
        # The R and A channels should not swap, which is indicative of
        # an endianness issues.

        samples = {
            "blank": Image.new("L", (2, 2), 0),
            "filled": Image.new("L", (2, 2), 255),
            "dirty": Image.new("L", (2, 2), 0),
        }
        samples["dirty"].putpixel((1, 1), 128)

        # samples resized with current filter
        references = {
            name: self.resize(ch, (4, 4), resample) for name, ch in samples.items()
        }

        for channels in set(permutations(channels_set)):
            # compile image from different channels permutations
            im = Image.merge(mode, [samples[ch] for ch in channels])
            resized = self.resize(im, (4, 4), resample)

            for i, ch in enumerate(resized.split()):
                # check what resized channel in image is the same
                # as separately resized channel
                assert_image_equal(ch, references[channels[i]])

    @pytest.mark.parametrize(
        "resample",
        (
            Image.Resampling.NEAREST,
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
            Image.Resampling.MKS2013,
            Image.Resampling.MKS2021,
        ),
    )
    def test_enlarge_zero(self, resample: Image.Resampling) -> None:
        r = self.resize(Image.new("RGB", (0, 0), "white"), (212, 195), resample)
        assert r.mode == "RGB"
        assert r.size == (212, 195)
        assert r.getpixel((0, 0)) == (0, 0, 0)

    def test_unknown_filter(self) -> None:
        with pytest.raises(ValueError):
            self.resize(hopper(), (10, 10), 9)  # type: ignore[arg-type]

    def test_cross_platform(self, tmp_path: Path) -> None:
        # This test is intended for only check for consistent behaviour across
        # platforms. So if a future Pillow change requires that the test file
        # be updated, that is okay.
        im = hopper().resize((64, 64))
        temp_file = tmp_path / "temp.gif"
        im.save(temp_file)

        with Image.open(temp_file) as reloaded:
            assert_image_equal_tofile(reloaded, "Tests/images/hopper_resized.gif")


@pytest.fixture
def gradients_image() -> Generator[ImageFile.ImageFile]:
    with Image.open("Tests/images/radial_gradients.png") as im:
        im.load()
    try:
        yield im
    finally:
        im.close()


class TestReducingGapResize:
    def test_reducing_gap_values(self, gradients_image: ImageFile.ImageFile) -> None:
        ref = gradients_image.resize(
            (52, 34), Image.Resampling.BICUBIC, reducing_gap=None
        )
        im = gradients_image.resize((52, 34), Image.Resampling.BICUBIC)
        assert_image_equal(ref, im)

        with pytest.raises(ValueError):
            gradients_image.resize((52, 34), Image.Resampling.BICUBIC, reducing_gap=0)

        with pytest.raises(ValueError):
            gradients_image.resize(
                (52, 34), Image.Resampling.BICUBIC, reducing_gap=0.99
            )

    @pytest.mark.parametrize(
        "box, epsilon",
        ((None, 4), ((1.1, 2.2, 510.8, 510.9), 4), ((3, 10, 410, 256), 10)),
    )
    def test_reducing_gap_1(
        self,
        gradients_image: ImageFile.ImageFile,
        box: tuple[float, float, float, float],
        epsilon: float,
    ) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize(
            (52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=1.0
        )

        assert_image_similar(ref, im, epsilon)
        assert ref.tobytes() != im.tobytes()

    @pytest.mark.parametrize(
        "box, epsilon",
        ((None, 1.5), ((1.1, 2.2, 510.8, 510.9), 1.5), ((3, 10, 410, 256), 1)),
    )
    def test_reducing_gap_2(
        self,
        gradients_image: ImageFile.ImageFile,
        box: tuple[float, float, float, float],
        epsilon: float,
    ) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize(
            (52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=2.0
        )

        assert ref.tobytes() != im.tobytes()
        assert_image_similar(ref, im, epsilon)

    @pytest.mark.parametrize(
        "box, epsilon",
        ((None, 1), ((1.1, 2.2, 510.8, 510.9), 1), ((3, 10, 410, 256), 0.5)),
    )
    def test_reducing_gap_3(
        self,
        gradients_image: ImageFile.ImageFile,
        box: tuple[float, float, float, float],
        epsilon: float,
    ) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize(
            (52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=3.0
        )

        assert ref.tobytes() != im.tobytes()
        assert_image_similar(ref, im, epsilon)

    @pytest.mark.parametrize("box", (None, (1.1, 2.2, 510.8, 510.9), (3, 10, 410, 256)))
    def test_reducing_gap_8(
        self,
        gradients_image: ImageFile.ImageFile,
        box: tuple[float, float, float, float],
    ) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize(
            (52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=8.0
        )

        assert_image_equal(ref, im)

    @pytest.mark.parametrize("mode", ("RGBA", "LA"))
    def test_reducing_gap_alpha(self, mode: str) -> None:
        with Image.open("Tests/images/transparent.png") as im:
            im_converted = im.convert(mode)

        # Premultiply the alpha manually
        premultiplied = im_converted.convert({"RGBA": "RGBa", "LA": "La"}[mode])
        expected = premultiplied.resize((16, 16), reducing_gap=2).convert(mode)

        # Test that resizing the image applies the alpha in the same way
        assert_image_equal(im_converted.resize((16, 16), reducing_gap=2), expected)

    @pytest.mark.parametrize(
        "box, epsilon",
        (((0, 0, 512, 512), 5.5), ((0.9, 1.7, 128, 128), 9.5)),
    )
    def test_box_filter(
        self,
        gradients_image: ImageFile.ImageFile,
        box: tuple[float, float, float, float],
        epsilon: float,
    ) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BOX, box=box)
        im = gradients_image.resize(
            (52, 34), Image.Resampling.BOX, box=box, reducing_gap=1.0
        )

        assert_image_similar(ref, im, epsilon)


class TestImageResize:
    def test_resize(self) -> None:
        def resize(mode: str, size: tuple[int, int] | list[int]) -> None:
            out = hopper(mode).resize(size)
            assert out.mode == mode
            assert out.size == tuple(size)

        for mode in "1", "P", "L", "RGB", "I", "F":
            resize(mode, (112, 103))
            resize(mode, [188, 214])

        # Test unknown resampling filter
        with hopper() as im:
            with pytest.raises(ValueError):
                im.resize((10, 10), -1)

    @skip_unless_feature("libtiff")
    def test_transposed(self) -> None:
        with Image.open("Tests/images/g4_orientation_5.tif") as im:
            im_resized = im.resize((64, 64))
            assert im_resized.size == (64, 64)

    @pytest.mark.parametrize(
        "mode", ("L", "RGB", "I", "I;16", "I;16L", "I;16B", "I;16N", "F")
    )
    def test_default_filter_bicubic(self, mode: str) -> None:
        im = hopper(mode)
        assert im.resize((20, 20), Image.Resampling.BICUBIC) == im.resize((20, 20))

    @pytest.mark.parametrize("mode", ("1", "P"))
    def test_default_filter_nearest(self, mode: str) -> None:
        im = hopper(mode)
        assert im.resize((20, 20), Image.Resampling.NEAREST) == im.resize((20, 20))

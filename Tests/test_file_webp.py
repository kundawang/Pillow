from __future__ import annotations

import contextlib
import io
import re
import sys
import warnings
from typing import Any

import pytest

from PIL import Image, WebPImagePlugin, features

from .helper import (
    assert_image_equal,
    assert_image_similar,
    assert_image_similar_tofile,
    hopper,
    skip_unless_feature,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path

try:
    from PIL import _webp

    HAVE_WEBP = True
except ImportError:
    HAVE_WEBP = False


class TestUnsupportedWebp:
    def test_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if HAVE_WEBP:
            monkeypatch.setattr(WebPImagePlugin, "SUPPORTED", False)

        file_path = "Tests/images/hopper.webp"
        with pytest.raises(OSError):
            with pytest.warns(UserWarning, match="WEBP support not installed"):
                with Image.open(file_path):
                    pass


@skip_unless_feature("webp")
class TestFileWebp:
    def setup_method(self) -> None:
        self.rgb_mode = "RGB"

    def test_version(self) -> None:
        version = features.version_module("webp")
        assert version is not None
        assert re.search(r"\d+\.\d+\.\d+$", version)

    def test_invalid_file(self) -> None:
        invalid_file = "Tests/images/flower.jpg"

        with pytest.raises(SyntaxError):
            WebPImagePlugin.WebPImageFile(invalid_file)

    def test_read_rgb(self) -> None:
        """
        Can we read a RGB mode WebP file without error?
        Does it have the bits we expect?
        """

        with Image.open("Tests/images/hopper.webp") as image:
            assert image.mode == self.rgb_mode
            assert image.size == (128, 128)
            assert image.format == "WEBP"
            image.load()

            # generated with:
            # dwebp -ppm ../../Tests/images/hopper.webp -o hopper_webp_bits.ppm
            assert_image_similar_tofile(image, "Tests/images/hopper_webp_bits.ppm", 1.0)

    def _roundtrip(
        self, tmp_path: Path, mode: str, epsilon: float, args: dict[str, Any] = {}
    ) -> None:
        temp_file = tmp_path / "temp.webp"

        hopper(mode).save(temp_file, **args)
        with Image.open(temp_file) as image:
            assert image.mode == self.rgb_mode
            assert image.size == (128, 128)
            assert image.format == "WEBP"
            image.load()

            if mode == self.rgb_mode:
                # generated with: dwebp -ppm temp.webp -o hopper_webp_write.ppm
                assert_image_similar_tofile(
                    image, "Tests/images/hopper_webp_write.ppm", 12.0
                )

            # This test asserts that the images are similar. If the average pixel
            # difference between the two images is less than the epsilon value,
            # then we're going to accept that it's a reasonable lossy version of
            # the image.
            target = hopper(mode)
            if mode != self.rgb_mode:
                target = target.convert(self.rgb_mode)
            assert_image_similar(image, target, epsilon)

    def test_write_rgb(self, tmp_path: Path) -> None:
        """
        Can we write a RGB mode file to webp without error?
        Does it have the bits we expect?
        """

        self._roundtrip(tmp_path, self.rgb_mode, 12.5)

    def test_write_method(self, tmp_path: Path) -> None:
        self._roundtrip(tmp_path, self.rgb_mode, 12.0, {"method": 6})

        buffer_no_args = io.BytesIO()
        hopper().save(buffer_no_args, format="WEBP")

        buffer_method = io.BytesIO()
        hopper().save(buffer_method, format="WEBP", method=6)
        assert buffer_no_args.getbuffer() != buffer_method.getbuffer()

    def test_save_all(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "temp.webp"
        im = Image.new("RGB", (1, 1))
        im2 = Image.new("RGB", (1, 1), "#f00")
        im.save(temp_file, save_all=True, append_images=[im2])

        with Image.open(temp_file) as reloaded:
            assert_image_equal(im, reloaded)

            reloaded.seek(1)
            assert_image_similar(im2, reloaded, 1)

    def test_unsupported_image_mode(self) -> None:
        im = Image.new("1", (1, 1))
        with pytest.raises(ValueError):
            _webp.WebPEncode(im.getim(), False, 0, 0, "", 4, 0, b"", "")

    def test_icc_profile(self, tmp_path: Path) -> None:
        self._roundtrip(tmp_path, self.rgb_mode, 12.5, {"icc_profile": None})
        self._roundtrip(
            tmp_path, self.rgb_mode, 12.5, {"icc_profile": None, "save_all": True}
        )

    def test_write_unsupported_mode_L(self, tmp_path: Path) -> None:
        """
        Saving a black-and-white file to WebP format should work, and be
        similar to the original file.
        """

        self._roundtrip(tmp_path, "L", 10.0)

    def test_write_unsupported_mode_P(self, tmp_path: Path) -> None:
        """
        Saving a palette-based file to WebP format should work, and be
        similar to the original file.
        """

        self._roundtrip(tmp_path, "P", 50.0)

    @pytest.mark.skipif(sys.maxsize <= 2**32, reason="Requires 64-bit system")
    def test_write_encoding_error_message(self, tmp_path: Path) -> None:
        im = Image.new("RGB", (15000, 15000))
        with pytest.raises(ValueError, match="encoding error 6"):
            im.save(tmp_path / "temp.webp", method=0)

    @pytest.mark.skipif(sys.maxsize <= 2**32, reason="Requires 64-bit system")
    def test_write_encoding_error_bad_dimension(self, tmp_path: Path) -> None:
        im = Image.new("L", (16384, 16384))
        with pytest.raises(ValueError) as e:
            im.save(tmp_path / "temp.webp")
        assert (
            str(e.value)
            == "encoding error 5: Image size exceeds WebP limit of 16383 pixels"
        )

    def test_WebPEncode_with_invalid_args(self) -> None:
        """
        Calling encoder functions with no arguments should result in an error.
        """
        with pytest.raises(TypeError):
            _webp.WebPAnimEncoder()
        with pytest.raises(TypeError):
            _webp.WebPEncode()

    def test_WebPAnimDecoder_with_invalid_args(self) -> None:
        """
        Calling decoder functions with no arguments should result in an error.
        """
        with pytest.raises(TypeError):
            _webp.WebPAnimDecoder()

    def test_no_resource_warning(self, tmp_path: Path) -> None:
        file_path = "Tests/images/hopper.webp"
        with Image.open(file_path) as image:
            with warnings.catch_warnings(action="error"):
                image.save(tmp_path / "temp.webp")

    def test_file_pointer_could_be_reused(self) -> None:
        file_path = "Tests/images/hopper.webp"
        with open(file_path, "rb") as blob:
            Image.open(blob).load()
            Image.open(blob).load()

    @pytest.mark.parametrize(
        "background",
        (0, (0,), (-1, 0, 1, 2), (253, 254, 255, 256)),
    )
    def test_invalid_background(
        self, background: int | tuple[int, ...], tmp_path: Path
    ) -> None:
        temp_file = tmp_path / "temp.webp"
        im = hopper()
        with pytest.raises(OSError):
            im.save(temp_file, save_all=True, append_images=[im], background=background)

    def test_background_from_gif(self, tmp_path: Path) -> None:
        out_webp = tmp_path / "temp.webp"

        # Save L mode GIF with background
        with Image.open("Tests/images/no_palette_with_background.gif") as im:
            im.save(out_webp, save_all=True)

        # Save P mode GIF with background
        with Image.open("Tests/images/chi.gif") as im:
            original_value = im.convert("RGB").getpixel((1, 1))
            assert isinstance(original_value, tuple)

            # Save as WEBP
            im.save(out_webp, save_all=True)

        # Save as GIF
        out_gif = tmp_path / "temp.gif"
        with Image.open(out_webp) as im:
            im.save(out_gif)

        with Image.open(out_gif) as reread:
            reread_value = reread.convert("RGB").getpixel((1, 1))
        assert isinstance(reread_value, tuple)
        difference = sum(abs(original_value[i] - reread_value[i]) for i in range(3))
        assert difference < 5

    def test_duration(self, tmp_path: Path) -> None:
        out_webp = tmp_path / "temp.webp"

        with Image.open("Tests/images/dispose_bgnd.gif") as im:
            assert im.info["duration"] == 1000
            im.save(out_webp, save_all=True)

        with Image.open(out_webp) as reloaded:
            reloaded.load()
            assert reloaded.info["duration"] == 1000

    def test_roundtrip_rgba_palette(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "temp.webp"
        im = Image.new("RGBA", (1, 1)).convert("P")
        assert im.mode == "P"
        assert im.palette is not None
        assert im.palette.mode == "RGBA"
        im.save(temp_file)

        with Image.open(temp_file) as im:
            assert im.getpixel((0, 0)) == (0, 0, 0, 0)


def _webp_chunks(data: bytes) -> list[list[bytes | bytearray]]:
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WEBP"
    chunks: list[list[bytes | bytearray]] = []
    offset = 12
    while offset + 8 <= len(data):
        fourcc = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        chunks.append([fourcc, bytearray(data[offset + 8 : offset + 8 + size])])
        offset += 8 + size + (size & 1)
    return chunks


def _write_webp_chunks(chunks: list[list[bytes | bytearray]]) -> bytes:
    body = b""
    for fourcc, payload in chunks:
        payload = bytes(payload)
        body += (
            fourcc
            + len(payload).to_bytes(4, "little")
            + payload
            + (b"\0" if len(payload) & 1 else b"")
        )
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WEBP" + body


@skip_unless_feature("webp")
class TestWebpHeaderSizeCheck:
    def _patch(self, path: str, chunk_name: bytes, mutate) -> bytes:
        with open(path, "rb") as f:
            chunks = _webp_chunks(f.read())
        for chunk in chunks:
            if chunk[0] == chunk_name:
                mutate(chunk[1])
        return _write_webp_chunks(chunks)

    @pytest.mark.parametrize(
        "path",
        [
            "Tests/images/transparent.webp",  # VP8X static
            "Tests/images/iss634.webp",  # VP8X animated
        ],
    )
    def test_oversized_vp8x_canvas_rejected_before_decoder(
        self, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mutate(payload: bytearray) -> None:
            # Canvas dimensions are 24-bit values minus one
            payload[4:10] = b"\xff\xff\xff\xff\xff\xff"

        data = self._patch(path, b"VP8X", mutate)

        constructed = []
        original = _webp.WebPAnimDecoder
        monkeypatch.setattr(
            _webp,
            "WebPAnimDecoder",
            lambda *args: constructed.append(1) or original(*args),
        )

        with pytest.raises(Image.DecompressionBombError):
            Image.open(io.BytesIO(data))
        assert constructed == []

    def test_oversized_animated_frame_rejected_before_decoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mutate(payload: bytearray) -> None:
            # ANMF layout: x(3) y(3) width(3) height(3) duration(3) flags(1)
            payload[6:12] = b"\xff\xff\xff\xff\xff\xff"

        data = self._patch("Tests/images/iss634.webp", b"ANMF", mutate)

        constructed = []
        original = _webp.WebPAnimDecoder
        monkeypatch.setattr(
            _webp,
            "WebPAnimDecoder",
            lambda *args: constructed.append(1) or original(*args),
        )

        with pytest.raises(Image.DecompressionBombError):
            Image.open(io.BytesIO(data))
        assert constructed == []

    def test_oversized_vp8_bitstream(self) -> None:
        def mutate(payload: bytearray) -> None:
            # Width/height are 16-bit values after the 3-byte frame tag and
            # the 3-byte keyframe sync code
            payload[6:10] = b"\xff\xff\xff\xff"

        data = self._patch("Tests/images/hopper.webp", b"VP8 ", mutate)
        with pytest.raises(Image.DecompressionBombError):
            Image.open(io.BytesIO(data))

    def test_oversized_vp8l_bitstream(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "lossless.webp"
        with Image.open("Tests/images/hopper.webp") as im:
            im.save(temp_file, lossless=True)
        with open(temp_file, "rb") as f:
            chunks = _webp_chunks(f.read())

        for chunk in chunks:
            if chunk[0] == b"VP8L":
                payload = chunk[1]
                assert payload[0] == 0x2F
                # 14-bit width-1 then 14-bit height-1 packed little-endian
                packed = 0x3FFF | (0x3FFF << 14)
                payload[1] = packed & 0xFF
                payload[2] = (packed >> 8) & 0xFF
                payload[3] = (packed >> 16) & 0xFF
                payload[4] = (packed >> 24) & 0xF
        data = _write_webp_chunks(chunks)

        with pytest.raises(Image.DecompressionBombError):
            Image.open(io.BytesIO(data))

    @pytest.mark.parametrize(
        "path, chunk_name",
        [
            ("Tests/images/transparent.webp", b"VP8X"),
            ("Tests/images/hopper.webp", b"VP8 "),
        ],
    )
    def test_truncated_header_raises_syntax_error(
        self, path: str, chunk_name: bytes
    ) -> None:
        with open(path, "rb") as f:
            chunks = _webp_chunks(f.read())
        for chunk in chunks:
            if chunk[0] == chunk_name:
                chunk[1] = chunk[1][:4]
        data = _write_webp_chunks(chunks)

        # Image.open wraps the plugin SyntaxError as UnidentifiedImageError
        with pytest.raises(Image.UnidentifiedImageError):
            Image.open(io.BytesIO(data))
        with pytest.raises(SyntaxError):
            WebPImagePlugin.WebPImageFile(io.BytesIO(data))

    def test_zero_width_raises_syntax_error(self) -> None:
        def mutate(payload: bytearray) -> None:
            payload[6:8] = b"\x00\x00"  # width = 0

        data = self._patch("Tests/images/hopper.webp", b"VP8 ", mutate)
        with pytest.raises(SyntaxError):
            WebPImagePlugin.WebPImageFile(io.BytesIO(data))

    def test_bomb_warning_emitted_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limit = 100 * 100
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", limit)

        def mutate(payload: bytearray) -> None:
            # 101 x 101 = 10201 pixels: above the limit but below 2x, so a
            # warning rather than an error is expected
            payload[6:8] = (101).to_bytes(2, "little")
            payload[8:10] = (101).to_bytes(2, "little")

        data = self._patch("Tests/images/hopper.webp", b"VP8 ", mutate)
        # The default warning filter deduplicates identical messages, so the
        # threshold warning fires only once even across repeated opens
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default", Image.DecompressionBombWarning)
            for _ in range(2):
                with contextlib.suppress(Exception):
                    Image.open(io.BytesIO(data))
        bomb_warnings = [
            w for w in caught if issubclass(w.category, Image.DecompressionBombWarning)
        ]
        assert len(bomb_warnings) == 1

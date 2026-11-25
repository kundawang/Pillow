from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from PIL import Image

from .helper import assert_image_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("bits", (1, 2, 4, 8))
@pytest.mark.parametrize("width", (1, 3, 5, 9, 17))
def test_save_l_bits_quantized_rows(bits: int, width: int, tmp_path: Path) -> None:
    height = 5
    im = Image.new("L", (width, height))
    original = [(x * 17 + y * 53) % 256 for y in range(height) for x in range(width)]
    im.putdata(original)

    out = tmp_path / "rows.png"
    im.save(out, bits=bits)

    data = out.read_bytes()
    assert data[24] == bits
    assert data[25] == 0  # grayscale

    max_sample = (1 << bits) - 1
    expansion = 255 // max_sample
    expected = [((value * max_sample + 127) // 255) * expansion for value in original]

    with Image.open(out) as reloaded:
        assert reloaded.size == (width, height)
        actual = list(reloaded.convert("L").tobytes())

    assert actual == expected


@pytest.mark.parametrize("bits", (1, 2, 4, 8))
def test_save_l_bits_gradient_and_partial_row(bits: int, tmp_path: Path) -> None:
    # Width 13 is not a multiple of 8, 4, or 2.
    im = Image.linear_gradient("L").resize((13, 17))
    original = list(im.tobytes())

    out = tmp_path / "gradient.png"
    im.save(out, bits=bits)

    assert out.read_bytes()[24] == bits
    with Image.open(out) as reloaded:
        actual = list(reloaded.convert("L").tobytes())

    max_sample = (1 << bits) - 1
    expansion = 255 // max_sample
    expected = [((value * max_sample + 127) // 255) * expansion for value in original]
    assert actual == expected


@pytest.mark.parametrize("mode", ("L", "P"))
@pytest.mark.parametrize("bits", (0, 3, 5, 16, -1, "4", None))
def test_invalid_png_bits(mode: str, bits: object, tmp_path: Path) -> None:
    im = Image.new(mode, (1, 1))
    if mode == "P":
        im.putpalette((0, 0, 0, 255, 255, 255))

    with pytest.raises(ValueError, match="bits"):
        im.save(tmp_path / "invalid.png", bits=bits)  # type: ignore[arg-type]


@pytest.mark.parametrize("bits", (1, 2, 4, 8))
def test_save_p_bits_transparency_compatible(bits: int, tmp_path: Path) -> None:
    im = Image.new("P", (2, 2))
    im.putpalette((0, 0, 0, 255, 255, 255))
    im.info["transparency"] = 1

    out = tmp_path / "p.png"
    im.save(out, bits=bits)

    with Image.open(out) as reloaded:
        assert reloaded.info["transparency"] == 1
        assert_image_equal(reloaded, im)

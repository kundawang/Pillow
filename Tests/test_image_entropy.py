from __future__ import annotations

import pytest

from PIL import Image

from .helper import hopper


def test_entropy() -> None:
    def entropy(mode: str) -> float:
        return hopper(mode).entropy()

    assert entropy("1") == pytest.approx(0.9138803254693582)
    for mode in ("L", "I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
        assert entropy(mode) == pytest.approx(7.063008716585465)
    assert entropy("P") == pytest.approx(5.082506854662517)
    assert entropy("RGB") == pytest.approx(8.821286587714319)
    assert entropy("RGBA") == pytest.approx(7.42724306524488)
    assert entropy("CMYK") == pytest.approx(7.4272430652448795)
    assert entropy("YCbCr") == pytest.approx(7.698360534903628)


def test_i16_entropy() -> None:
    values = (0, 255, 256, 65535)
    base = Image.new("I", (len(values), 1))
    base.putdata(values)
    entropies = []
    for mode in ("I;16", "I;16L", "I;16B", "I;16N"):
        im = base.convert(mode)
        # Three values map to the first bin and 255 to the last bin
        assert im.entropy() == pytest.approx(0.8112781244591328)
        entropies.append(im.entropy())
    assert entropies == [entropies[0]] * len(entropies)

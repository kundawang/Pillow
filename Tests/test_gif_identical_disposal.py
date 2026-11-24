from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from PIL import GifImagePlugin, Image, ImageDraw, ImageSequence

from .helper import assert_image_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("container", [list, tuple])
def test_identical_frames_to_single_frame_disposal(
    container: Any, tmp_path: Path
) -> None:
    frame = Image.new("L", (4, 4), 128)
    out = tmp_path / "identical-disposal.gif"

    frame.save(
        out,
        save_all=True,
        append_images=[frame.copy(), frame.copy()],
        duration=container([100, 200, 300]),
        disposal=container([2, 2, 2]),
    )

    with Image.open(out) as reread:
        assert isinstance(reread, GifImagePlugin.GifImageFile)
        assert reread.n_frames == 1
        assert reread.info["duration"] == 600
        assert reread.disposal_method == 2
        assert reread.convert("L").getpixel((2, 2)) == 128


def test_identical_frames_scalar_disposal_still_works(tmp_path: Path) -> None:
    frame = Image.new("L", (4, 4), 128)
    out = tmp_path / "identical-scalar-disposal.gif"

    frame.save(
        out,
        save_all=True,
        append_images=[frame.copy(), frame.copy()],
        duration=200,
        disposal=2,
    )

    with Image.open(out) as reread:
        assert isinstance(reread, GifImagePlugin.GifImageFile)
        assert reread.n_frames == 1
        assert reread.info["duration"] == 600
        assert reread.disposal_method == 2


@pytest.mark.parametrize("container", [list, tuple])
def test_merged_animation_metadata_and_final_pixels(
    container: Any, tmp_path: Path
) -> None:
    def make_frame(
        background: int, box: tuple[int, int, int, int] | None = None
    ) -> Image.Image:
        im = Image.new("L", (8, 8), background)
        if box is not None:
            ImageDraw.Draw(im).rectangle(box, fill=255)
        return im

    first = make_frame(0)
    middle = make_frame(85, (2, 2, 5, 5))
    last = make_frame(170, (1, 1, 6, 6))
    frames = [first, first.copy(), middle, last, last.copy()]

    out = tmp_path / "merged-animation.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=container([100, 200, 300, 400, 500]),
        disposal=container([0, 0, 2, 1, 1]),
    )

    with Image.open(out) as reread:
        assert isinstance(reread, GifImagePlugin.GifImageFile)
        durations: list[int] = []
        disposals: list[int] = []
        for gif_frame in ImageSequence.Iterator(reread):
            assert isinstance(gif_frame, GifImagePlugin.GifImageFile)
            durations.append(gif_frame.info["duration"])
            disposals.append(gif_frame.disposal_method)

        assert reread.n_frames == 3
        assert durations == [300, 300, 900]
        assert disposals == [0, 2, 1]

        reread.seek(2)
        assert_image_equal(reread.convert("L"), last)

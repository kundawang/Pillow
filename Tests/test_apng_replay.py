from __future__ import annotations

import pytest

from PIL import Image, ImageSequence

from .helper import assert_image_equal

REGION_DISPOSAL_IMAGES = (
    "Tests/images/apng/dispose_op_background_region.png",
    "Tests/images/apng/dispose_op_previous_frame.png",
)


@pytest.mark.parametrize("path", REGION_DISPOSAL_IMAGES)
def test_seek_zero_one_repeatedly(path: str) -> None:
    # Rewinding must restore the sequence number, chunk queue, frame index,
    # canvas, and disposal/blend state. These fixtures use sub-canvas regions
    # and non-default disposal operations.
    expected: dict[int, bytes] = {}

    with Image.open(path) as im:
        for _ in range(3):
            for frame in (0, 1, 0, 1):
                im.seek(frame)
                assert im.tell() == frame
                pixels = im.tobytes()

                if frame not in expected:
                    expected[frame] = pixels
                assert pixels == expected[frame]

        # Seeking to the current frame is a no-op and must not change state.
        for frame in (0, 1):
            im.seek(frame)
            im.seek(frame)
            assert im.tell() == frame
            assert im.tobytes() == expected[frame]


def test_image_sequence_replay() -> None:
    path = "Tests/images/apng/dispose_op_background_region.png"

    with Image.open(path) as im:
        first_pass = [frame.copy() for frame in ImageSequence.Iterator(im)]
        second_pass = [frame.copy() for frame in ImageSequence.Iterator(im)]

    assert len(first_pass) == len(second_pass)
    for first, second in zip(first_pass, second_pass):
        assert_image_equal(first, second)

    with Image.open(path) as im:
        expected_frame_0 = im.copy()
        im.seek(1)
        expected_frame_1 = im.copy()

        im.seek(0)
        assert im.tell() == 0
        assert_image_equal(im, expected_frame_0)

        im.seek(1)
        assert im.tell() == 1
        assert_image_equal(im, expected_frame_1)

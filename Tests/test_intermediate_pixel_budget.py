from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from PIL import Image

# The workers assign this to Image.MAX_IMAGE_PIXELS.  The deliberately small
# budget lets tiny images and kernels represent sizes far above the limit
# without asking the operating system for a large amount of memory.
LOW_BUDGET = 64
ERROR_THRESHOLD = 2 * LOW_BUDGET

# In an editable checkout this is ``<checkout>/src``; in a normal install it
# is the directory containing the ``PIL`` package.
PIL_ROOT = Path(Image.__file__).resolve().parents[1]

WORKER = r"""
import json
import sys

# Isolated mode ignores PYTHONPATH, so explicitly use the copy under test.
sys.path.insert(0, sys.argv[1])

from PIL import Image, ImageFilter, ImageOps

case = sys.argv[2]
Image.MAX_IMAGE_PIXELS = int(sys.argv[3])


def emit(payload: dict[str, object], returncode: int = 0) -> None:
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(returncode)


def rejected(exc: Image.DecompressionBombError, size: tuple[int, int]) -> None:
    emit(
        {
            "status": "rejected",
            "error": type(exc).__name__,
            "message": str(exc),
            "size": list(size),
        }
    )


class AllocationAttempt(RuntimeError):
    pass


if case == "cover-small":
    source = (1, 64)
    requested = (64, 1)
    intermediate = (64, 4096)
    try:
        result = ImageOps.cover(Image.new("L", source), requested)
    except Image.DecompressionBombError as exc:
        rejected(exc, intermediate)
    emit({"status": "allowed", "size": list(result.size)}, 10)

elif case == "rank-small":
    source = (2, 2)
    kernel_size = 31
    margin = kernel_size // 2
    intermediate = (
        source[0] + 2 * margin,
        source[1] + 2 * margin,
    )
    try:
        result = Image.new("L", source).filter(
            ImageFilter.RankFilter(kernel_size, 0)
        )
    except Image.DecompressionBombError as exc:
        rejected(exc, intermediate)
    emit({"status": "allowed", "size": list(result.size)}, 10)

elif case == "cover-allocation-barrier":
    intermediate = (256, 25_600_000)

    def blocked_resize(
        self: Image.Image,
        size: tuple[int, int],
        resample: int | None = None,
        box: tuple[float, float, float, float] | None = None,
        reducing_gap: float | None = None,
    ) -> Image.Image:
        # The baseline reaches this method.  Do not call the real resize,
        # because its nominal intermediate size is intentionally enormous.
        raise AllocationAttempt(tuple(size))

    Image.Image.resize = blocked_resize  # type: ignore[method-assign]

    try:
        ImageOps.cover(Image.new("L", (1, 100_000)), (256, 256))
    except Image.DecompressionBombError as exc:
        rejected(exc, intermediate)
    except AllocationAttempt as exc:
        emit({"status": "allocation_attempt", "size": list(exc.args[0])}, 11)
    emit({"status": "completed_without_check"}, 12)

elif case == "rank-allocation-barrier":
    kernel_size = 23169  # largest odd size accepted by __init__'s C guard
    margin = kernel_size // 2
    intermediate = (1 + 2 * margin, 1 + 2 * margin)

    class BlockedCore:
        mode = "L"
        size = (1, 1)

        def expand(self, margin: int) -> Any:
            # The baseline reaches this method.  Returning/allocating nothing
            # keeps the enormous nominal size purely arithmetic.
            raise AllocationAttempt((1 + 2 * margin, 1 + 2 * margin))

    try:
        rank_filter = ImageFilter.RankFilter(kernel_size, 0)
        rank_filter.filter(BlockedCore())  # type: ignore[arg-type]
    except Image.DecompressionBombError as exc:
        rejected(exc, intermediate)
    except AllocationAttempt as exc:
        emit({"status": "allocation_attempt", "size": list(exc.args[0])}, 11)
    emit({"status": "completed_without_check"}, 12)

elif case == "controls":
    import warnings

    operations = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        im = Image.new("L", (4, 4))
        assert im.resize((8, 8)).size == (8, 8)
        operations.append("upscale-to-budget")

        # resize() honours the caller's requested output size; this gate is
        # concerned with unrequested aspect-ratio/window expansion.
        assert im.resize((128, 128)).size == (128, 128)
        operations.append("normal-resize-over-low-budget")

        assert Image.new("L", (8, 8)).resize((4, 4)).size == (4, 4)
        operations.append("downscale")

        assert ImageOps.cover(Image.new("L", (2, 2)), (8, 8)).size == (8, 8)
        operations.append("same-ratio-square")

        # Same ratio means cover() performs no extra expansion, even when the
        # explicitly requested size is above the tiny test budget.
        assert ImageOps.cover(Image.new("L", (4, 4)), (128, 128)).size == (128, 128)
        operations.append("same-ratio-request-over-low-budget")

        assert ImageOps.cover(Image.new("L", (2, 8)), (4, 16)).size == (4, 16)
        operations.append("same-ratio-rectangle")

        for kernel_size in (1, 3, 5):
            filtered = Image.new("L", (4, 4)).filter(
                ImageFilter.RankFilter(kernel_size, 0)
            )
            assert filtered.size == (4, 4)
            operations.append(f"rank-{kernel_size}")

    emit({"status": "ok", "operations": operations, "budget": Image.MAX_IMAGE_PIXELS})

else:
    emit({"status": "unknown-case", "case": case}, 64)
"""


def cover_intermediate_size(
    source: tuple[int, int], requested: tuple[int, int]
) -> tuple[int, int]:
    """Return the size ``ImageOps.cover`` asks ``resize`` to allocate."""
    source_width, source_height = source
    requested_width, requested_height = requested

    # Exact ratio comparison, followed by the same two expansions as cover().
    if source_width * requested_height == source_height * requested_width:
        return requested
    if source_width / source_height < requested_width / requested_height:
        new_height = round(source_height / source_width * requested_width)
        return requested_width, new_height
    new_width = round(source_width / source_height * requested_height)
    return new_width, requested_height


def rank_intermediate_size(
    source: tuple[int, int], kernel_size: int
) -> tuple[int, int]:
    """Return the size created by ``RankFilter.filter``'s border expansion."""
    margin = kernel_size // 2
    return source[0] + 2 * margin, source[1] + 2 * margin


SECURITY_CASES: dict[str, dict[str, Any]] = {
    "cover-small": {
        "path": "cover",
        "source": (1, 64),
        "requested": (64, 1),
    },
    "rank-small": {
        "path": "rank",
        "source": (2, 2),
        "kernel_size": 31,
    },
    "cover-allocation-barrier": {
        "path": "cover",
        "source": (1, 100_000),
        "requested": (256, 256),
    },
    "rank-allocation-barrier": {
        "path": "rank",
        "source": (1, 1),
        "kernel_size": 23169,
    },
}


def _calculated_size(spec: dict[str, Any]) -> tuple[int, int]:
    if spec["path"] == "cover":
        return cover_intermediate_size(spec["source"], spec["requested"])
    return rank_intermediate_size(spec["source"], spec["kernel_size"])


def _run_worker(case: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            WORKER,
            str(PIL_ROOT),
            case,
            str(LOW_BUDGET),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _worker_failure_details(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"return code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("case", sorted(SECURITY_CASES))
def test_intermediate_sizes_obey_pixel_budget(case: str) -> None:
    spec = SECURITY_CASES[case]
    intermediate = _calculated_size(spec)
    pixels = intermediate[0] * intermediate[1]

    # Gate precondition: the independently calculated intermediate image is
    # in the hard-error band of the low MAX_IMAGE_PIXELS budget.
    assert pixels > ERROR_THRESHOLD, (intermediate, pixels)

    result = _run_worker(case)
    assert result.returncode == 0, _worker_failure_details(result)

    payload = json.loads(result.stdout)
    assert payload["status"] == "rejected", payload
    assert payload["error"] == "DecompressionBombError"
    assert tuple(payload["size"]) == intermediate
    assert f"exceeds limit of {ERROR_THRESHOLD} pixels" in payload["message"]


def test_normal_operations_and_legal_filters_are_not_rejected() -> None:
    result = _run_worker("controls")

    assert result.returncode == 0, _worker_failure_details(result)
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok", payload
    assert payload["budget"] == LOW_BUDGET
    assert payload["operations"] == [
        "upscale-to-budget",
        "normal-resize-over-low-budget",
        "downscale",
        "same-ratio-square",
        "same-ratio-request-over-low-budget",
        "same-ratio-rectangle",
        "rank-1",
        "rank-3",
        "rank-5",
    ]

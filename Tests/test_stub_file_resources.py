from __future__ import annotations

import builtins
import io
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from PIL import (
    BufrStubImagePlugin,
    GribStubImagePlugin,
    Hdf5StubImagePlugin,
    Image,
    ImageFile,
    WmfImagePlugin,
)


class ControlledHandlerError(RuntimeError):
    pass


class TrackingStream(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


class RecordingHandler(ImageFile.StubHandler):
    def __init__(
        self, *, fail: bool = False, lazy: bool = False, color: tuple[int, int, int]
    ) -> None:
        self.fail = fail
        self.lazy = lazy
        self.color = color
        self.open_count = 0
        self.load_count = 0
        self.load_fps: list[object] = []
        self.lazy_read: bytes | None = None

    def open(self, im: ImageFile.StubImageFile) -> None:
        self.open_count += 1
        im._size = (3, 2)

    def load(self, im: ImageFile.StubImageFile) -> Image.Image:
        self.load_count += 1
        self.load_fps.append(im.fp)
        im._size = (3, 2)
        if self.fail:
            msg = "controlled handler failure"
            raise ControlledHandlerError(msg)

        if self.lazy:
            source = im
            color = self.color
            handler = self

            class LazyImage(Image.Image):
                def load(self) -> Image.core.PixelAccess | None:
                    if self._im is not None:
                        return Image.Image.load(self)
                    assert source.fp is not None
                    source.fp.seek(0)
                    handler.lazy_read = source.fp.read(8)
                    loaded = Image.new("RGB", source.size, color)
                    self.__dict__.update(loaded.__dict__)
                    return Image.Image.load(self)

            image = LazyImage()
            image._mode = "RGB"
            image._size = source.size
            return image

        return Image.new("RGB", im.size, self.color)


STUB_CASES = [
    pytest.param(
        BufrStubImagePlugin.BufrStubImageFile,
        "Tests/images/gfs.t06z.rassda.tm00.bufr_d",
        (11, 22, 33),
        id="bufr",
    ),
    pytest.param(
        GribStubImagePlugin.GribStubImageFile,
        "Tests/images/WAlaska.wind.7days.grb",
        (44, 55, 66),
        id="grib",
    ),
    pytest.param(
        Hdf5StubImagePlugin.HDF5StubImageFile,
        "Tests/images/hdf5.h5",
        (77, 88, 99),
        id="hdf5",
    ),
    pytest.param(
        WmfImagePlugin.WmfStubImageFile,
        "Tests/images/drawing.emf",
        (111, 122, 133),
        id="wmf",
    ),
]


class OpenRecord:
    def __init__(self, fp: Any) -> None:
        self.fp = fp
        self.close_count = 0


def track_path_open(monkeypatch: pytest.MonkeyPatch, path: str) -> list[OpenRecord]:
    records: list[OpenRecord] = []
    real_open = builtins.open

    def tracking_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        fp = real_open(file, *args, **kwargs)
        if os.fspath(file) == path:
            record = OpenRecord(fp)
            real_close = fp.close

            def tracking_close() -> Any:
                record.close_count += 1
                return real_close()

            fp.close = tracking_close
            records.append(record)
        return fp

    monkeypatch.setattr(builtins, "open", tracking_open)
    return records


@pytest.mark.parametrize("stub_cls, path, color", STUB_CASES)
def test_owned_file_closed_after_successful_load(
    stub_cls: type[ImageFile.StubImageFile],
    path: str,
    color: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = RecordingHandler(color=color)
    monkeypatch.setattr(stub_cls, "_handler", handler)
    records = track_path_open(monkeypatch, path)

    with Image.open(path) as im:
        assert len(records) == 1
        record = records[0]

        im.load()

        assert handler.open_count == 1
        assert handler.load_count == 1
        assert record.close_count == 1
        assert record.fp.closed
        assert im.fp is None
        assert im.getpixel((0, 0)) == color

        # A loaded image does not invoke the stub handler again or close twice.
        im.load()
        assert handler.load_count == 1
        assert record.close_count == 1

    assert record.close_count == 1


@pytest.mark.parametrize("stub_cls, path, color", STUB_CASES)
def test_owned_file_closed_after_handler_error_and_retry(
    stub_cls: type[ImageFile.StubImageFile],
    path: str,
    color: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = RecordingHandler(fail=True, color=color)
    monkeypatch.setattr(stub_cls, "_handler", handler)
    records = track_path_open(monkeypatch, path)

    im = Image.open(path)
    assert len(records) == 1
    record = records[0]
    fp = im.fp

    with pytest.raises(ControlledHandlerError, match="controlled handler failure"):
        im.load()

    assert handler.load_count == 1
    assert record.close_count == 1
    assert fp is not None
    assert fp.closed
    assert im.fp is None

    with pytest.raises(ControlledHandlerError, match="controlled handler failure"):
        im.load()

    assert handler.load_count == 2
    assert record.close_count == 1

    # Closing the image afterwards must not attempt another tracked close.
    im.close()
    assert record.close_count == 1


@pytest.mark.parametrize("stub_cls, path, color", STUB_CASES)
@pytest.mark.parametrize("fail", [False, True])
def test_caller_owned_stream_stays_open(
    stub_cls: type[ImageFile.StubImageFile],
    path: str,
    color: tuple[int, int, int],
    fail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = RecordingHandler(fail=fail, color=color)
    monkeypatch.setattr(stub_cls, "_handler", handler)
    stream = TrackingStream(Path(path).read_bytes())

    with Image.open(stream) as im:
        if fail:
            with pytest.raises(ControlledHandlerError):
                im.load()
        else:
            im.load()
            im.load()

        assert handler.load_count == 1
        assert stream.close_count == 0
        assert not stream.closed

    assert stream.close_count == 0
    assert not stream.closed

    stream.close()
    assert stream.close_count == 1


@pytest.mark.parametrize("stub_cls, path, color", STUB_CASES)
def test_handler_can_be_registered_after_failed_first_load(
    stub_cls: type[ImageFile.StubImageFile],
    path: str,
    color: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stub_cls, "_handler", None)

    with Image.open(path) as im:
        fp = im.fp
        with pytest.raises(OSError, match="cannot find loader"):
            im.load()

        assert fp is not None
        assert not fp.closed

        handler = RecordingHandler(color=color)
        monkeypatch.setattr(stub_cls, "_handler", handler)
        im.load()

        assert handler.load_count == 1
        assert im.getpixel((0, 0)) == color
        assert fp.closed


@pytest.mark.parametrize("stub_cls, path, color", STUB_CASES)
def test_lazy_returned_image_can_read_source_before_close(
    stub_cls: type[ImageFile.StubImageFile],
    path: str,
    color: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = RecordingHandler(lazy=True, color=color)
    monkeypatch.setattr(stub_cls, "_handler", handler)

    with Image.open(path) as im:
        fp = im.fp
        im.load()

        assert handler.lazy_read == Path(path).read_bytes()[:8]
        assert im.getpixel((0, 0)) == color
        assert fp is not None
        assert fp.closed


def _linux_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="requires /proc/self/fd"
)
@pytest.mark.parametrize("stub_cls, path, color", STUB_CASES)
@pytest.mark.parametrize("fail", [False, True])
def test_repeated_imports_do_not_leak_file_descriptors(
    stub_cls: type[ImageFile.StubImageFile],
    path: str,
    color: tuple[int, int, int],
    fail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stub_cls, "_handler", RecordingHandler(fail=fail, color=color))

    def load_once() -> None:
        im = Image.open(path)
        if fail:
            with pytest.raises(ControlledHandlerError):
                im.load()
        else:
            im.load()

    for _ in range(5):
        load_once()

    baseline = _linux_fd_count()
    for _ in range(30):
        load_once()

    assert _linux_fd_count() == baseline

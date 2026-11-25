from __future__ import annotations

import pytest

from PIL import Image


def readonly_l_buffer() -> tuple[bytes, Image.Image]:
    data = bytes(4)
    im = Image.frombuffer("L", (2, 2), data, "raw", "L", 0, 1)
    return data, im


def writable_l_buffer() -> tuple[bytearray, Image.Image]:
    data = bytearray(4)
    im = Image.frombuffer("L", (2, 2), data, "raw", "L", 0, 1)
    return data, im


@pytest.mark.parametrize(
    "buffer_factory",
    [
        lambda data: data,
        memoryview,
    ],
)
def test_frombytes_rejects_readonly_buffer(buffer_factory) -> None:
    raw = bytes(4)
    buffer = buffer_factory(raw)
    im = Image.frombuffer("L", (2, 2), buffer, "raw", "L", 0, 1)

    assert im.readonly
    with pytest.raises(ValueError, match="image is readonly"):
        im.frombytes(b"\x01\x02\x03\x04")

    assert raw == bytes(4)


def test_decoder_setimage_rejects_readonly_buffer() -> None:
    data, im = readonly_l_buffer()
    decoder = Image._getdecoder("L", "raw", ("L",))

    with pytest.raises(ValueError, match="image is readonly"):
        decoder.setimage(im.im, (0, 0, 2, 2))

    # Cleaning up after a rejected target must remain safe.
    decoder.cleanup()
    assert data == bytes(4)


def test_writable_buffer_frombytes_updates_in_place() -> None:
    data, im = writable_l_buffer()

    assert not im.readonly
    im.frombytes(b"\x01\x02\x03\x04")

    assert bytes(data) == b"\x01\x02\x03\x04"
    assert im.tobytes() == b"\x01\x02\x03\x04"


def test_writable_buffer_decoder_updates_in_place() -> None:
    data, im = writable_l_buffer()
    decoder = Image._getdecoder("L", "raw", ("L",))
    decoder.setimage(im.im, (0, 0, 2, 2))

    assert decoder.decode(b"\x01\x02\x03\x04") == (-1, 0)
    assert bytes(data) == b"\x01\x02\x03\x04"


def test_imagecms_apply_rejects_readonly_output() -> None:
    pytest.importorskip("PIL._imagingcms")
    from PIL import ImageCms

    transform = ImageCms.buildTransform(
        ImageCms.createProfile("sRGB"),
        ImageCms.createProfile("sRGB"),
        "RGBA",
        "RGBA",
    )
    source = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
    data = bytes(4)
    output = Image.frombuffer("RGBA", (1, 1), data, "raw", "RGBA", 0, 1)

    with pytest.raises(ValueError, match="image is readonly"):
        transform.apply(source, output)
    assert data == bytes(4)

    with pytest.raises(ValueError, match="image is readonly"):
        transform.transform.apply(source.getim(), output.getim())
    assert data == bytes(4)

    with pytest.raises(ValueError, match="image is readonly"):
        transform.apply_in_place(output)
    assert data == bytes(4)


def test_imagecms_apply_updates_writable_output() -> None:
    pytest.importorskip("PIL._imagingcms")
    from PIL import ImageCms

    transform = ImageCms.buildTransform(
        ImageCms.createProfile("sRGB"),
        ImageCms.createProfile("sRGB"),
        "RGBA",
        "RGBA",
    )
    source = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
    data = bytearray(4)
    output = Image.frombuffer("RGBA", (1, 1), data, "raw", "RGBA", 0, 1)

    assert not output.readonly
    result = transform.apply(source, output)

    assert result is output
    assert tuple(data) == (255, 0, 0, 255)
    assert output.getpixel((0, 0)) == (255, 0, 0, 255)

"""Build malformed PNG/APNG samples by splicing chunks in memory.

Outputs go to ../samples (a separate directory from this script).
Run with: python3 -I work/make_samples.py
"""

from __future__ import annotations

import struct
import zlib
from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "samples"
OUT.mkdir(exist_ok=True)

MAGIC = PngImagePlugin._MAGIC


def parse_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Return list of (cid, full_chunk_bytes_including_length_and_crc)."""
    assert data[:8] == MAGIC
    chunks = []
    i = 8
    while i < len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        cid = data[i + 4 : i + 8]
        raw = data[i : i + 12 + length]  # length field + cid + data + crc
        chunks.append((cid, raw))
        i += 12 + length
    return chunks


def build(chunks: list[tuple[bytes, bytes]]) -> bytes:
    return MAGIC + b"".join(raw for _, raw in chunks)


def putchunk(cid: bytes, data: bytes) -> tuple[bytes, bytes]:
    fp = BytesIO()
    PngImagePlugin.putchunk(fp, cid, data)
    return cid, fp.getvalue()


def make_valid_apng() -> list[tuple[bytes, bytes]]:
    red = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    green = Image.new("RGBA", (4, 4), (0, 255, 0, 255))
    buf = BytesIO()
    red.save(buf, "PNG", save_all=True, append_images=[green])
    return parse_chunks(buf.getvalue())


def ihdr(width: int, height: int) -> tuple[bytes, bytes]:
    return putchunk(
        b"IHDR",
        struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00",
    )


def write(name: str, data: bytes) -> None:
    (OUT / name).write_bytes(data)
    print(f"wrote {name} ({len(data)} bytes)")


def main() -> None:
    # --- 1. plain PNG with a second IHDR spliced before IDAT ---
    plain = Image.new("RGB", (4, 4), (10, 20, 30))
    buf = BytesIO()
    plain.save(buf, "PNG")
    chunks = parse_chunks(buf.getvalue())
    spliced = []
    for cid, raw in chunks:
        spliced.append((cid, raw))
        if cid == b"IHDR":
            spliced.append(ihdr(16, 16))  # duplicate IHDR, bigger canvas
    write("png_duplicate_ihdr.png", build(spliced))

    # same, but the duplicate claims a huge canvas (bomb-sized)
    spliced = []
    for cid, raw in chunks:
        spliced.append((cid, raw))
        if cid == b"IHDR":
            spliced.append(ihdr(2**31 - 1, 1))
    write("png_duplicate_ihdr_huge.png", build(spliced))

    # --- 2. valid 2-frame APNG, then splice a second IHDR after frame 0 ---
    apng = make_valid_apng()
    print("valid apng chunks:", [c for c, _ in apng])

    spliced = []
    for cid, raw in apng:
        spliced.append((cid, raw))
        if cid == b"IDAT":
            # second IHDR appears after the first frame's image data
            spliced.append(ihdr(16, 16))
    write("apng_duplicate_ihdr.png", build(spliced))

    # --- 3. APNG with a bogus frame sequence number in the 2nd fcTL ---
    bad_seq = []
    fctl_seen = 0
    for cid, raw in apng:
        if cid == b"fcTL":
            fctl_seen += 1
            if fctl_seen == 2:
                data = raw[8:-4]
                broken = struct.pack(">I", 99) + data[4:]  # seq=99 instead of 2
                # recompute crc
                import zlib as _zlib

                body = broken
                crc = _zlib.crc32(cid + body) & 0xFFFFFFFF
                raw = raw[:8] + body + struct.pack(">I", crc)
        bad_seq.append((cid, raw))
    write("apng_bad_sequence.png", build(bad_seq))

    # --- 4. control: untouched valid APNG and plain PNG ---
    write("apng_valid.png", build(apng))
    write("png_valid.png", buf.getvalue())


if __name__ == "__main__":
    main()

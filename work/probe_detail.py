"""Detailed probe of error paths after the fix.

Run with: python3 -I work/probe_detail.py
"""

from __future__ import annotations

import traceback
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFile, PngImagePlugin

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def section(title: str) -> None:
    print(f"--- {title} ---")


# 1. plain PNG duplicate IHDR: direct PngImageFile construction
section("plain PNG duplicate IHDR, direct open")
try:
    PngImagePlugin.PngImageFile(SAMPLES / "png_duplicate_ihdr.png")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# 2. plain PNG duplicate IHDR with huge size
section("plain PNG duplicate IHDR huge, direct open")
try:
    PngImagePlugin.PngImageFile(SAMPLES / "png_duplicate_ihdr_huge.png")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# 3. APNG duplicate IHDR: open then seek(1) without loading frame 0
section("APNG duplicate IHDR, seek then load")
try:
    im = PngImagePlugin.PngImageFile(SAMPLES / "apng_duplicate_ihdr.png")
    print(f"  open: size={im.size} n_frames={im.n_frames}")
    im.seek(1)
    print("  seek(1) ok (no error yet)")
    im.load()
    print("  load ok?!")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# 4. APNG duplicate IHDR: load frame 0 first
section("APNG duplicate IHDR, load frame 0 then seek")
try:
    im = PngImagePlugin.PngImageFile(SAMPLES / "apng_duplicate_ihdr.png")
    im.load()
    print("  frame0 load ok?!")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# 5. same with Image.open wrapper
section("APNG duplicate IHDR via Image.open, seek+load")
try:
    with Image.open(SAMPLES / "apng_duplicate_ihdr.png") as im:
        im.seek(1)
        im.load()
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# 6. truncated mode: plain PNG
section("plain PNG duplicate IHDR, LOAD_TRUNCATED_IMAGES")
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    with Image.open(SAMPLES / "png_duplicate_ihdr.png") as im:
        print(f"  size={im.size} mode={im.mode}")
        im.load()
        print(f"  loaded, pixel(0,0)={im.getpixel((0, 0))}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
finally:
    ImageFile.LOAD_TRUNCATED_IMAGES = False

# 7. truncated mode: APNG
section("APNG duplicate IHDR, LOAD_TRUNCATED_IMAGES")
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    with Image.open(SAMPLES / "apng_duplicate_ihdr.png") as im:
        print(f"  size={im.size} n_frames={im.n_frames}")
        im.seek(1)
        im.load()
        print(f"  frame1 pixel(0,0)={im.getpixel((0, 0))}")
except Exception as e:
    traceback.print_exc(limit=3)
    print(f"  {type(e).__name__}: {e}")
finally:
    ImageFile.LOAD_TRUNCATED_IMAGES = False

# 8. valid regression
section("valid APNG regression")
with Image.open(SAMPLES / "apng_valid.png") as im:
    assert im.size == (4, 4)
    assert im.n_frames == 2
    assert im.getpixel((0, 0)) == (255, 0, 0, 255)
    im.seek(1)
    im.load()
    assert im.getpixel((0, 0)) == (0, 255, 0, 255)
    print("  ok")

"""Probe current Pillow behavior on the malformed samples.

Run with: python3 -I work/probe.py
"""

from __future__ import annotations

import traceback
from pathlib import Path

from PIL import Image, PngImagePlugin

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def probe(name: str) -> None:
    print(f"=== {name} ===")
    path = SAMPLES / name
    try:
        with Image.open(path) as im:
            print(f"  open ok: size={im.size} mode={im.mode} n_frames={im.n_frames}")
            print(f"  is_animated={im.is_animated} mimetype={im.get_format_mimetype()}")
            try:
                im.load()
                print(f"  frame 0 load ok, pixel(0,0)={im.getpixel((0, 0))}")
            except Exception:
                print("  frame 0 load failed:")
                traceback.print_exc(limit=2)
            for f in range(1, im.n_frames):
                try:
                    im.seek(f)
                    im.load()
                    print(f"  frame {f} seek+load ok, pixel(0,0)={im.getpixel((0, 0))}")
                except Exception as e:
                    print(f"  frame {f} failed: {type(e).__name__}: {e}")
                    break
    except Exception as e:
        print(f"  open failed: {type(e).__name__}: {e}")
    print()


for name in (
    "png_valid.png",
    "png_duplicate_ihdr.png",
    "png_duplicate_ihdr_huge.png",
    "apng_valid.png",
    "apng_duplicate_ihdr.png",
    "apng_bad_sequence.png",
):
    probe(name)

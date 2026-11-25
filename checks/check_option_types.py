"""
Fixed-sample regression check for the internal option-type refactoring
(dither / quantize method / resampling filter / transform method).

Runs deterministic samples across image modes and boundary parameters and
prints a compact hash summary. The output must be byte-identical before and
after the refactoring: public Python API, exception types, pixel bytes and
serialized results must not change.

Run with the workspace interpreter, e.g.:
    .venv/bin/python checks/check_option_types.py
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import sys
import traceback

from PIL import Image, ImageTransform

# Deterministic source images, built from a fixed seed so the sample is stable.
SEED = 20261125
MODES = ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I", "F", "I;16")
RESAMPLE_FILTERS = [
    Image.Resampling.NEAREST,
    Image.Resampling.BOX,
    Image.Resampling.BILINEAR,
    Image.Resampling.HAMMING,
    Image.Resampling.BICUBIC,
    Image.Resampling.LANCZOS,
    Image.Resampling.MKS2013,
    Image.Resampling.MKS2021,
]
# Boundary-ish output sizes: zero handled separately, 1 is the minimum.
SIZES = (1, 2, 7, 31)
DITHERS = (Image.Dither.NONE, Image.Dither.FLOYDSTEINBERG)
QUANTIZE_METHODS = (
    Image.Quantize.MEDIANCUT,
    Image.Quantize.MAXCOVERAGE,
    Image.Quantize.FASTOCTREE,
)
TRANSFORM_METHODS = (
    Image.Transform.AFFINE,
    Image.Transform.PERSPECTIVE,
    Image.Transform.QUAD,
)


def _stable_seed(mode: str, size: int) -> int:
    # Deterministic across interpreter runs: never use the built-in hash(),
    # which is randomized per process for strings.
    import hashlib

    digest = hashlib.sha256(f"{mode}:{size}".encode()).digest()
    return SEED ^ (int.from_bytes(digest[:4], "little") % 100000)


def make_sample(mode: str, size: int) -> Image.Image:
    rnd = random.Random(_stable_seed(mode, size))
    if mode == "1":
        row_bytes = (size + 7) // 8
        return Image.frombytes(
            "1",
            (size, size),
            bytes(rnd.randrange(256) for _ in range(row_bytes * size)),
        )
    pixel_size = len(Image.new(mode, (1, 1)).tobytes())
    raw = bytes(
        rnd.randrange(256) for _ in range(size * size * pixel_size)
    )
    return Image.frombytes(mode, (size, size), raw)


def digest(obj) -> str:
    """Stable hash of pixel bytes, palette bytes and basic geometry."""
    h = hashlib.sha256()
    h.update(obj.mode.encode())
    h.update(str(obj.size).encode())
    h.update(obj.tobytes())
    palette = obj.getpalette()
    if palette is not None:
        h.update(repr(palette).encode())
    return h.hexdigest()[:16]


def png_bytes(obj: Image.Image) -> str:
    # PNG cannot encode every mode (CMYK, I, F, I;16, PA); TIFF covers those.
    png_modes = ("1", "L", "LA", "P", "RGB", "RGBA", "RGBX")
    fmt = "PNG" if obj.mode in png_modes else "TIFF"
    buf = io.BytesIO()
    obj.save(buf, format=fmt)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def section(lines: list[str], name: str) -> None:
    lines.append(f"\n== {name} ==")


def record(lines: list[str], label: str, obj) -> None:
    if isinstance(obj, Image.Image):
        lines.append(f"{label}: {digest(obj)} png={png_bytes(obj)}")
    else:
        lines.append(f"{label}: {obj!r}")


def expect_error(lines: list[str], label: str, exc_type, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        h = hashlib.sha256(f"{type(e).__name__}:{e}".encode()).hexdigest()[:16]
        lines.append(f"{label}: {type(e).__name__} {h}")
    except BaseException as e:  # wrong exception type: must stay identical too
        lines.append(f"{label}: WRONG-EXC {type(e).__name__}: {e}")
    else:
        lines.append(f"{label}: MISSING-ERROR")


def main() -> int:
    lines: list[str] = [f"python={sys.version.split()[0]} pillow={Image.__version__}"]
    failures = 0

    # ---- 1. convert() with dither options across modes ----------------
    section(lines, "convert+dither")
    for mode in ("L", "RGB", "RGBA"):
        base = make_sample(mode, 31)
        for target in ("1", "P", "PA"):
            for dither in DITHERS:
                try:
                    # Public API: Image.convert normalizes the Dither enum
                    # to the 0/nonzero flag the C core understands.
                    out = base.convert(target, dither=dither)
                    record(lines, f"convert {mode}->{target} dither={int(dither)}", out)
                except Exception as e:
                    lines.append(
                        f"convert {mode}->{target} dither={int(dither)}: "
                        f"ERROR {type(e).__name__}: {e}"
                    )
                    failures += 1
        # Direct core boundary: the C flag is 0 vs nonzero (Floyd-Steinberg).
        # L/RGB convert to P, RGBA to PA.
        core_target = "PA" if mode == "RGBA" else "P"
        for flag in (0, 1):
            out = base._new(base.im.convert(core_target, flag))
            record(lines, f"core convert {mode}->{core_target} flag={flag}", out)
        # web vs adaptive palette path
        for palette in (Image.Palette.WEB, Image.Palette.ADAPTIVE):
            out = base.convert("P", palette=palette)
            record(lines, f"convert {mode}->P palette={int(palette)}", out)

    # convert_into (dither=0 internal call)
    rgb = make_sample("RGB", 15)
    target = Image.new("L", rgb.size)
    rgb.im.convert_into(target.im)
    record(lines, "convert_into RGB->L", target)

    # ---- 2. quantize() with every method and boundary colors ----------
    section(lines, "quantize")
    for mode in ("L", "RGB", "RGBA"):
        base = make_sample(mode, 31)
        for colors in (1, 2, 16, 256):
            for method in QUANTIZE_METHODS:
                label = f"quantize {mode} colors={colors} method={int(method)}"
                if mode == "RGBA" and method in (
                    Image.Quantize.MEDIANCUT,
                    Image.Quantize.MAXCOVERAGE,
                ):
                    expect_error(
                        lines,
                        label,
                        ValueError,
                        base.quantize,
                        colors,
                        method=method,
                    )
                else:
                    try:
                        record(lines, label, base.quantize(colors, method=method))
                    except Exception as e:
                        lines.append(f"{label}: ERROR {type(e).__name__}: {e}")
                        failures += 1
        # kmeans boundary parameter (median cut; unsupported on RGBA)
        if mode != "RGBA":
            record(
                lines,
                f"quantize {mode} kmeans=0",
                base.quantize(16, method=Image.Quantize.MEDIANCUT, kmeans=0),
            )
        else:
            expect_error(
                lines,
                "quantize RGBA kmeans=0",
                ValueError,
                base.quantize,
                16,
                method=Image.Quantize.MEDIANCUT,
                kmeans=0,
            )

    # quantize error boundaries: colors out of range, negative kmeans
    base = make_sample("RGB", 12)
    expect_error(lines, "quantize colors=0", ValueError, base.quantize, 0)
    expect_error(lines, "quantize colors=257", ValueError, base.quantize, 257)
    expect_error(
        lines,
        "quantize kmeans=-1",
        ValueError,
        base.quantize, 16, kmeans=-1,
    )

    # ---- 3. resample() with every filter and boundary sizes -----------
    section(lines, "resample")
    for mode in MODES:
        base = make_sample(mode, 31)
        for f in RESAMPLE_FILTERS:
            for size in SIZES:
                label = f"resample {mode} filter={int(f)} size={size}"
                try:
                    # Public resize() normalizes P/1 to NEAREST, so it never
                    # raises for valid filters; record the actual output.
                    record(lines, label, base.resize((size, size), f))
                except Exception as e:
                    lines.append(f"{label}: ERROR {type(e).__name__}: {e}")
                    failures += 1
        # Core boundary: ImagingResample itself still rejects P/1 for every
        # non-transform filter. Exercise it directly via the core entry.
        if mode in ("P", "1"):
            expect_error(
                lines,
                f"core resample {mode} mode-error",
                Exception,
                base.im.resize,
                (8, 8),
                int(Image.Resampling.BOX),
            )
        # upsize + crop box boundary on one RGB-style sample
        if mode not in ("P", "1"):
            record(
                lines,
                f"resample {mode} filter=NEAREST box",
                base.resize((10, 10), Image.Resampling.NEAREST, box=(1, 1, 11, 11)),
            )
    # zero-size must remain an error
    expect_error(
        lines, "resize 0x0", Exception, make_sample("RGB", 8).resize, (0, 8)
    )

    # reduce() internal path (thumbnail uses it): integer scales
    base = make_sample("RGB", 32)
    record(lines, "reduce scale=2", base.reduce(2))
    record(lines, "reduce scale=(2,4) box", base.reduce((2, 4), box=(0, 0, 16, 16)))

    # ---- 4. transform() with methods/filters/fill boundary -----------
    section(lines, "transform")
    base = make_sample("RGB", 32)
    affine_data = (1.2, 0.1, 4.0, 0.0, 0.9, -3.0)
    perspective_data = (1.0, 2e-4, -5.0, 1e-4, 1.0, 3.0, 0.0, 0.0)
    quad_data = tuple(float(v) for v in (0, 0, 31, 2, 29, 31, 3, 28))
    data_by_method = {
        Image.Transform.AFFINE: affine_data,
        Image.Transform.PERSPECTIVE: perspective_data,
        Image.Transform.QUAD: quad_data,
    }
    for method in TRANSFORM_METHODS:
        for f in (
            Image.Resampling.NEAREST,
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
        ):
            for fill in (0, 1):
                label = f"transform method={int(method)} filter={int(f)} fill={fill}"
                out = base.transform(
                    (24, 24), method, data_by_method[method], resample=f, fillcolor=0
                )
                record(lines, label, out)
    # fill=0 path through the C keyword directly (Image.transform default fill=1;
    # call the core to cover the fill flag). The core transform is in-place.
    fill_target = base.copy()
    fill_target.im.transform(
        (0, 0, 24, 24),
        base.im,
        int(Image.Transform.AFFINE),
        affine_data,
        int(Image.Resampling.NEAREST),
        0,
    )
    record(lines, "core transform fill=0", fill_target)

    # invalid method / filter must keep raising ValueError at the core
    expect_error(
        lines,
        "core transform bad method",
        ValueError,
        base.im.transform,
        (0, 0, 16, 16), base.im, 99, affine_data, 0, 1,
    )
    expect_error(
        lines,
        "core transform bad filter",
        ValueError,
        base.im.transform,
        (0, 0, 16, 16), base.im, int(Image.Transform.AFFINE),
        affine_data, 99, 1,
    )
    expect_error(
        lines,
        "resample bad filter",
        Exception,
        base.resize, (16, 16), 99,
    )

    # ---- 5. transpose constants still map to the same operations ------
    section(lines, "transpose")
    base = make_sample("RGBA", 13)
    for op in (
        Image.Transpose.FLIP_LEFT_RIGHT,
        Image.Transpose.FLIP_TOP_BOTTOM,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
        Image.Transpose.TRANSPOSE,
        Image.Transpose.TRANSVERSE,
    ):
        record(lines, f"transpose {int(op)}", base.transpose(op))

    # ImageTransform wrappers exercise the same core constants
    record(
        lines,
        "AffineTransform",
        base.transform((20, 20), ImageTransform.AffineTransform(affine_data)),
    )

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    overall = hashlib.sha256(text.encode()).hexdigest()
    sys.stdout.write(f"\nOVERALL {overall}\n")
    sys.stdout.write(f"FAILURES {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)

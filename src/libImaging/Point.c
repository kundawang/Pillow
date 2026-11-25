/*
 * The Python Imaging Library
 * $Id$
 *
 * point (pixel) translation
 *
 * history:
 * 1995-11-27 fl   Created
 * 1996-03-31 fl   Fixed colour support
 * 1996-08-13 fl   Support 8-bit to "1" thresholding
 * 1997-05-31 fl   Added floating point transform
 * 1998-07-02 fl   Added integer point transform
 * 1998-07-17 fl   Support L to anything lookup
 * 2004-12-18 fl   Refactored; added I to L lookup
 *
 * Copyright (c) 1997-2004 by Secret Labs AB.
 * Copyright (c) 1995-2004 by Fredrik Lundh.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

typedef struct {
    const void *table;
} im_point_context;

static Imaging
im_point_new_output(ModeID mode, int xsize, int ysize) {
    /* With non-default row alignment, loops do not write alignment gaps. */
    if (ImagingDefaultArena.alignment > 1) {
        return ImagingNew(mode, xsize, ysize);
    }
    return ImagingNewDirty(mode, xsize, ysize);
}

// Note: all point handlers must fill the entirety of `imOut`,
//       as it is allocated dirty by ImagingPoint().

static void
im_point_8_8(Imaging imOut, Imaging imIn, im_point_context *context) {
    /* 8-bit source, 8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    int xsize = imIn->xsize;
    for (int y = 0; y < imIn->ysize; y++) {
        UINT8 *restrict in = imIn->image8[y];
        UINT8 *restrict out = imOut->image8[y];
        int x = 0;
        for (; x + 4 <= xsize; x += 4) {
            UINT32 values = MAKE_UINT32(
                table[in[x]], table[in[x + 1]], table[in[x + 2]], table[in[x + 3]]
            );
            memcpy(out + x, &values, sizeof(values));
        }
        for (; x < xsize; x++) {
            out[x] = table[in[x]];
        }
    }
}

static void
im_point_2x8_2x8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 2x8-bit source, 2x8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    UINT8 *table_256 = table + 256;
    int xsize = imIn->xsize;
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *restrict in = (UINT8 *)imIn->image[y];
        UINT32 *restrict out = (UINT32 *)imOut->image32[y];
        for (x = 0; x < xsize; x++) {
            out[x] = MAKE_UINT32(table[in[0]], 0, 0, table_256[in[3]]);
            in += 4;
        }
    }
}

static void
im_point_3x8_3x8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 3x8-bit source, 3x8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    UINT8 *table_256 = table + 256;
    UINT8 *table_512 = table + 512;
    int xsize = imIn->xsize;
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *restrict in = (UINT8 *)imIn->image[y];
        UINT32 *restrict out = (UINT32 *)imOut->image32[y];
        for (x = 0; x < xsize; x++) {
            out[x] = MAKE_UINT32(table[in[0]], table_256[in[1]], table_512[in[2]], 0);
            in += 4;
        }
    }
}

static void
im_point_4x8_4x8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 4x8-bit source, 4x8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    UINT8 *table_256 = table + 256;
    UINT8 *table_512 = table + 512;
    UINT8 *table_768 = table + 768;
    int xsize = imIn->xsize;
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *restrict in = (UINT8 *)imIn->image[y];
        UINT32 *restrict out = (UINT32 *)imOut->image32[y];
        for (x = 0; x < xsize; x++) {
            out[x] = MAKE_UINT32(
                table[in[0]], table_256[in[1]], table_512[in[2]], table_768[in[3]]
            );
            in += 4;
        }
    }
}

static void
im_point_8_32(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 8-bit source, 32-bit destination */
    char *table = (char *)context->table;
    int xsize = imIn->xsize;
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *restrict in = imIn->image8[y];
        INT32 *restrict out = imOut->image32[y];
        for (x = 0; x < xsize; x++) {
            memcpy(out + x, table + in[x] * sizeof(INT32), sizeof(INT32));
        }
    }
}

static void
im_point_32_8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 32-bit source, 8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    int xsize = imIn->xsize;
    for (y = 0; y < imIn->ysize; y++) {
        INT32 *restrict in = imIn->image32[y];
        UINT8 *restrict out = imOut->image8[y];
        for (x = 0; x < xsize; x++) {
            int v = in[x];
            if (v < 0) {
                v = 0;
            } else if (v > 65535) {
                v = 65535;
            }
            out[x] = table[v];
        }
    }
}

Imaging
ImagingPoint(Imaging imIn, ModeID mode, const void *table) {
    /* lookup table transform */

    ImagingSectionCookie cookie;
    Imaging imOut;
    im_point_context context;

    if (!imIn) {
        return (Imaging)ImagingError_ModeError();
    }

    if (mode == IMAGING_MODE_UNKNOWN) {
        mode = imIn->mode;
    }

    if (imIn->type == IMAGING_TYPE_UINT8) {
        if (imIn->mode != mode && !imIn->image8) {
            goto mode_mismatch;
        }
    } else if (imIn->type == IMAGING_TYPE_INT32) {
        if (mode != IMAGING_MODE_L) {
            // I images can only be used with L output mode
            goto mode_mismatch;
        }
    } else {
        // Reject F modes and modes based on I;16
        goto mode_mismatch;
    }

    imOut = im_point_new_output(mode, imIn->xsize, imIn->ysize);
    if (!imOut) {
        return NULL;
    }

    ImagingCopyPalette(imOut, imIn);

    ImagingSectionEnter(&cookie);
    context.table = table;

    if (imIn->type == IMAGING_TYPE_UINT8) {
        if (imIn->bands == imOut->bands && imIn->type == imOut->type) {
            switch (imIn->bands) {
                case 1:
                    im_point_8_8(imOut, imIn, &context);
                    break;
                case 2:
                    im_point_2x8_2x8(imOut, imIn, &context);
                    break;
                case 3:
                    im_point_3x8_3x8(imOut, imIn, &context);
                    break;
                case 4:
                    im_point_4x8_4x8(imOut, imIn, &context);
                    break;
                default:
                    /* this cannot really happen */
                    im_point_8_8(imOut, imIn, &context);
                    break;
            }
        } else {
            im_point_8_32(imOut, imIn, &context);
        }
    } else {
        im_point_32_8(imOut, imIn, &context);
    }

    ImagingSectionLeave(&cookie);

    return imOut;

mode_mismatch:
    return (Imaging)ImagingError_ValueError(
        "point operation not supported for this mode"
    );
}

Imaging
ImagingPointTransform(Imaging imIn, double scale, double offset) {
    /* scale/offset transform */

    ImagingSectionCookie cookie;
    Imaging imOut;
    int x, y;
    int xsize;
    int ysize;

    if (!imIn || (imIn->mode != IMAGING_MODE_I && imIn->mode != IMAGING_MODE_I_16 &&
                  imIn->mode != IMAGING_MODE_F)) {
        return (Imaging)ImagingError_ModeError();
    }

    xsize = imIn->xsize;
    ysize = imIn->ysize;

    imOut = im_point_new_output(imIn->mode, xsize, ysize);
    if (!imOut) {
        return NULL;
    }

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            ImagingSectionEnter(&cookie);
            for (y = 0; y < ysize; y++) {
                INT32 *restrict in = imIn->image32[y];
                INT32 *restrict out = imOut->image32[y];
                /* FIXME: add clipping? */
                for (x = 0; x < xsize; x++) {
                    out[x] = in[x] * scale + offset;
                }
            }
            ImagingSectionLeave(&cookie);
            break;
        case IMAGING_TYPE_FLOAT32:
            ImagingSectionEnter(&cookie);
            for (y = 0; y < ysize; y++) {
                FLOAT32 *restrict in = (FLOAT32 *)imIn->image32[y];
                FLOAT32 *restrict out = (FLOAT32 *)imOut->image32[y];
                for (x = 0; x < xsize; x++) {
                    out[x] = in[x] * scale + offset;
                }
            }
            ImagingSectionLeave(&cookie);
            break;
        case IMAGING_TYPE_I16:
            if (imIn->mode == IMAGING_MODE_I_16) {
                ImagingSectionEnter(&cookie);
                for (y = 0; y < ysize; y++) {
                    char *restrict in = (char *)imIn->image[y];
                    char *restrict out = (char *)imOut->image[y];
                    /* FIXME: add clipping? */
                    for (x = 0; x < xsize; x++) {
                        UINT16 v;
                        memcpy(&v, in + x * sizeof(v), sizeof(v));
                        v = v * scale + offset;
                        memcpy(out + x * sizeof(UINT16), &v, sizeof(v));
                    }
                }
                ImagingSectionLeave(&cookie);
                break;
            }
            /* FALL THROUGH */
        default:
            ImagingDelete(imOut);
            return (Imaging)ImagingError_ValueError("internal error");
    }

    return imOut;
}

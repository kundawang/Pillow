/*
 * The Python Imaging Library.
 * $Id$
 *
 * decoder for PackBits image data.
 *
 * history:
 * 96-04-19 fl Created
 *
 * Copyright (c) Fredrik Lundh 1996.
 * Copyright (c) Secret Labs AB 1997.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

/* Unpack the completed current line and advance to the next one.
   Returns -1 when the last line has been written, 0 otherwise. */
static int
packbits_newline(Imaging im, ImagingCodecState state) {
    state->shuffle(
        (UINT8 *)im->image[state->y + state->yoff] + state->xoff * im->pixelsize,
        state->buffer,
        state->xsize
    );

    state->x = 0;

    if (++state->y >= state->ysize) {
        /* End of file (errcode = 0) */
        return -1;
    }
    return 0;
}

/* Write "count" copies of "value" into the line buffer, spanning line
   boundaries when a run is longer than the space left in the line. */
static int
packbits_fill(Imaging im, ImagingCodecState state, UINT8 value, int count) {
    while (count > 0) {
        if (state->x >= state->bytes) {
            if (packbits_newline(im, state) < 0) {
                return -1;
            }
        }

        int take = MIN(state->bytes - state->x, count);
        memset(state->buffer + state->x, value, take);
        state->x += take;
        count -= take;
    }

    return 0;
}

/* Copy "count" literal bytes from "in" into the line buffer, spanning line
   boundaries when the literal crosses one. Returns a pointer just past the
   consumed input, or NULL when the image is complete. */
static const UINT8 *
packbits_copy(Imaging im, ImagingCodecState state, const UINT8 *in, int count) {
    while (count > 0) {
        if (state->x >= state->bytes) {
            if (packbits_newline(im, state) < 0) {
                return NULL;
            }
        }

        int take = MIN(state->bytes - state->x, count);
        memcpy(state->buffer + state->x, in, take);
        state->x += take;
        in += take;
        count -= take;
    }

    return in;
}

int
ImagingPackbitsDecode(
    Imaging im, ImagingCodecState state, UINT8 *buf, Py_ssize_t bytes
) {
    UINT8 n;
    UINT8 *ptr;

    ptr = buf;

    for (;;) {
        if (bytes < 1) {
            return ptr - buf;
        }

        if (ptr[0] & 0x80) {
            if (ptr[0] == 0x80) {
                /* Nop */
                ptr++;
                bytes--;
                continue;
            }

            /* Run */
            if (bytes < 2) {
                return ptr - buf;
            }

            n = 257 - ptr[0];

            if (packbits_fill(im, state, ptr[1], n) < 0) {
                return -1;
            }

            ptr += 2;
            bytes -= 2;

        } else {
            /* Literal */
            n = ptr[0] + 2;

            if (bytes < n) {
                return ptr - buf;
            }

            const UINT8 *next = packbits_copy(im, state, ptr + 1, n - 1);
            if (!next) {
                return -1;
            }
            bytes -= next - ptr;
            ptr = (UINT8 *)next;
        }

        if (state->x >= state->bytes) {
            if (packbits_newline(im, state) < 0) {
                return -1;
            }
        }
    }
}

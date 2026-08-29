"""Anti-aliased rasterisation of a nanometre layout onto an arbitrary pixel grid.

This is the piece that makes option (b) work. The same Layout is sampled twice
-- once at 1 nm/px over a 1 um window, once at 10 nm/px over a 10 um window --
so the wide view is a genuinely coarser acquisition rather than a resize of the
fine image, and the target coordinate stays analytic.

Three design points carry the performance and the quality:

  * Rotation is applied once, to the sampling grid, not per shape. Every
    primitive is then axis-aligned in the layout frame, so rotation is free.
  * Periodic arrays are painted by a modulo expression over the whole grid
    rather than shape by shape. A 10 um DRAM view holds ~7000 contacts; one
    vectorised expression replaces 7000 draw calls.
  * Coverage comes from a signed distance field, not supersampling. Every
    primitive here has a closed-form distance, and a linear ramp across one
    pixel is the exact area coverage for a straight edge. That works at output
    resolution -- 1e6 cells instead of the 1.6e7 a 4x supersample would need --
    and gives smoother edges than 4x supersampling does.
"""
from __future__ import annotations

import math
import numpy as np

from .layout import Circle, ContactLattice, Layout, LineArray, Rect


def _signed_offset(t, pitch):
    """Signed distance from t to the nearest lattice line of the given pitch."""
    return (t + 0.5 * pitch) % pitch - 0.5 * pitch


def _coverage(dist_nm, px_nm):
    """Fractional pixel coverage from a signed distance (negative = inside).
    The linear ramp spans one pixel, which is the exact area a straight edge
    sweeps as it crosses a pixel."""
    return np.clip(0.5 - dist_nm * (1.0 / px_nm), 0.0, 1.0)


def _paint(tile, dist, px_nm, intensity):
    """Composite one primitive over the tile using painter's-algorithm alpha."""
    cov = _coverage(dist, px_nm)
    np.multiply(cov, np.float32(intensity), out=dist)   # reuse buffer
    tile *= (1.0 - cov)
    tile += dist


def rasterize(layout: Layout, origin_nm: tuple, px_nm: float, n_px: int,
              supersample: int = 1) -> np.ndarray:
    """Sample `layout` onto an n_px x n_px grid.

    origin_nm   : (x0, y0) nm coordinate of the window's top-left corner
    px_nm       : physical size of one output pixel, in nm
    supersample : extra supersampling on top of the analytic coverage. The
                  default of 1 is correct for Manhattan geometry; higher values
                  exist only to verify that against brute force.

    Returns float32 grey levels, shape (n_px, n_px).
    """
    S = int(supersample)
    N = n_px * S
    sub_nm = px_nm / S
    x0, y0 = origin_nm

    a = math.radians(-layout.angle_deg)      # image frame -> layout frame
    ca, sa = math.cos(a), math.sin(a)
    ccx, ccy = layout.centre_nm

    gx = (x0 + (np.arange(N, dtype=np.float64) + 0.5) * sub_nm) - ccx
    gy = (y0 + (np.arange(N, dtype=np.float64) + 0.5) * sub_nm) - ccy

    u = (gx[None, :] * ca - gy[:, None] * sa + ccx).astype(np.float32)
    v = (gx[None, :] * sa + gy[:, None] * ca + ccy).astype(np.float32)

    tile = np.full((N, N), np.float32(layout.background), dtype=np.float32)

    for arr in layout.arrays:
        if isinstance(arr, LineArray):
            t = v if arr.axis == "h" else u
            off = _signed_offset(t - arr.phase_nm, arr.pitch_nm)
            r = arr.roughness
            if r is None or not r.resolvable_at(sub_nm):
                d = np.abs(off)
                d -= arr.width_nm / 2
            else:
                # Each edge of each line wobbles independently, so the two
                # half-widths are perturbed separately -- line-width roughness
                # rather than a line that merely wandering sideways.
                s = u if arr.axis == "h" else v
                line_idx = np.rint((t - arr.phase_nm - off) *
                                   (1.0 / arr.pitch_nm)).astype(np.int32)
                r_lo, r_hi = r.sample_both(line_idx, s.copy())
                half = arr.width_nm / 2
                d = np.maximum(off - (half + r_hi), (-half - r_lo) - off)
        elif isinstance(arr, ContactLattice):
            du = _signed_offset(u - arr.phase_x_nm, arr.pitch_x_nm)
            dv = _signed_offset(v - arr.phase_y_nm, arr.pitch_y_nm)
            d = np.hypot(du, dv)
            d -= arr.r_nm
        else:
            raise TypeError(f"unsupported array: {type(arr).__name__}")
        _paint(tile, d, sub_nm, arr.intensity)

    # Shapes are evaluated only inside their own footprint. A coarse-periodic
    # landmark lattice can carry several hundred shapes, and computing each one
    # over the full grid would cost more than the arrays do. The layout-space
    # bounding box is mapped forward into image space to get the index range;
    # one pixel of slack absorbs the coverage ramp.
    ca_f, sa_f = math.cos(-a), math.sin(-a)      # layout frame -> image frame
    for sh in layout.shapes:
        bx0, by0, bx1, by1 = sh.bbox_nm()
        pxs, pys = [], []
        for qx, qy in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)):
            dx, dy = qx - ccx, qy - ccy
            pxs.append(ccx + dx * ca_f - dy * sa_f)
            pys.append(ccy + dx * sa_f + dy * ca_f)
        j0 = int(math.floor((min(pxs) - x0) / sub_nm)) - 1
        j1 = int(math.ceil((max(pxs) - x0) / sub_nm)) + 2
        i0 = int(math.floor((min(pys) - y0) / sub_nm)) - 1
        i1 = int(math.ceil((max(pys) - y0) / sub_nm)) + 2
        j0, i0 = max(j0, 0), max(i0, 0)
        j1, i1 = min(j1, N), min(i1, N)
        if j1 <= j0 or i1 <= i0:
            continue

        us, vs = u[i0:i1, j0:j1], v[i0:i1, j0:j1]
        if isinstance(sh, Rect):
            d = np.maximum(np.abs(us - sh.cx_nm) - sh.w_nm / 2,
                           np.abs(vs - sh.cy_nm) - sh.h_nm / 2)
        elif isinstance(sh, Circle):
            d = np.hypot(us - sh.cx_nm, vs - sh.cy_nm)
            d -= sh.r_nm
        else:
            raise TypeError(f"unsupported primitive: {type(sh).__name__}")
        _paint(tile[i0:i1, j0:j1], d, sub_nm, sh.intensity)

    if S == 1:
        return tile
    return tile.reshape(n_px, S, n_px, S).mean(axis=(1, 3)).astype(np.float32)


def capture(layout: Layout, centre_nm: tuple, px_nm: float, n_px: int = 1000,
            supersample: int = 1) -> np.ndarray:
    """Rasterise an n_px window centred on a nm coordinate."""
    half = n_px * px_nm / 2.0
    return rasterize(layout, (centre_nm[0] - half, centre_nm[1] - half),
                     px_nm, n_px, supersample)


def make_pair(layout: Layout, target_nm: tuple, offset_px: tuple,
              ref_px_nm: float = 1.0, wide_px_nm: float = 10.0,
              n_px: int = 1000, supersample: int = 1, absent: bool = False,
              relative_theta_deg: float = 0.0):
    """Render one (reference, wide) pair from a single layout.

    target_nm : nm coordinate of the site, normally the landmark centre
    offset_px : where the site lands in the wide image, as an offset in wide
                pixels from the wide image centre -- the stage navigation error

    absent    : Phase 2 Set C. The reference is rendered with its landmark, but
                the wide view is rendered from the same layout with the landmark
                shapes removed -- the same periodic architecture, a different die
                region where the reference's unique site does not occur. The
                returned gt is then meaningless (the caller marks it absent).

    relative_theta_deg : Phase 2 `theta`. The wide capture's rotation relative
                to the reference, CCW-positive, about the match centre. Default
                0.0 renders wide at the same `layout.angle_deg` as the
                reference -- exactly today's (Phase 1) behaviour, byte-for-byte.
                Nonzero: rotate ONLY the wide render, about the same pivot the
                rasteriser already uses for `layout.angle_deg` (Layout.centre_nm,
                which is the landmark -- i.e. exactly (gt_x, gt_y)'s nm
                coordinate). Because the pivot IS the ground-truth point, it is
                a fixed point of the rotation, so gt_x/gt_y need no correction
                here (unlike barrel/scan distortion, which shift a point that
                is not the pivot).

                Sign convention verified empirically against the rasteriser
                (not assumed): increasing `layout.angle_deg` turns the rendered
                pattern CLOCKWISE (see docs/PHASE2_RESEARCH_NOTES.md). So a
                CCW-positive relative rotation of the wide is realised as
                `layout.angle_deg -= relative_theta_deg` for the wide render
                only.

    Returns (ref, wide, gt_xy) with gt in wide-image pixel coordinates,
    fractional and exact: it comes from the placement, never from the image.
    """
    ref = capture(layout, target_nm, ref_px_nm, n_px, supersample)

    gt_x = n_px / 2.0 + offset_px[0]
    gt_y = n_px / 2.0 + offset_px[1]
    wide_origin = (target_nm[0] - gt_x * wide_px_nm,
                   target_nm[1] - gt_y * wide_px_nm)

    saved_angle = layout.angle_deg
    if relative_theta_deg != 0.0:
        layout.angle_deg = saved_angle - relative_theta_deg
    try:
        if absent:
            # Temporarily drop the landmark shapes for the wide render only;
            # the periodic arrays (the architecture) stay, so the wide is
            # periodically similar but contains no true instance. Restored
            # immediately after.
            saved_shapes = layout.shapes
            layout.shapes = []
            try:
                wide = rasterize(layout, wide_origin, wide_px_nm, n_px, supersample)
            finally:
                layout.shapes = saved_shapes
        else:
            wide = rasterize(layout, wide_origin, wide_px_nm, n_px, supersample)
    finally:
        layout.angle_deg = saved_angle

    assert ref.shape == (n_px, n_px), ref.shape
    assert wide.shape == (n_px, n_px), wide.shape
    return ref, wide, (gt_x, gt_y)


def make_pair_shared_canvas(layout: Layout, target_nm: tuple, offset_px: tuple,
                            ref_px_nm: float = 1.0, wide_px_nm: float = 10.0,
                            n_px: int = 1000):
    """Faithful stand-in for the organisers' starter pipeline.

    Their prompt renders one large fine field, crops the reference out of it,
    and resizes the same field down to make the wide view. Both images then
    descend from identical pixels, so the reference is an exact 10x upsample of
    its target region. That shared lineage -- not the resampling arithmetic --
    is the real shortcut: a network can learn "find the patch whose downsample
    matches these pixels" instead of learning to find the pattern.

    An earlier attempt at this mode rendered the reference independently and
    only changed how the wide view was resampled. It measured as the same
    domain (correlation 0.900, of which independent noise alone explains 0.915),
    because exact analytic coverage and box-averaging a fine render agree, and
    the wide defocus erases what little differs. Sharing the canvas is what
    actually creates the gap.

    Memory: the fine field is 10000x10000 float32, ~400 MB, so this path wants
    far fewer workers than the physical one.
    """
    # Roughness is disabled on this path. The 10000x10000 intermediate makes
    # every per-pixel table lookup a 400 MB allocation, and this mode exists to
    # imitate the organisers' starter pipeline, which models no roughness at
    # all -- so dropping it here makes the stand-in more faithful, not less.
    for _a in layout.arrays:
        if isinstance(_a, LineArray):
            _a.roughness = None

    ratio = int(round(wide_px_nm / ref_px_nm))
    gt_x = n_px / 2.0 + offset_px[0]
    gt_y = n_px / 2.0 + offset_px[1]
    wide_origin = (target_nm[0] - gt_x * wide_px_nm,
                   target_nm[1] - gt_y * wide_px_nm)

    fine = rasterize(layout, wide_origin, ref_px_nm, n_px * ratio, supersample=1)

    cx, cy = int(round(gt_x * ratio)), int(round(gt_y * ratio))
    h = n_px // 2
    x0 = min(max(cx - h, 0), fine.shape[1] - n_px)
    y0 = min(max(cy - h, 0), fine.shape[0] - n_px)
    ref = fine[y0:y0 + n_px, x0:x0 + n_px].copy()

    assert ref.shape == (n_px, n_px)
    return ref, fine, (gt_x, gt_y), ratio


def make_pair_resize(layout: Layout, target_nm: tuple, offset_px: tuple,
                     ref_px_nm: float = 1.0, wide_px_nm: float = 10.0,
                     n_px: int = 1000, fine_px_nm: float = 2.0):
    """Render a pair the way Applied Materials' starter prompt does it.

    Their prompt builds the wide view by rendering a larger continuous field at
    fine resolution and then *resizing it down*. That makes the wide image a
    pixel-domain copy of the fine one, which leaves an exact deterministic
    relationship between reference and target that a network will happily learn
    instead of learning to find the pattern.

    We never train on this. It exists as a held-out validation domain: a
    stand-in for their generator, so we can show the method transfers rather
    than having memorised our own renderer. Anything trained on the physical
    path and evaluated here is being asked to cross a real domain gap.

    fine_px_nm trades fidelity against memory. At 1 nm/px the intermediate is
    10000x10000; 2 nm/px keeps it to 5000x5000, still a genuine downsample.
    """
    ref = capture(layout, target_nm, ref_px_nm, n_px, supersample=1)

    gt_x = n_px / 2.0 + offset_px[0]
    gt_y = n_px / 2.0 + offset_px[1]
    wide_origin = (target_nm[0] - gt_x * wide_px_nm,
                   target_nm[1] - gt_y * wide_px_nm)

    factor = int(round(wide_px_nm / fine_px_nm))
    if factor < 1 or abs(factor * fine_px_nm - wide_px_nm) > 1e-9:
        raise ValueError("wide_px_nm must be an integer multiple of fine_px_nm")

    fine = rasterize(layout, wide_origin, fine_px_nm, n_px * factor,
                     supersample=1)

    assert ref.shape == (n_px, n_px)
    assert fine.shape == (n_px * factor, n_px * factor)
    # the fine field is returned undownsampled: the caller must apply the
    # fine-scale optics *before* shrinking, which is what creates the domain
    # difference this mode exists to represent
    return ref, fine, (gt_x, gt_y), factor


def to_uint8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)

"""Vector layout in physical nanometre coordinates.

Nothing here knows about pixels. A Layout describes geometry in nm; the
rasteriser samples it at whatever pixel size a given capture uses. This is the
core of option (b): the 1 nm/px reference and the 10 nm/px wide view are two
independent samplings of one ground truth, so the target coordinate is
analytic rather than recovered from a resize.

Geometry is Manhattan per the problem-statement webinar -- axis-aligned
rectangles and circles, with the whole layout rotated by the small stage angle.

Periodic structure is stored *analytically* (pitch, width, phase) rather than
as thousands of individual shapes. A DRAM field over a 10 um view contains
~7000 contacts; drawing those one at a time dominates render time, whereas one
modulo expression paints the entire array in a single vectorised pass and makes
rotation cost nothing. Only aperiodic features are explicit shapes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Grey levels are 8-bit targets measured on real IC SEM imagery (MIIC, n=200):
# dark phase 59, bright phase 146, Michelson contrast 0.42.
SUBSTRATE = 59.0
METAL = 146.0
CONTACT = 175.0
CONTACT_CORE = 70.0


# --------------------------------------------------------------------------
# Analytic periodic primitives (painted first, in layout frame)
# --------------------------------------------------------------------------

@dataclass
class EdgeRoughness:
    """Line-edge roughness, as a displacement table shared by both captures.

    Roughness is a property of the *wafer*, not of the capture, so the
    reference and the wide view must see the identical edge wobble. Generating
    it per render would make the same physical line wobble differently in the
    two images, which is unphysical and would hand a matcher a false cue.

    The cited model is PSD(f) = PSD(0) / [1 + (2*pi*f*xi)^(2H+1)] with the
    roughness exponent H = 0.5, which makes the exponent 2 -- a Lorentzian,
    i.e. exactly the spectrum of an exponentially correlated (Ornstein-
    Uhlenbeck) process. That is generated exactly by a one-pole AR(1) filter of
    white noise with phi = exp(-ds/xi), rather than approximated by Gaussian
    smoothing.

    Reported values for modern lithography: xi = 7-13 nm, 3-sigma LWR 2.6-3.5 nm
    (Cutler et al., J. Micro/Nanopattern. Mater. Metrol. 20(1) 010901, 2021).

    Two independent tables are held so the two edges of a line move
    independently, which is line-*width* roughness rather than a line that
    merely wanders.
    """
    lo: object          # (n_lines, n_s) float32 displacement of the low edge
    hi: object          # ditto, high edge
    s0_nm: float
    ds_nm: float
    sigma_nm: float
    xi_nm: float

    def sample_both(self, line_idx, s_nm):
        """Displacement of both edges at once.

        The index arithmetic is identical for the two tables, so it is done
        once and shared -- this is the hot path, evaluated at every pixel of
        every grating.
        """
        n_lines, n_s = self.lo.shape
        li = line_idx % n_lines
        f = (s_nm - self.s0_nm) * (1.0 / self.ds_nm)
        np.clip(f, 0, n_s - 1.001, out=f)
        i0 = f.astype(np.int32)
        w = f - i0
        a_lo, b_lo = self.lo[li, i0], self.lo[li, i0 + 1]
        a_hi, b_hi = self.hi[li, i0], self.hi[li, i0 + 1]
        return a_lo + (b_lo - a_lo) * w, a_hi + (b_hi - a_hi) * w

    def resolvable_at(self, px_nm: float) -> bool:
        """Whether this roughness is worth rendering at a given pixel size.

        At 10 nm/px a 1 nm edge displacement is a tenth of a pixel: it is far
        below the sampling limit and costs real time to compute for no visible
        effect. Skipping it there is the same statement the physics already
        makes about the secondary-electron edge band.
        """
        return self.sigma_nm >= 0.15 * px_nm


def make_edge_roughness(rng, sigma_nm=1.0, xi_nm=10.0, H=0.5, n_lines=96,
                        span_nm=18000.0, ds_nm=4.0, s0_nm=-2000.0):
    """Build a roughness table by shaping white noise with the cited PSD."""
    from scipy.fft import next_fast_len

    n_need = int(span_nm / ds_nm) + 2
    # A prime-ish transform length drops the FFT onto Bluestein's algorithm and
    # costs an order of magnitude; round up to a 5-smooth length instead.
    n_s = next_fast_len(n_need)

    # Synthesised straight from the published PSD rather than from a filter
    # that happens to have the same spectrum: shape white noise in the
    # frequency domain by sqrt(PSD). Exact for any roughness exponent H, and
    # one FFT pair rather than a recursion over thousands of samples.
    e = rng.standard_normal((2, n_lines, n_s)).astype(np.float32)
    freq = np.fft.rfftfreq(n_s, d=ds_nm)
    psd = 1.0 / (1.0 + (2.0 * np.pi * freq * xi_nm) ** (2.0 * H + 1.0))
    x = np.fft.irfft(np.fft.rfft(e, axis=2) * np.sqrt(psd), n=n_s, axis=2)
    x *= sigma_nm / x.std()
    x = np.ascontiguousarray(x.astype(np.float32))
    return EdgeRoughness(x[0], x[1], s0_nm, ds_nm, sigma_nm, xi_nm)


@dataclass
class LineArray:
    """Infinite periodic line grating. axis='h' -> lines of constant y."""
    axis: str
    pitch_nm: float
    width_nm: float
    intensity: float
    phase_nm: float = 0.0
    roughness: object = None


@dataclass
class ContactLattice:
    """Periodic disc lattice. A via is a bright disc with a darker core, which
    reproduces the annular appearance measured in real SEM imagery."""
    pitch_x_nm: float
    pitch_y_nm: float
    r_nm: float
    intensity: float
    phase_x_nm: float = 0.0
    phase_y_nm: float = 0.0


# --------------------------------------------------------------------------
# Explicit aperiodic primitives (painted after the arrays)
# --------------------------------------------------------------------------

@dataclass
class Rect:
    cx_nm: float
    cy_nm: float
    w_nm: float
    h_nm: float
    intensity: float

    def bbox_nm(self):
        return (self.cx_nm - self.w_nm / 2, self.cy_nm - self.h_nm / 2,
                self.cx_nm + self.w_nm / 2, self.cy_nm + self.h_nm / 2)


@dataclass
class Circle:
    cx_nm: float
    cy_nm: float
    r_nm: float
    intensity: float

    def bbox_nm(self):
        return (self.cx_nm - self.r_nm, self.cy_nm - self.r_nm,
                self.cx_nm + self.r_nm, self.cy_nm + self.r_nm)


@dataclass
class Layout:
    extent_nm: float
    background: float = SUBSTRATE
    angle_deg: float = 0.0          # rigid rotation of the whole layout
    arrays: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    landmark_xy_nm: tuple | None = None
    rot_centre_nm: tuple | None = None
    meta: dict = field(default_factory=dict)

    @property
    def centre_nm(self):
        """Pivot of the rigid rotation.

        This defaults to the landmark rather than the geometric centre. The
        rasteriser rotates the sampling grid about this point, so any point
        away from it lands somewhere else once the field is rotated. With the
        pivot at the geometric centre, a landmark 1.4 um away drifts by up to
        120 nm at 5 deg -- the reference then is not centred on the landmark it
        is supposed to show. Pivoting on the target makes it a fixed point, so
        the reference is always centred on the site and the ground truth is
        exactly the landmark position.
        """
        if self.rot_centre_nm is not None:
            return self.rot_centre_nm
        if self.landmark_xy_nm is not None:
            return self.landmark_xy_nm
        return (self.extent_nm / 2, self.extent_nm / 2)

    def add_array(self, a):
        self.arrays.append(a)
        return a

    def add(self, s):
        self.shapes.append(s)
        return s


# --------------------------------------------------------------------------
# Landmarks -- the aperiodic feature that breaks translational symmetry.
# Sized into the 30-400 nm band: below ~30 nm a feature spans <3 px in the
# 10 nm/px wide view and the pair is unsolvable by construction.
# --------------------------------------------------------------------------

CLEAR_FACTOR = 1.22        # keep-out size relative to the landmark
CLEAR_MAX_NM = 470.0       # never blank out more than ~47% of the reference


def _lattice_offsets(repeat_nm: float | None, reach_nm: float):
    """Offsets at which a coarse-periodic feature recurs, covering +-reach."""
    if not repeat_nm:
        return [(0.0, 0.0)]
    k = int(np.ceil(reach_nm / repeat_nm))
    return [(i * repeat_nm, j * repeat_nm)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]


def add_plus_landmark(lay: Layout, cx: float, cy: float,
                      arm_len_nm: float = 300.0, arm_w_nm: float = 96.0,
                      intensity: float = METAL, clear_nm: float = 380.0):
    """The plus/cross pad used throughout the Applied Materials webinar.

    `clear_nm` first stamps a substrate-coloured keep-out square. A real
    aperiodic feature -- a strap region, a pad, a sub-array boundary --
    displaces the surrounding array rather than sitting on top of it, and
    without the keep-out a metal-coloured landmark drawn over metal-coloured
    lines is nearly invisible.
    """
    if clear_nm > 0:
        lay.add(Rect(cx, cy, clear_nm, clear_nm, lay.background))
    lay.add(Rect(cx, cy, arm_len_nm, arm_w_nm, intensity))
    lay.add(Rect(cx, cy, arm_w_nm, arm_len_nm, intensity))
    lay.landmark_xy_nm = (cx, cy)
    lay.meta["landmark"] = dict(type="plus", arm_len_nm=arm_len_nm,
                                arm_w_nm=arm_w_nm, clear_nm=clear_nm,
                                size_nm=arm_len_nm)
    return cx, cy


def add_pad_landmark(lay: Layout, cx: float, cy: float, size_nm: float = 240.0,
                     intensity: float = CONTACT_CORE):
    """Solid block, as in the deck's FinFET example figure."""
    lay.add(Rect(cx, cy, size_nm, size_nm, intensity))
    lay.landmark_xy_nm = (cx, cy)
    lay.meta["landmark"] = dict(type="pad", size_nm=size_nm)
    return cx, cy


LANDMARKS = {"plus": add_plus_landmark, "pad": add_pad_landmark}


def add_landmark_lattice(lay: Layout, cx: float, cy: float, kind: str,
                         repeat_nm: float | None, reach_nm: float,
                         jitter_nm: float = 0.0, rng=None, **kw):
    """Place the landmark, optionally repeating it on a coarse lattice.

    A real die does not carry one unique feature in a sea of perfect
    periodicity. Word-line straps, stitch regions and sub-array boundaries
    recur every 32-128 cells, which at DRAM pitches is a repeat every 1-4 um --
    so a 10 um wide view holds several near-identical candidates. That is the
    physical origin of "if more than one region matches", and it is the case the
    centre tie-break exists to resolve.

    The lattice is phased through the target, so the true site is itself one of
    the instances. The caller is responsible for keeping the target within half
    a period of the image centre, which makes the true site the centre-most
    candidate and therefore the answer the stated rule selects.
    """
    offs = _lattice_offsets(repeat_nm, reach_nm)
    fn = LANDMARKS[kind]
    for k, (dx, dy) in enumerate(offs):
        jx = jy = 0.0
        if jitter_nm and rng is not None and (dx or dy):
            jx, jy = rng.uniform(-jitter_nm, jitter_nm, 2)
        fn(lay, cx + dx + jx, cy + dy + jy, **kw)
    # the true site is the one at zero offset; restore it as the landmark
    lay.landmark_xy_nm = (cx, cy)
    lay.meta["landmark"]["repeat_nm"] = repeat_nm
    lay.meta["landmark"]["n_instances"] = len(offs)
    return len(offs)


# --------------------------------------------------------------------------
# Style builders
# --------------------------------------------------------------------------

def build_dram(extent_nm: float = 12000.0,
               wl_pitch_nm: float = 130.0, wl_width_nm: float = 44.0,
               bl_pitch_nm: float = 105.0, bl_width_nm: float = 34.0,
               contact_r_nm: float = 21.0, contact_core_r_nm: float = 8.0,
               angle_deg: float = 0.0,
               landmark_xy_nm: tuple | None = None,
               landmark: str = "plus") -> Layout:
    """DRAM-style: horizontal word lines crossed by vertical bit lines with a
    contact at every intersection -- the arrangement named in the problem
    statement. Pitches default into the resolved regime (~10-13 px per period
    in the wide view), matching the deck's own example figures and the
    100-150 nm metal pitch inferred from real SEM imagery.

    Line widths sit near a third of pitch so substrate stays visible between
    lines; at ~45% fill the two gratings merge and the image reads as a
    checkerboard rather than as crossing lines.
    """
    lay = Layout(extent_nm=extent_nm, angle_deg=angle_deg)
    lay.meta.update(style="dram", wl_pitch_nm=wl_pitch_nm, bl_pitch_nm=bl_pitch_nm,
                    wl_width_nm=wl_width_nm, bl_width_nm=bl_width_nm,
                    contact_r_nm=contact_r_nm, angle_deg=angle_deg)

    lay.add_array(LineArray("h", wl_pitch_nm, wl_width_nm, METAL))
    lay.add_array(LineArray("v", bl_pitch_nm, bl_width_nm, METAL * 0.88))
    lay.add_array(ContactLattice(bl_pitch_nm, wl_pitch_nm, contact_r_nm, CONTACT))
    lay.add_array(ContactLattice(bl_pitch_nm, wl_pitch_nm, contact_core_r_nm,
                                 CONTACT_CORE))

    if landmark_xy_nm is not None:
        LANDMARKS[landmark](lay, *landmark_xy_nm)
    return lay


def build_finfet(extent_nm: float = 12000.0,
                 fin_pitch_nm: float = 90.0, fin_width_nm: float = 34.0,
                 gate_pitch_nm: float = 620.0, gate_width_nm: float = 130.0,
                 angle_deg: float = 0.0,
                 landmark_xy_nm: tuple | None = None,
                 landmark: str = "pad") -> Layout:
    """FinFET-style: dense parallel vertical fins crossed by horizontal gate
    bars -- the second architecture offered in the problem statement."""
    lay = Layout(extent_nm=extent_nm, angle_deg=angle_deg)
    lay.meta.update(style="finfet", fin_pitch_nm=fin_pitch_nm,
                    fin_width_nm=fin_width_nm, gate_pitch_nm=gate_pitch_nm,
                    gate_width_nm=gate_width_nm, angle_deg=angle_deg)

    lay.add_array(LineArray("v", fin_pitch_nm, fin_width_nm, METAL * 0.9))
    lay.add_array(LineArray("h", gate_pitch_nm, gate_width_nm, METAL))

    if landmark_xy_nm is not None:
        LANDMARKS[landmark](lay, *landmark_xy_nm)
    return lay


BUILDERS = {"dram": build_dram, "finfet": build_finfet}



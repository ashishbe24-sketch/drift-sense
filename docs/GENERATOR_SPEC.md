# PS-02 Dataset Generator — Physical Specification
### Every parameter, with its source. Compiled overnight, 1–2 Aug 2026.

This closes the three gaps that were blocking construction: **(1) real nanometre dimensions, (2) a physically justified landmark taxonomy, (3) defensible ranges for rotation, drift and dose.** Nothing below is invented; each number traces to a roadmap table, a peer-reviewed paper, a patent, or a measurement we made on real SEM data.

Architecture is **option (b)**: one vector layout in nanometres, two independent renders. Option (a) — render-large-then-resize — is retained behind `--render-mode resize` as a held-out validation domain standing in for Applied Materials' own generator.

---

## 0. The five findings that changed the design

**0.1 — The "closest to centre" tie-break is not arbitrary. It's a stage-accuracy prior.**
Published wafer-inspection stage accuracy is **<1.5 µm**, described variously as "a fraction of a micron" to "several micrometers." The wide FOV is 10 µm = 1000 px. So a stage error of σ ≈ 1 µm puts the true target within **±100 px of centre at 1σ, ±300 px at 3σ**. The target lands near the centre *because the stage is good*. Applied Materials' rule is the Bayesian posterior of that physics.
→ **Sample the target offset from a centred distribution (σ ≈ 1–2 µm), not uniformly.** Say why. Guard against degeneracy by generating a uniform-placement minority (see §5.3).

**0.2 — The genuine multi-match case has a physical origin: coarse-periodic straps.**
A DRAM MAT is 512 or 1024 cells. Word-line straps / stitch regions recur every 32–128 cells. At 1α (F = 14 nm, word-line pitch 2F = 28 nm) that is a repeat of **0.9–3.6 µm**, so a 10 µm wide view contains **2.8–11 repeats per axis → 9 to 121 near-identical candidates.**
→ That is where "if more than one region matches" comes from. Generate it deliberately with a `coarse_period` parameter, rather than hoping ambiguity emerges.

**0.3 — The wide view is noisier because it is a fast survey scan, not because it is wide.**
Both images are 1000×1000. Same pixel count ⇒ same acquisition time at equal dwell. Increasing pixel size lowers acquisition time *while maintaining the same measurement precision per pixel* (Trampert et al.). So geometry alone predicts **equal** SNR. The noise difference comes from **dwell time and frame count**: the reference was a slow characterisation capture, the wide view is a quick look taken while the tool is trying to recover.
→ Dose becomes a *derived* parameter (dwell × current × frames), not an invented knob. Ratios in §4.2.

**0.4 — The two captures differ in noise colour, not just amplitude.**
CD-SEM frame integration: at **≥8 frames the high-frequency noise region is flat (white); at 2–4 frames it is noticeably sloped (correlated).** The careful reference (many frames) gets white noise; the fast wide scan (few frames) gets correlated noise.
→ Two different noise *processes*, not one process at two amplitudes. Almost no competing team will model this.

**0.5 — Dose and blur are physically coupled and must not be sampled independently.**
Raising beam current adds electrons per pixel without costing time, "however… the virtual spot size of the electron gun increases and the image appears blurred" (Trampert et al.).
→ **σ_PSF must increase with beam current.** Independent sampling of dose and blur produces physically impossible image pairs.

---

## 1. Geometry — the fixed frame

| | Reference ("100×") | Wide search ("10×") |
|---|---|---|
| Size | 1000 × 1000 px | 1000 × 1000 px |
| Pixel | 1 nm | 10 nm |
| FOV | 1.00 µm | 10.0 µm |
| Reference footprint inside wide | — | **100 × 100 px (1% of area)** |

**Independent validation that this spec is real CD-SEM practice**, not a toy:
- Cutler et al. 2021: Hitachi CG4000 CD-SEM, **1024 × 1024 px, 100 k×, pixel size 1.32 nm**, 500 V.
- CD-SEM roughness work: **500 eV landing energy, 0.8 nm pixel, 1024 × 1024**.
- Trampert et al.: FEI Helios 650, **5 nm and 10 nm pixel sizes**, 1024 × 884.

Our 1 nm/px 1000² reference and 10 nm/px 1000² wide view sit squarely inside published practice. Worth one line in the deck.

---

## 2. Layout dimensions — GAP 1 CLOSED

### 2.1 Logic ground rules — IRDS 2022 More Moore, Table MM-7 / Figure MM-2

| Year of production | 2022 | 2025 | 2028 | 2031 | 2034 | 2037 |
|---|---|---|---|---|---|---|
| IRDS node label | G48M24 | G45M20 | G42M16 | G40M16/T2 | G38M16/T4 | G38M16/T6 |
| Industry "node range" | "3nm" | "2nm" | "1.5nm" | "1.0nm eq" | "0.7nm eq" | "0.5nm eq" |
| **Gate pitch (nm)** | **48** | **45** | **42** | **40** | **38** | **38** |
| **M0 pitch (nm)** | **24** | **20** | **16** | **16** | **16** | **16** |
| **M1 pitch (nm)** | **32** | **23** | **21** | **20** | **19** | **19** |
| **Mx pitch (nm)** | **32** | **24** | **20** | **16** | **16** | **16** |
| Gate length Lg, HP (nm) | 16 | 14 | 14 | 12 | 12 | 12 |
| Spacer width (nm) | 6 | 6 | 5 | 5 | 4 | 4 |
| Contact CD (nm) | 20 | 19 | 20 | 18 | 18 | 18 |
| FinFET fin width (nm) | 5.0 | — | — | — | — | — |
| PN separation width (nm) | 45 | 40 | 20 | 15 | 15 | 10 |
| Platform device | finFET | LGAA | LGAA/CFET | LGAA-3D | LGAA-3D | LGAA-3D |

*IRDS notation: Gxx = contacted gate pitch, Mxx = tightest metal pitch. Note Applied Materials sits on the IRDS More Moore team — citing IRDS is citing a body the sponsor helps write.*

**Published foundry values** (cross-check): TSMC N5 — **fin pitch 28 nm, contacted gate pitch 51 nm**. TSMC N3 — contacted gate pitch 45 nm.

### 2.2 DRAM — IRDS 2022 §5.1 + published node data

- Cell is **1T-1C**; cell-size factor **a = [cell size]/[half-pitch]²**; **6F² (a = 6) is currently most common**, migrating toward 4F².
- **6F² geometry: cell = 2F × 3F. Word-line pitch = 2F. Bit-line pitch = 3F.** F = half-pitch = half the minimum line pitch.
- DRAM half-pitch F by generation: **1x ≈ 18–19 nm (2015–16), 1y ≈ 17 nm (2017–18), 1z ≈ 16 nm (2019–20), 1α ≈ 14 nm (2021), 1β ≈ 12 nm (2023), 1γ ≈ 10 nm (2025).**
- Sub-array (**MAT**) = **512 × 512 or 1024 × 1024** cells; MATs group into banks.

Derived cell pitches: 1z → WL 32 nm / BL 48 nm. 1α → WL 28 nm / BL 42 nm. 1β → WL 24 nm / BL 36 nm.

### 2.3 The Nyquist consequence — computed, and it matters

Nyquist needs ≥2 px per period **in the wide view** (10 nm/px):

| Structure | Pitch | px in reference | px in wide | Status |
|---|---|---|---|---|
| DRAM 1β word line (2F) | 24 nm | 24.0 | **2.40** | marginal |
| Logic 2022 M0 | 24 nm | 24.0 | **2.40** | marginal |
| TSMC N5 fin pitch | 28 nm | 28.0 | **2.80** | marginal |
| DRAM 1α word line (2F) | 28 nm | 28.0 | **2.80** | marginal |
| DRAM 1z word line (2F) | 32 nm | 32.0 | 3.20 | resolved, coarse |
| Logic 2022 gate pitch | 48 nm | 48.0 | 4.80 | resolved, coarse |
| TSMC N5 contacted gate pitch | 51 nm | 51.0 | 5.10 | resolved, coarse |
| Upper metal (MIIC-inferred, low) | 70 nm | 70.0 | 7.00 | well resolved |
| **AMAT example figures (approx)** | **~110 nm** | 110 | **11.0** | **well resolved** |
| Upper metal (MIIC-inferred, median) | 128 nm | 128.0 | 12.8 | well resolved |
| Upper metal (MIIC-inferred, high) | 320 nm | 320.0 | 32.0 | well resolved |

**Read this carefully.** A modern DRAM cell array or fin array is **at or below Nyquist in the wide view** — 2.4–2.8 px per period. It does not resolve; it **aliases into moiré**. This is a documented electron-microscopy phenomenon: aliasing "is formally linked to the formation of 2D moiré fringes," occurring "when less than two pixels are used to record a lattice spacing."

Applied Materials' own example figures sit at ~11 px/period — the comfortably-resolved regime, consistent with an upper metal layer.

→ **Sweep both regimes.** Resolved (70–320 nm) is the primary case and matches AMAT's examples. Aliased (24–48 nm) is physically real, genuinely hard, and is exactly the kind of honest failure case the 10% explainability bucket wants.

### 2.4 Recovering MIIC's unknown scale from physics

MIIC ships no scale bar, so our measured pitch (median 51 px) had no nanometre value. We can recover it: we measured the **edge-brightening overshoot peaking 2 px inside the feature**, and the SE escape depth / edge-effect band is **< 10 nm, typically 3–7 nm**. Therefore 2 px ≈ 3–7 nm ⇒ **MIIC pixel ≈ 1.5–3.5 nm/px**.

Then MIIC's median 51 px pitch ⇒ **~100–150 nm**, and its 27–128 px range ⇒ **~55–380 nm**. Consistent with upper-metal pitches and with AMAT's example figures. Two independent routes agreeing is the strongest evidence we have that our chosen pitch band is right.

---

## 3. Landmark taxonomy — GAP 2 CLOSED

### 3.1 Hard size bounds, computed

| Requirement | px in wide view | Physical size |
|---|---|---|
| Nyquist floor | 2 | 20 nm |
| Barely recognisable | 3 | **30 nm** |
| Usable | 5 | 50 nm |
| Comfortable | 10 | 100 nm |

The reference FOV is 1000 nm, so the landmark must also *fit*. **Usable band: ~30–400 nm.** Below ~30 nm the pair is unsolvable by construction — generate a labelled minority of those deliberately as failure cases.

### 3.2 What actually breaks periodicity on a real die

**Ruled out on physical grounds — and worth stating:** alignment and overlay marks. Box-in-box targets are **~20 µm per side**, and scribe-line alignment marks run **~120 µm × 38 µm**. They are *larger than our entire 10 µm wide field of view*. A generator that drops an overlay mark into a 1 µm reference is physically incoherent. Saying so in the deck demonstrates we checked.

**In-band, citable landmark types (30–400 nm):**

| # | Landmark | Physical basis | Source |
|---|---|---|---|
| L1 | **Word-line strap / stitch region** | Straps recur every 32–128 cells; locally aperiodic, globally coarse-periodic | "Memory array with strap cells", US9847120 / US9607685 |
| L2 | **Sub-array boundary** — sense-amplifier or sub-word-driver band | SWD and SA are placed alternately around each cell array; SWD pitch = 2× sub-word-line pitch, SA pitch = 2× bit-line pitch | DRAM architecture patents; Semiwiki SA/SWD metal patterning |
| L3 | **Dummy cells / dummy word lines / dummy bit lines** | Inserted at array edges to isolate sub-arrays; shared between adjacent sub-arrays in sense-amp-sharing DRAM | US5886939; SRAM dummy-row practice |
| L4 | **Line-end terminations / cuts** | Line patterns deliberately terminate at different lengths to prevent shorts/breaks | US6495870 |
| L5 | **Dummy-fill / pattern-density boundary** | CMP requires uniform metal density; regions failing density rules get non-functional fill, changing local linewidth and spacing | GF180MCU PDK §13.3 design rules; US20030229479A1 |
| L6 | **Defect** — particle, bridge, break, missing/merged contact | The site was "characterized earlier"; defect-review navigation is the canonical use case | PS premise; MIIC anomaly set (116 annotated) |
| L7 | **Contact/via chain irregularity** | Vias render as **bright annuli with dark cores** (measured in MIIC), not filled discs | measured, §4.4 |

**Two-tier landmark model.** L1/L2/L3 are *coarse-periodic* — they define `coarse_period` and manufacture the honest multi-match case (§0.2). L4/L5/L6/L7 are *singular* — one instance, fully disambiguating. Sweep the mix; report accuracy separately for each, because they are different problems.

---

## 4. Physics stack — GAP 3 CLOSED

### 4.1 Render order

```
vector layout (nm)
  → analytic anti-aliased rasterisation at the view's own pixel size
  → SE yield map:  δ(φ) = δ₀ / cos(φ)  →  edge brightening + dark halo
  → Gaussian probe PSF, σ coupled to beam current
  → drift shear + vibration jitter (per scan line)
  → over-dispersed Poisson shot noise (dose-dependent)
  → Gaussian read noise
  → charging (optional, v2)
  → 8-bit quantisation
```

Paradigm follows **NIST ARTIMAGEN** (Cizmar/Vladár/Postek): define the sample, then render it, with *defined, known* amounts of each artefact. That is the whole point of synthetic SEM data, and NIST says so explicitly — they built artificial 2-D images specifically "to test the instrument and the measurement algorithms."

### 4.2 Parameter table

| Layer | Model | Value / range | Source |
|---|---|---|---|
| **Edge brightening** | SE yield vs surface tilt: **δ(φ) = δ(0)/cos φ**; more generally **δ(φ) ∝ secⁿφ, n ≈ 1.3 → 0.8 with Z** | overshoot **+16%**, peaking **+2 px inside** the feature | sec-law: Reimer/Springer SE imaging; magnitude: **measured on MIIC, n=749 edges** |
| **Dark halo** | SE depletion just outside the edge | **−19%** below background, ~4 px wide | **measured on MIIC** (absent from AMAT's starter prompt) |
| **Edge band width** | SE escape depth | **< 10 nm, typically 3–7 nm** | SE escape depth literature |
| **Probe PSF** | Gaussian | **σ ≈ 0.56 px** at MIIC scale (10–90% rise = **1.44 px**); sweep 0.4–1.2 px | **measured**; ARTIMAGEN |
| **PSF ↔ dose coupling** | higher beam current ⇒ larger virtual spot ⇒ blurrier | couple σ to current; **never sample independently** | Trampert et al. |
| **Shot noise** | **over-dispersed Poisson**, Fano ≈ **1.9** | SE-count variance from 500 eV primaries on Si is **1.9× the Poisson value** | *Microscopy* (Oxford) 68(4):279 |
| **Noise sources** | five Poisson channels: primary emission, secondary emission, scintillator, photocathode, PMT | — | *Scanning*, 10.1002/sca.4950260106 |
| **Dose** | dwell × current × frames | dwell **10 / 30 / 40 µs**; current **0.1–0.8 nA**; frames **4 / 8 / 16 / 32** | Trampert et al.; CD-SEM practice |
| **Wide:ref noise ratio** | √(dose ratio) | **1.5× – 4×** (e.g. ref 30 µs×16 vs wide 10 µs×4 ⇒ 12× dose ⇒ **3.46× noise**) | computed from the above |
| **Noise colour** | ≥8 frames ⇒ white; 2–4 frames ⇒ sloped/correlated | reference white, wide correlated | CD-SEM frame-integration studies |
| **LER / LWR** | **PSD(f) = PSD(0) / [1 + (2πfξ)^(2H+1)]**, with **σ² ≈ PSD(0)/[(2H+1)ξ]** | **H = 0.5**; **ξ = 7–13 nm**; **3σ LWR = 2.6–3.5 nm** | **Cutler et al., J. Micro/Nanopattern. Mater. Metrol. 20(1) 010901 (2021)** — open access CC-BY |
| **Vibration** | per-scan-line displacement ⇒ **serrated edges**; frame averaging ⇒ position-misassignment blur | documented **25 nm** line broadening from a single cooling fan | **Postek, Vladár, Cizmar, Proc. SPIE 9236, 923605 (2014)** — NIST, public domain |
| **Drift** | slow creep ⇒ **elongation** along creep direction + loss of sharpness | **nm/s range at room temperature** | same NIST paper; SPM/SEM drift studies |
| **Rotation** | in-plane θ between captures | **0–5 mrad.** θ-stage repeatability 4 µrad; same-tool revisit sub-mrad; wafer flat/notch alignment **±0.3° = ±5.2 mrad** | stage specs; wafer aligner practice |
| **Charging** (v2) | negative ⇒ bright regions; positive ⇒ dark; beam deflection ⇒ **bright streaks, sudden shifts, dark bands** | qualitative in v1, off by default | charging literature; unity-crossover practice |

### 4.3 Rotation, in pixels

| Rotation | across 100 px footprint | across full 1000 px frame |
|---|---|---|
| 0.004 mrad (θ-stage repeatability) | 0.00 px | 0.00 px |
| 1.0 mrad (same-tool revisit) | 0.10 px | **1.00 px** |
| 5.2 mrad (±0.3° flat alignment) | 0.52 px | **5.20 px** |

At sub-pixel tolerance, ≥1 mrad already matters at the frame edge. Sweep 0–5 mrad.

### 4.4 Appearance details measured on real SEM (MIIC, n = 120–200)

| Quantity | Value |
|---|---|
| Dark phase / bright phase | 59 / 146 (8-bit) |
| Michelson contrast | **0.42** |
| Grey mean / std | 94.1 / 41.3 |
| Histogram | clearly **bimodal** (p25 = 62, p75 = 136) |
| Radial PSD log-log slope | **−3.32** |
| Vias | **bright annuli with dark cores**, not filled discs |
| Line edges | visible roughness + width bulges at contacts |

**Caveat, stated openly:** MIIC is JPEG-compressed (blockiness 1.81; σ_noise 0.58 ⇒ SNR ≈ 71, physically impossible for SEM). So it calibrates the **deterministic** optics — contrast, pitch, edge profile, overshoot, halo — and the **noise model comes from the physics papers instead**. Anyone calibrating noise amplitude against MIIC is calibrating against JPEG artifacts.

---

## 5. Sampling policy

### 5.1 Style
`--style {dram, finfet}` is a required CLI parameter per the submission spec, so both are built. DRAM = word lines × bit lines + contacts; FinFET = parallel fins + gate bars.

### 5.2 Regime mix
| Regime | Pitch band | Share | Rationale |
|---|---|---|---|
| Resolved (primary) | 70–320 nm | 60% | matches AMAT examples + MIIC-inferred metal layers |
| Coarse | 42–70 nm | 25% | gate/contact pitch regime |
| Aliased (hard) | 24–42 nm | 15% | DRAM cell array / fin pitch — moiré in wide view |

### 5.3 Target placement — the stage prior
Offset from wide-image centre drawn from an isotropic Gaussian, **σ = 1.0–2.0 µm (100–200 px)**, truncated so the 100 px footprint stays inside the frame.
**Guard: 20% of pairs use uniform placement.** Without this a network can degenerate to "always predict centre" and still score well on our own set — and Applied Materials' generator may well place uniformly. Report accuracy on both sub-populations separately.

### 5.4 Breadth principle
Our distribution must strictly contain theirs. Every range above is deliberately wider than a single plausible setting: pitch spans both Nyquist regimes, noise spans 1.5–4×, rotation spans 0–5 mrad, landmark saliency spans below-floor to comfortable.

### 5.5 Determinism
One seed per pair drives every draw. `labels.csv` records `pair_id, style, gt_x, gt_y (fractional), pitch_nm, regime, landmark_type, coarse_period_nm, saliency, rot_mrad, dwell_us, current_nA, frames, psf_sigma_px, noise_sigma, seed`. Ground truth is analytic — the landmark is at the nanometre coordinate we placed it at, divided by 10 nm/px. Never rounded.

---

## 6. Citation ledger (for the 30% bucket)

**Synthetic SEM generation paradigm**
1. Cizmar, Vladár, Ming, Postek. *Simulated SEM Images for Resolution Measurement.* **Scanning** 30(5):381–391, 2008.
2. Cizmar, Vladár, Postek. *Optimization of Accurate SEM Imaging by Use of Artificial Images.* **Proc. SPIE** 7378, 737815, 2009.
3. ARTIMAGEN, NIST, public domain — `github.com/strec007/artimagen`.

**Vibration, drift, and why synthetic images are used to test algorithms**
4. Postek, Vladár, Cizmar. *Does Your SEM Really Tell the Truth? Part 3: Vibration and Drift.* **Proc. SPIE** 9236, 923605, 2014. doi:10.1117/12.2065235. *(NIST, not subject to copyright.)*

**Noise physics**
5. *Impact of secondary electron emission noise in SEM.* **Microscopy** (Oxford) 68(4):279 — Fano ≈ 1.9 on Si at 500 eV.
6. *Effect of shot noise and secondary emission noise in SEM images.* **Scanning**, doi:10.1002/sca.4950260106.
7. Trampert, Bourghorbel, Potocek, Peemen, Schlinkmann, Dahmen, Slusallek. *How should a fixed budget of dwell time be spent in SEM to optimize image quality?* arXiv:1801.04085 — dwell/current/blur trade-off.

**Roughness**
8. Cutler, Thackeray, Trefonas, Millward, Lee, Mack. *Pattern roughness analysis using power spectral density.* **J. Micro/Nanopattern. Mater. Metrol.** 20(1) 010901, 2021. CC-BY. — the PSD model we sample from.

**Dimensions**
9. **IRDS 2022 Update, More Moore** — Table MM-7, Figure MM-2, §5.1 DRAM. IEEE.
10. Published foundry data: TSMC N5 fin pitch 28 nm / CGP 51 nm; N3 CGP 45 nm.

**Layout structures**
11. US9847120 / US9607685 — memory array with strap cells.
12. US5886939 — sub dummy bit line and sub dummy word line.
13. US6495870 — line patterns terminating at different lengths.
14. US20030229479A1 — dummy fill for integrated circuits; GF180MCU PDK §13.3 dummy-metal design rules.

**The problem class (for the approach slides)**
15. US20160275672A1 (KLA-Tencor) — *Sub-Pixel Alignment of Inspection to Design*.
16. US8750597 — robust inspection alignment using design information.
17. US10922582 — *Localization of planar objects in images bearing repetitive patterns*.
18. *Towards reliable matching of images containing repetitive patterns*, Pattern Recognition Letters, 2011.
19. *SIFT Saliency Analysis for Matching Repetitive Structures*, Math. Problems in Eng., 2017.
20. Moiré/aliasing in scanning microscopy — *Moiré sampling in STEM*, Ultramicroscopy.

**Reference imagery**
21. Huang, Cheng, Yang, Lin, Shi, Yang, Gwee, Wen. *Joint Anomaly Detection and Inpainting for Microscopy Images via Deep Self-Supervised Learning.* **Proc. IEEE ICIP**, 2021. (MIIC; non-commercial research use.)

**Prior art for our own approach**
22. *Industrial wafer edge segmentation for alignment: a dual-resolution deep learning approach with synthetic pretraining.* **J. Intelligent Manufacturing**, 2026.
23. Shinde et al. *Defect Detection in Photolithographic Patterns Using Deep Learning Models Trained on Synthetic Data.* arXiv:2505.10192, 2025.
24. *Deep-CNN-Based Layout-to-SEM Image Reconstruction…* **Electronics** 14:2973.

---

## 7. Status

**Closed:** nanometre dimensions (IRDS + foundry + DRAM node data); landmark taxonomy (7 types, sized, sourced, with overlay marks explicitly ruled out); rotation/drift/dose ranges (stage specs + NIST + Trampert). Plus five design-changing findings that were not on the original list.

**Ready to build.** Remaining inputs are yours, not research: target pair count, and teammates.

**Deliberately deferred to v2:** charging artifacts, scan-line streaking, contamination drift between captures. All three are robustness augmentations rather than core physics, and all three have citations already in hand.

*(Update, later than "v2": charging and scan-line streaking were implemented for Phase 1 —
`driftsense/physics.py: apply_charging`. Contamination drift between captures remains unimplemented.)*

---

## 8. Phase 2 addendum — optical aberrations (29 Aug 2026)

Four parameters were named in §"Pipeline order" (top of `physics.py`'s docstring, itself following
this document's citation of the NIST ARTIMAGEN paradigm) but never actually implemented — astigmatism,
barrel distortion, vignetting, gamma. Closed for Phase 2, since Set B ("scan distortion") needs the
geometric-warp category these represent, and the earlier gap was a real spec-vs-code mismatch, not a
deliberate deferral like the ones above.

**A framing difference from the rest of this ledger, stated plainly:** the parameters above (edge
overshoot, Fano factor, LER, pitch tables) are *measured or standards-sourced* values with numeric
citations. The four below are *standard optical/detector models*, cited to the textbook or classical
result that defines the model form — not to a paper measuring SEM-specific parameter values, because
no such measurement was sourced for this generator. The parameter *ranges* (e.g. barrel k1 ∈
[−0.06, 0.06]) are therefore engineering choices calibrated to produce a visually/quantitatively
plausible effect size on our own images, not literature-derived numbers, and should be described that
way if asked, not oversold as measured.

25. **Astigmatism** — modelled as an elliptical (direction-dependent) point-spread function: sharp
    along one axis, blurred along the perpendicular one, set by an aberration angle. This is the
    standard electron-optical astigmatism model described in Goldstein et al., *Scanning Electron
    Microscopy and X-Ray Microanalysis*, 4th ed., Springer, 2018 (the standard SEM reference text;
    astigmatism as an uncorrected stigmator error producing elliptical probe shape).
26. **Barrel / pincushion distortion** — single-term radial model, `r' = r(1 + k1·(r/r_max)²)`. This
    is the classical Brown–Conrady radial lens-distortion model: Brown, D.C., *Decentering Distortion
    of Lenses*, Photogrammetric Engineering, 1966. Applied here as a generic raster-scan geometric
    nonlinearity (magnetic/electrostatic scan-coil nonlinearity is the SEM-specific analogue of lens
    distortion), not because a specific SEM instrument's k1 was measured.
27. **Vignetting** — quadratic radial intensity falloff toward the corners, modelling reduced
    detector/collection solid-angle away from the optical axis. Modelled generically (not the
    photographic cos⁴ law, which assumes a different aperture geometry than an SEM secondary-electron
    detector); no SEM-specific vignetting-profile citation is claimed.
28. **Gamma** — a power-law intensity transform, `I_out = I_in^gamma`, modelling detector/amplifier
    non-linearity. Standard detector-response model; no specific SEM amplifier curve is cited.

**Ground-truth handling under these warps** (relevant to citation 17, repetitive-pattern localization,
above): barrel distortion moves the landmark's *apparent position*, so the label must be shifted by
the same forward map applied to the pixels — implemented and the sign convention verified empirically
(not derived-and-trusted) against a synthetic test with a known landmark location. See
`docs/PHASE2_RESEARCH_NOTES.md` for the validation numbers.

29. **Scan distortion** (`driftsense/physics.py: scan_distortion_field`) — a smooth, low-frequency
    2-D displacement field warping the wide capture, built from two low-frequency sinusoids. This
    represents the *static spatial distortion* component of SEM raster scanning, documented as
    distinct from time-varying drift (already cited separately, ref. 4/22 above): Cui et al.,
    *Scanning Electron Microscope Calibration Using a Multi-Image Non-Linear Minimization Process*,
    **Machining Science and Technology**, 2015, doi:10.1080/15599612.2015.1034903 (IRISA/Université
    Rennes 1 Lagadic team; also an IEEE ICRA 2014 conference version, IEEE Xplore doc 6907621) —
    which explicitly decomposes SEM image distortion into static spatial nonlinearity (from
    raster-scan instabilities) versus temporally-varying drift, and calibrates the former with a
    grating-based multi-image method. **Same honest framing as astigmatism/barrel/vignette/gamma
    above:** the paper establishes that this class of distortion is real and documented, and
    motivates modelling it as a smooth low-order spatial field rather than a single global shift —
    it is not the source of our specific sinusoid parameterisation or amplitude range (max 6 px),
    which are engineering choices tuned to be a visible but plausible warp on our own images, not a
    fit to this paper's measured coefficients. Applied to the wide capture only (a *relative* warp
    between reference and wide), gated off by default so Phase 1 pairs are unaffected; ground truth
    is shifted by the field's value at the landmark, keeping the label exact under the warp
    (validated in `docs/PHASE2_RESEARCH_NOTES.md`).

30. **Absent pairs (Set C, `present=0`)** — rendered as a *different die region of the same
    architecture*: the periodic arrays are kept, the landmark shapes are dropped, so the negative is
    plausible and periodically similar rather than a random unrelated crop. This is a hard-negative
    construction, not an arbitrary one: the design follows the standard rationale in the
    Siamese/metric-learning literature that random or unrelated negatives are too easy and under-test
    a rejection mechanism — see the general pair-construction discussion in Rosebrock, *Building
    image pairs for siamese networks with Python*, PyImageSearch, 2020, and the broader hard-negative-
    mining practice it summarises. The organizers' own description of Set C ("a different die region
    of the same architecture... plausible and periodically similar") is itself a hard-negative
    specification, which this generator implements directly rather than approximating with an easier
    negative.

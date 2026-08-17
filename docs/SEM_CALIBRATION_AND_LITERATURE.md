# Deep Dive: Real-SEM Calibration + Literature/Patent Sweep
### PS-02 Drift-Sense — 1 Aug 2026

Two things happened here: the MIIC dataset was downloaded and **measured**, and the literature sweep moved off "datasets" onto **patents and papers**, which is where the graded citations live.

---

## PART A — Real SEM imagery, downloaded and measured

### A.1 What was acquired

Source: **MIIC (Microscopic Images of Integrated Circuits)**, Nanyang Technological University research-data repository, DOI `10.21979/N9/WBLTFI`. Downloaded via the Dataverse public API — no authentication required.

| Path | Contents |
|---|---|
| `D:\semicon\data\reference\Anomaly_test\normal_img\` | **1,272** grayscale SEM images, 512×512, JPEG |
| `…\Anomaly_test\abnormal_img\` + `_mask` + `_bbox` | 116 anomalous images with pixel masks and boxes |
| `D:\semicon\data\reference\Inpainting_test\` | 270 further images |
| `D:\semicon\data\miic_readme.txt` | licence + citation |

Only the two *test* archives were pulled (87 MB). The 666 MB training archive was skipped — we need appearance statistics, not volume.

**Licence:** copyright NTU, **non-commercial research use only**. We use it strictly as a calibration reference; we never train on it and never redistribute it. Required citation:

> L. Huang, D. Cheng, X. Yang, T. Lin, Y. Shi, K. Yang, B.-H. Gwee, B. Wen, *"Joint Anomaly Detection and Inpainting for Microscopy Images via Deep Self-Supervised Learning,"* Proc. IEEE ICIP, 2021.

*Note: extraction needed Windows' built-in `tar.exe` (libarchive) — the installed WinRAR is v4.0 (2011) and cannot read RAR5.*

### A.2 What it actually looks like

Real SEM of the **metal layer of a finished IC**: parallel metal lines at fixed pitch, vias, and staggered pill/capsule-shaped pads. Critically — this closely matches the "DRAM staggered array" figure in Applied Materials' own Example 1. **MIIC is a legitimate appearance target for our generator.**

Observations that change what we build:

1. **Vias render as bright annuli with dark cores**, not filled dots. A naive generator draws a filled circle for a contact; the real thing is a ring. Easy to get right, and visually obvious to a judge who looks at SEM every day.
2. **Many images are pure parallel lines with no unique feature at all** — the archetypal unsolvable-by-construction case. Real evidence that the ambiguity in this problem is not hypothetical.
3. **Line-edge roughness and line-width bulges are clearly visible** — edges wobble, lines swell at contacts. Straight-edged rectangles will look synthetic.
4. **A dark halo hugs the outside of every bright feature** before the edge rises. This is a real SE-emission signature and it is *not* in Applied Materials' starter prompt (which mentions only edge-brightening).

### A.3 Measured numbers — the generator's parameters, no longer guessed

Measured over 120–200 images (`normal_img`):

| Quantity | Measured value | Method |
|---|---|---|
| Image size / depth | 512×512, uint8, range 34–192 | — |
| Grey mean / std | **94.1 / 41.3** | 200 images |
| Dark phase / bright phase | **59 / 146**, Michelson contrast **0.42** | 20th/85th percentile |
| Histogram | clearly **bimodal** (p25=62, p75=136) | 16-bin |
| Dominant line pitch | **median 51 px, IQR 34–57, range 27–128** | FFT of 240 periodic profiles |
| **Edge 10–90% rise width** | **1.44 px → equivalent Gaussian PSF σ ≈ 0.56 px** | 749 aligned step edges |
| **Edge-brightening overshoot** | **+16% above plateau, peaking 2 px inside the feature** | same |
| **Dark undershoot outside edge** | **−19% below background**, extending ~4 px | same |
| Radial PSD log-log slope | **−3.32** (natural images ≈ −2) | 40 images |
| Sobel gradient magnitude | 45.5 mean | 60 images |

The median normalised edge profile — the single most useful thing measured:

```
offset  -6    -4    -2     0    +1    +2    +3    +5    +8   +10
value  -.08  -.15  -.19  +.16  +.81  1.16  1.16  1.09  1.01  .97
        <--- dark halo --->  <rise>  <overshoot>  <-- settles -->
```

This is a textbook SEM edge signature and we now have it **quantitatively, from real data**. Our renderer should reproduce this shape, not an idealised step.

### A.4 The honest caveat — and why stating it is worth marks

Two measurements say these images are **not raw detector output**:

- **Estimated noise σ = 0.58 grey levels (SNR ≈ 71).** Physically implausible for SEM, where low-dose imaging is dominated by Poisson shot noise.
- **Blockiness ratio 1.81** at the 8-px grid — unambiguous **JPEG compression**. Visible as 8×8 blocking under 4× zoom.

So MIIC has been compressed and effectively denoised. **Conclusion: calibrate the *deterministic* parts against MIIC — contrast, pitch, edge profile, overshoot, halo, PSD shape — and take the *noise* model from physics papers instead** (§B.3). Anyone calibrating noise amplitude against MIIC would be calibrating against JPEG artifacts.

Declaring this limitation in the deck is stronger than hiding it. It is exactly the kind of methodological honesty the 10% explainability bucket rewards.

---

## PART B — Literature and patent sweep

### B.1 The industry's real name for this problem

The searchable term is **die-to-database alignment** / **pattern-to-design alignment (PDA)**, not "template matching." Searching the industry vocabulary surfaced material the consumer-CV vocabulary never would.

- **US20160275672A1 — "Sub-Pixel Alignment of Inspection to Design"** (KLA-Tencor). Determining the position of inspection output in *design data space*; imports design files at recipe setup and aligns the design image to the wafer image. **Directly our problem, sub-pixel, from the sponsor's direct competitor.**
- **US8750597 — "Robust inspection alignment of semiconductor inspection tools using design information."**
- **US6690021B2 — wafer alignment in photolithography**; **TW541642B — wafer alignment method**; **CN102063015A — wafer and pattern alignment method.**
- KLA production practice: align a real SEM image against an *ideal SEM image rendered from post-OPC design*, then apply corrections. Note the shape of that — **render the reference, don't just crop it.** Same paradigm as our vector-first generator.
- *Precise Pattern Alignment for Die-to-Database Inspection Based on a GAN* — transforms SEM into CAD-like images so conventional alignment works. A learned domain bridge across the same gap we face.

**Why this matters:** citing the sponsor's own industry's patent literature is a different class of evidence from citing an OpenCV tutorial. The deck allows patents explicitly, and almost no student team will use them.

### B.2 Matching under repetitive patterns — the actual hard part, with a literature

Confirmation that our central difficulty is a named, studied problem:

- **US10922582 — "Localization of planar objects in images bearing repetitive patterns."** States the failure mode precisely: features on repetitive structures are either not distinctive enough to match, or match incorrectly — so existing algorithms **fail to localize**. This is our problem, patented.
- *Towards reliable matching of images containing repetitive patterns* (Pattern Recognition Letters, 2011) — matches *pairs* of interest points to cut local ambiguity.
- *SIFT Saliency Analysis for Matching Repetitive Structures* (Math. Problems in Eng., 2017) — **feature-saliency weighting to disambiguate.** This is the formal version of our "weight the aperiodic landmark" instinct.
- *Meaningful Matches in Stereovision* (arXiv 2011) — periodic façades produce a significant match at *every* repetition; introduces a statistical significance criterion rather than raw best-score.

**Idea this unlocks:** don't report a match score — report a **significance/uniqueness measure**. The right question isn't "how well does this location correlate?" but "how much better is this peak than the rest of the peak *population*?" In a periodic scene the peak population is the null distribution, and it is free. That also gives us a principled confidence value to report alongside (x, y) — which feeds the failure-analysis bucket directly.

### B.3 SEM noise physics — for the parts MIIC can't tell us

- Shot noise in SEM follows a **Poisson process, signal-dependent, not Gaussian** — the standard model.
- **Five distinct noise sources** are documented: primary emission, secondary emission, scintillator, photocathode, photomultiplier — each Poisson.
- *Effect of shot noise and secondary emission noise in SEM images* — **Scanning**, Wiley (`10.1002/sca.4950260106`).
- *Impact of secondary electron emission noise in SEM*, **Microscopy** (Oxford), 68(4):279. Key quantitative finding: the variance of SE count from 500 eV primaries on a **Si wafer is 1.9× the pure-Poisson value** — i.e. **pure Poisson underestimates real SEM noise by nearly 2×**.
- *SEM Image Signal-to-Noise Ratio Monitoring* (HAL, open access) — SNR estimation methodology.
- *Poisson shot-noise parameter estimation from a single SEM image* — how to fit noise parameters from one image.

**Idea this unlocks:** an **over-dispersed Poisson** (Poisson with a Fano factor ≈ 1.9, or negative-binomial) noise model, cited, instead of plain `np.random.poisson`. That is a small code change and a genuinely defensible one — precisely the kind of choice the 30% bucket is asking us to justify.

Together with ARTIMAGEN's taxonomy from the last sweep, the physics stack now has a citation behind every layer:

| Layer | Anchor citation |
|---|---|
| Vector layout → render paradigm | Cizmar et al., *Scanning* 30(5), 2008 (ARTIMAGEN) |
| Gaussian probe PSF | measured σ ≈ 0.56 px from MIIC + ARTIMAGEN |
| Edge brightening (+16%) & dark halo (−19%) | **measured**, + SE-yield literature |
| Over-dispersed Poisson shot noise (Fano ≈ 1.9) | *Microscopy* 68(4):279 |
| Multi-source detector noise | *Scanning* 10.1002/sca.4950260106 |
| Drift-distortion, vibration | ARTIMAGEN + the PS's own premise |
| Line-edge roughness | observed in MIIC; IRDS lithography roadmap |

### B.4 Cross-resolution matching from an adjacent field

Remote sensing has solved our shape of problem for decades — locate a high-res chip inside a coarse wide scene:

- *Coarse-to-fine matching via cross-fusion of satellite images* — dual-branch network producing descriptors at **both** high and low resolution; coarse match on the low-res map, then transposed onto high-res for refinement.
- *An Accurate and Robust Multimodal Template Matching Method Based on **Center-Point Localization** in Remote Sensing Imagery* (Remote Sensing 16(15):2831) — **the output is a center point, exactly our required output format.**
- *Coarse-to-Fine Geometric SIFT for large high-resolution satellite registration* — robust match at low resolution, then use it as a geometric constraint on the fine match.

**Idea this unlocks:** the two-branch, two-resolution structure is the established answer to a scale gap, and it fits our compute budget — cheap global coarse pass to generate candidates, expensive precise pass only on the survivors. Since compute time sits inside the 50%, "expensive work only on candidates" is a scoring argument as well as an engineering one.

### B.5 Public dimension numbers

IRDS is the right citable source for pitches, but the specific tables are inside PDFs rather than search results. Direct documents to pull: `2024IRDS_MET.pdf` (metrology), `2022IRDS_MM.pdf` (More Moore, gate/metal pitch), `2020IRDS_Litho.pdf`. IRDS notation: **Gxx = contacted gate pitch, Mxx = tightest metal pitch (nm)**. TechInsights' public DRAM scaling articles are a secondary source.

**Open item:** extract concrete pitch numbers from those PDFs before finalising the layout parameters, so every dimension in the generator traces to a public roadmap figure rather than to invention.

---

## PART C — What actually changed in the plan

**Confirmed, now with evidence rather than intuition:**
- Vector-first generation — matches both NIST ARTIMAGEN and KLA's production "render the reference from design" practice.
- Ambiguity, not detection, is the core difficulty — a patent (US10922582) states the failure mode outright.
- Coarse-to-fine over two resolutions is the field-standard answer to a scale gap.

**New, from this sweep:**
1. **Measured edge profile replaces an assumed one** — σ ≈ 0.56 px, +16% overshoot at +2 px, −19% dark halo. From real data, with the plot to show.
2. **Vias are annuli, not discs.**
3. **Over-dispersed Poisson noise (Fano ≈ 1.9)**, cited — not plain Poisson.
4. **A dark halo outside every bright edge** — absent from Applied Materials' own starter prompt.
5. **Report peak *significance*, not peak score** — the periodic peak population is a free null distribution, and it yields a confidence value for the failure-analysis bucket.
6. **Calibrate deterministic optics against MIIC, noise against physics papers** — and say plainly why, because MIIC is JPEG-compressed (blockiness 1.81, σ 0.58).
7. **Pitch range 27–128 px (median 51)** as a real-world sanity band for the layout randomiser.

**Still open:** option (b) sign-off, target pair count, teammates, and pulling concrete pitch numbers out of the IRDS PDFs.

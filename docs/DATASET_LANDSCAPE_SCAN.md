# Dataset & Prior-Art Scan — GitHub + Kaggle
### For PS-02 (Applied Materials, Drift-Sense). Inspiration only — we are not replicating anything here.

**Date:** 1 Aug 2026. Method: 24 GitHub repo searches via `gh search repos`, plus web/Kaggle searches.

---

## 0. Headline finding

**No dataset exists for this problem.** Nothing on GitHub or Kaggle provides *(high-res reference, wide low-res search)* image pairs with a known scale gap and ground-truth coordinates on periodic die layouts. That is not a gap in my search — it is why Applied Materials made "generate your own data" the problem. Confirmed, and it means our generator has no competition to be compared against.

What *does* exist splits into four buckets, each useful for a different reason.

---

## 1. THE find — ARTIMAGEN (NIST)

**`strec007/artimagen`** — *Artificial SEM Image Generator*, Petr Cizmar & Benjamin Swedlove, **U.S. NIST**. C++, **public domain**.

> "generate artificial SEM images of various samples, including gold-on-carbon resolution sample, or some **semiconductor structures**. Numerous effects that appear in real SEMs are simulated (**noise, drift-distortion, edge-effect**, etc.)"

Why this matters more than everything else combined:

1. **It is the same problem, solved by a national metrology lab.** Synthetic SEM images with *defined, known* amounts of each artefact — created precisely so imaging/metrology algorithms can be assessed against ground truth. That is exactly our use case.
2. **Its effect taxonomy is a published, citable structure** for our physics stack: noise → drift-distortion → edge-effect → vibration → focus/astigmatism. The rubric wants 2–3 credible public sources per augmentation choice; this hands us a defensible spine rather than a list we invented.
3. **Its two papers are directly citable** for the 30% bucket:
   - [1] P. Cizmar, A. E. Vladar, B. Ming, M. T. Postek. *Simulated SEM Images for Resolution Measurement.* **Scanning** 30(5):381–391, Sep–Oct 2008.
   - [2] P. Cizmar, A. E. Vladar, M. T. Postek. *Optimization of Accurate SEM Imaging by Use of Artificial Images.* **Proc. SPIE** 7378, 737815, May 2009.
4. **It independently validates our architecture choice.** ARTIMAGEN defines a *sample/feature description* and then *renders* it — vector-first, raster-second. That is option (b) from our discussion. We can state in the deck that our generator follows the NIST ARTIMAGEN paradigm. That sentence is worth marks.

**What we take:** the effect taxonomy, the vector→render paradigm, and the citations.
**What we do not take:** the code. It is 2009-era C++ with libtiff/fftw3/lua5.3 + CMake — and submissions must be **Python only**. Porting it is a trap. We reimplement the *concepts* in numpy.

---

## 2. Real SEM imagery — for calibrating what "realistic" looks like

We have no real data and can't use proprietary fab data. But we *can* look at public SEM imagery to check our synthetic output isn't cartoonish. This is a visual-calibration reference, not training data.

| Resource | Content | Why useful |
|---|---|---|
| **MIIC** — `wenbihan/MIIC-IAD`, NTU ([researchdata.ntu.edu.sg](https://researchdata.ntu.edu.sg/)) | **25,276 grayscale SEM images, 512×512**, metal layer of finished ICs (25,160 normal / 116 anomalous) | Closest public match to our domain: real SEM, real IC, grayscale, quasi-periodic metal routing. Best reference for contrast, noise texture, edge brightness. **Non-commercial research use only.** |
| **NFFA-EUROPE** — Nature *Sci. Data* 2018 ([sdata2018172](https://www.nature.com/articles/sdata2018172)) | 21,272 annotated SEM images, 10 nanostructure categories | The canonical citable public SEM dataset. Categories incl. *patterned surface* and *MEMS/electrodes*. Good citation for "what SEM images statistically look like". |
| Kaggle — [SEM images with nanoscale features](https://www.kaggle.com/datasets/adrianacosta0/sem-images-with-nanoscale-features) | NFFA-EUROPE repackaged | Convenient download path for the above. |
| Kaggle — [Two-Stage Semiconductor Defect Dataset (SEM Img)](https://www.kaggle.com/datasets/surjini/two-stage-semiconductor-defect-dataset-sem-img) | SEM defect imagery | Secondary appearance reference. |

**Idea this unlocks:** we can quantitatively match our synthetic images to real SEM statistics — grey-level histogram, noise power spectrum, edge-profile sharpness — instead of asserting realism. "Our generator's noise PSD matches real SEM data from MIIC" is a far stronger claim than "we added Gaussian noise." That is a concrete way to win the 30%.

---

## 3. Deliberately NOT relevant — and worth saying so

The overwhelming majority of "semiconductor dataset" results are **WM-811K wafer maps** ([Kaggle](https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image), 811,457 wafers / 172,950 labelled, 9 defect classes) and the dozens of student CNN repos built on it (`MD-Junayed000/…`, `juza-w/…`, `aryan-yadav4149/…`, `Anmol501/…`, and many more).

A wafer map is a **low-resolution pass/fail grid over a whole wafer** — one pixel per die. It is not microscopy, has no nanoscale structure, no scale gap, and no localization task. Zero transfer to PS-02. Everyone who searches "semiconductor dataset" lands here; we should note in the deck that we evaluated and rejected it, which demonstrates domain understanding.

Also rejected: `Multi-Class Semiconductor Wafer Image Dataset`, dicing-quality datasets, AOI datasets — all classification, none localization.

---

## 4. Layout-generation prior art — validates the vector-first plan

| Repo | Stars | Take |
|---|---|---|
| `heitzmann/gdspy` | 385 | The standard Python GDSII layout library. Proves the "describe geometry in physical units, render/export later" workflow is the industry-native one. |
| `gdsfactory` (PsiQ mirror) + `SudeepGopavaram/…GLayout` | 19 / 13 | **Parameterized cells (PCells)** — a cell defined by parameters, then instantiated and tiled. This is exactly the right shape for our DRAM/FinFET generator: one parameterized unit cell + a lattice tiling rule + a landmark injector. |
| `gdsfactory/klayout_pyxs`, `dimapu/klayout_pyxs` | 18 / 9 | KLayout cross-section scripting — how layer stacks are described. |

**Idea this unlocks:** structure our generator as *PCell → lattice tiling → landmark injection → physical sampling*, with all geometry stored in **nanometres, not pixels**. Then both the 1 nm/px and 10 nm/px views are just two different samplings of one ground truth, and the ground-truth coordinate is analytic and exact. We do not need gdspy as a dependency — we borrow the mental model and keep the code pure numpy.

---

## 5. Algorithm-side repos — for later, but two useful insights now

**Multi-scale template matching** (`Logeswaran123/…` 39★, `agiledots/…` 35★, plus ~6 clones): all do the same naive thing — loop the template over a pyramid of scales and take the best correlation.

> **Insight:** our scale factor is **known and fixed at exactly 10×**. We never need a scale search. Every one of these repos spends its compute on a search we can skip. Since **computation time is inside the 50% score**, that is a real, statable advantage — not a shortcut.

**Sub-pixel registration** — the mature end of the field:

| Repo | Stars | Note |
|---|---|---|
| `zdenyhraz/IPC` | 13 | Iterative phase correlation for high-precision sub-pixel registration |
| `mdw771/pytorch_phase_cross_correlation` | 14 | GPU/Torch port of `skimage.registration.phase_cross_correlation` |
| `michaelfsp/sggc-registration` | 18 | SGGC sub-pixel algorithm (HajiRassouliha et al.) |
| `Pyxel0524/Fourier-Mellin-Registration` | 6 | **Fourier–Mellin** — rotation + scale invariant registration (MATLAB) |
| `tony-azevedo/efficient_subpixel_registration` | 8 | The classic Guizar-Sicairos efficient sub-pixel algorithm |

> **Insight:** sub-pixel accuracy is a *solved, well-cited* problem — the literature exists and is standard. Our hard part is candidate **disambiguation** among periodic lattice peaks, not the final refinement. Confirms where the effort should go.

---

## 6. Competitor recon

- **`Asha-debug-hue/drift-sense`** — *"AI-Powered Navigation-Error Recovery for Wafer Inspection Tools"*. Same problem statement, same title. **Repository is currently empty** (git 409, no commits). Someone has claimed the name and nothing else.
- `Hardik-India/Semicon-Hackathon-2026` — empty placeholder.
- `elakkshaya/Semiconductor-Image-Restoration` — someone working **PS-01 (KLA)**, the other track.
- Several 2026 IESA DeepTech Hackathon edge-AI defect repos — different event, different problem.

**Read:** the field is at zero as of today. Nobody has shipped anything on PS-02. The prize is being first with something physically defensible, not being fastest to a template match.

---

## 7. What we carry forward — and what stays ours

**Adopted as inspiration:**
1. NIST ARTIMAGEN's *effect taxonomy* (noise / drift-distortion / edge-effect / vibration / focus) as the spine of our physics stack, with its two papers as anchor citations.
2. ARTIMAGEN's *vector-definition → render* paradigm, which independently confirms our option (b).
3. gdsfactory's *parameterized-cell + tiling* structure for the layout model.
4. Statistical calibration against real public SEM imagery (MIIC / NFFA-EUROPE) instead of asserted realism.
5. Skipping scale search entirely — the 10× ratio is known, and compute time is scored.

**Unchanged and still ours** — nothing found alters the plan:
- One vector layout in nanometres; both views sampled independently from it, each with its own PSF, dose and noise realisation. No repo found does this; ARTIMAGEN renders one view, not a matched multi-scale pair.
- Landmark saliency as a controlled, swept parameter, deliberately including near-unsolvable low-saliency cases to feed the 10% failure-analysis bucket.
- Distribution deliberately wider than Applied Materials' generator, to contain their test set.
- Exact analytic ground truth with per-pair seeds and a full parameter manifest.

**The gap in the world is the pair.** Everything public is single-view — a dataset of SEM images, or a layout tool, or a registration algorithm. Nobody publishes *matched cross-scale pairs with known correspondence*. That is precisely the object we are building, and it is why this cannot be copied from anywhere.

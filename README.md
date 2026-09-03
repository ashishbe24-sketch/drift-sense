# Drift-Sense — Sub-Pixel Site Re-Location in SEM Wafer Navigation

**PS-02 · Applied Materials · SEMICON India Hackathon 2026 (i4C / IESA)**

> Given a high-resolution **reference** image of a site and a wide, blurry,
> low-resolution **search** scan taken after the stage has drifted, recover where
> that site sits inside the search image — its centre `(x, y)`, its pose
> (`theta`, `scale`), whether it is present at all (`found`), and a confidence
> (`score`).

## ⭐ PHASE 2 SUBMISSION — START HERE

**Entry point (the exact required signature):**

```bash
python register.py --input pairs.csv --output predictions.csv
```

**Measured on the organizers' own 20-pair sample, scored against their withheld
ground truth** (never trained on, never used to tune a threshold):

| Metric | Result |
|---|---|
| **Localization** | **35.60 / 40** — Set A 1.000, Set B 0.800; 15/16 present pairs within 5 px |
| **Rejection** | **F1 0.968** (TP 15, FP 0, FN 1) |
| **Pose** | rotation **10/10**, scale **8.62/10** |
| **Calibration** | AUC 0.789 |
| **Runtime** | **3.6 s/pair median** — no GPU, no torch, no network |

Runtime was measured on the **reference machine profile, not our development box**:
4 cores, 8 GB, Python 3.11, over 200 pairs — median 3.6 s/pair, worst pair 4.1 s
against the 20 s hard timeout (4.9× margin), peak memory 176 MB of 8 GB, and
memory flat across the whole run (no growth). Output is byte-identical between
Python 3.11 and 3.12 and between the constrained and unconstrained machine, so the
scores above are exactly what the grader will compute. Failure analysis:
**[`failure_analysis.pdf`](failure_analysis.pdf)**.

> **Note on the sections below.** Everything after §1 documents the **Phase 1**
> system and its 94.5 % @5 px result. That work is the foundation this submission
> extends, and the numbers there are Phase 1 numbers measured under Phase 1
> conditions (fixed 10× zoom, no rotation, reference always present, GPU). They
> are kept for provenance — they are **not** the Phase 2 result, which is the
> table above. `infer.py`, `predict*.py` are Phase 1 / research entry points;
> **`register.py` is the Phase 2 submission entry point.**

---

## Team — "The T Guys"

| Name | Role | Email |
|---|---|---|
| **Aryan Chourasia** | Team Leader, B.Tech 3rd Year | achourasia_be24@thapar.edu |
| **Govinda Podder** | Member | gpodder_be24@thapar.edu |
| **Ashish Bajaj** | Member | kashish_be24@thapar.edu |

Thapar Institute of Engineering and Technology, Patiala.

---

## 1. The problem

A wafer-inspection tool characterises a site on one die, then must re-find *that
same site on a different die* from a wider, lower-resolution survey scan — after
thermal drift, vibration and stage backlash have thrown off its navigation.

| | Reference ("100×") | Search / wide ("10×") |
|---|---|---|
| Pixel size | 1 nm/px | 10 nm/px |
| Field of view | 1 µm | 10 µm |
| Image size | 1000 × 1000 | 1000 × 1000 |

The scale gap is exactly **10×**, so the reference's true footprint inside the
search image is exactly **100 × 100 px — 1 % of its area**. Two consequences
drive the whole design:

1. **A naive `matchTemplate` cannot even run.** The template is physically
   larger than the image it is searched in. It must be rescaled first, and the
   rescaling is where most accuracy is won or lost.
2. **The two images are different *acquisitions*, not a crop-and-resize pair.**
   The wide view is defocused, dose-starved, noisier, and may be tilted a few
   degrees. Anything that assumes pixel-wise similarity fails.

When several near-identical sites match (dense DRAM and fin arrays are periodic
by construction), the answer is the one **closest to the search-image centre** —
the stage-accuracy prior: the tool is lost, but not *that* lost.

**Metric:** Euclidean distance to ground truth, scored as accuracy within 5 px.
Computation time is a secondary tie-break.

---

## 2. Results

All figures below are measured on the **organizers' own reference generator and
test sets** (HuggingFace Space `aayushraina21/drift-sense-synthetic-data`), not
on our own data — this is the honest cross-generator test, and it is the
distribution the submission is evaluated on.

### Accuracy within 5 px

| Matcher | Default set (200 pairs) | Noisy held-out (300 pairs) |
|---|---|---|
| ZNCC baseline *(provided by organizers)* | 75.5 % | 59.0 % |
| DriftFind — classical only | 75.0 % | 56.3 % |
| DriftMatchNet — learned only | ~94 % | ~86 % |
| **DriftRoute — the submission** | **94.5 %** | **86.0 %** |

| | Default | Noisy |
|---|---|---|
| **Improvement over the provided baseline** | **+19.0 pts** | **+27.0 pts** |
| DriftRoute @ 10 px | 94.5 % | 93.3 % |

Throughput: **~150–430 ms per pair** (RTX 3050 Laptop, 4 GB). The classical
fallback path runs CPU-only at ~770 ms/pair.

### Why the classical matcher alone is not enough

DriftFind scores 75.0 % — essentially tying the ZNCC baseline. That is not a
coincidence: both are normalised-correlation methods, and both break on the
same thing. The organizers' generator applies **geometric** aberrations
(astigmatism, barrel distortion, vignette, gamma, corner rounding, linewidth
bias) that a rigid template cannot absorb. Correlation assumes the template and
the target differ by a *shift*; here they differ by a *warp*. The learned
matcher absorbs the warp because it was trained through it.

### Generalisation

The shipped network is deliberately **not** the highest scorer on the default
set. Two nets were trained:

| Checkpoint | Their default | Their noisy | Our generator |
|---|---|---|---|
| `best_theirs_domain.pt` — single-generator specialist | **95.5 %** | 84 % | 71 % |
| `best.pt` — union-trained generalist **(shipped)** | 94.5 % | **86 %** | **75 %** |

The specialist wins by 1 point on the one distribution it was trained on, and
loses on both others. Since the graders may hold back a third, unseen
configuration, the submission ships the generalist: it is trained on the union
of both generators (6 000 pairs), and it is backed by a classical fallback that
depends on no learned prior at all.

---

## 3. Quick start

```bash
git clone https://github.com/itsAryan-devop/drift-sense.git
cd drift-sense
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

### The Phase 2 submission entry point

```bash
python register.py --input pairs.csv --output predictions.csv
```

`pairs.csv` supplies `pair_id`, `reference_path`, `search_path`. Image paths are
resolved relative to the CSV's own directory and to the working directory, so the
command works from any location. `predictions.csv` contains **one row per
`pair_id`, every id exactly once**:

```
pair_id,x,y,theta,scale,found,score
p001,331.222,840.813,-0.1393,8.0,1,0.910075
p012,0.0,0.0,0.0,0.0,0,0.467089        <- found=0 writes 0 in every pose column
```

| Column | Meaning |
|---|---|
| `x, y` | match centre in search-image coordinates, sub-pixel (parabolic refinement) |
| `theta` | rotation in degrees, CCW-positive about the match centre, clamped to the disclosed [-5, +5] |
| `scale` | recovered down-scaling factor `z` in nm/px, clamped to the disclosed [8, 12] |
| `found` | 1 / 0 — thresholded on the peak NCC at 0.53 |
| `score` | **our confidence: the peak normalised cross-correlation, in [-1, 1].** Higher means more confident. It is the same quantity `found` thresholds, so the two are consistent by construction, and it is monotonic — suitable directly as the calibration ranking signal. |

**Set D (optical RGB) needs no special invocation.** Any non-grayscale input is
converted to luminance at load time (ITU-R 601) and fed to the same matcher;
grayscale inputs are passed through untouched, so Sets A/B/C are bit-identical
either way.

**It always runs.** torch is *not* a dependency — `pip install -r requirements.txt`
yields a torch-free environment, and the shipped path needs only numpy, scipy and
pillow. If torch or the checkpoint is absent the router falls back to the
classical path rather than raising. A pair whose image is missing or unreadable
still emits a valid `found=0` row rather than aborting the run.

### Other scripts (not the Phase 2 entry point)

`infer.py`, `predict.py`, `predict_net.py`, `predict_router.py` are **Phase 1**
CLIs kept for provenance; `fmt_pose.py` is a measured-and-rejected Fourier–Mellin
experiment. None are used by `register.py`.

---

## 4. How it works

The submission is **DriftRoute**: one function, two matchers, dispatched per
pair by a measured criterion.

```
                       reference (1000×1000)   search (1000×1000)
                                  │                   │
                                  └─────────┬─────────┘
                                            ▼
                              ┌─────────────────────────┐
                              │  DriftMatchNet forward  │  ~90 ms
                              │  → 250×250 heatmap      │
                              └────────────┬────────────┘
                                           ▼
                                  read the heatmap
                          ┌────────────────┴────────────────┐
                 one dominant peak                 ≥2 strong peaks
                          │                                 │
                          ▼                                 ▼
                 trust the network              hand to DriftFind
                 (plain case, ~94 %)            (multi-match specialist)
                          │                                 │
                          └────────────────┬────────────────┘
                                           ▼
                                    (x, y) sub-pixel
```

The routing rule is not a guess. The network is near-perfect on single-target
pairs and an order of magnitude faster; the classical matcher's explicit
centre-prior tie-break is stronger on repeated patterns. So the heatmap is
inspected: if a second well-separated peak reaches 60 % of the top peak
(`MULTI_RATIO`, swept on `eval200`), the pair is a repeated-pattern case and is
handed to the classical path. Otherwise the network's answer stands. **The
forward pass is reused** — routing costs nothing extra.

### 4.1 DriftFind — the classical matcher (`solve.py`)

Pure numpy/scipy, deterministic, no GPU.

1. **Rescale by block-averaging.** The reference is downsampled 10× to a
   100 × 100 stamp — averaging each 10 × 10 block, which is what the wide
   detector physically does. Not interpolation.
2. **FFT normalised cross-correlation.** NCC measures *shape* agreement
   independent of brightness and contrast, so it is unbothered by the wide
   view being darker and noisier. Computed the fast way (Lewis, 1995): the
   numerator by FFT, the local image statistics by FFT with a box kernel, so a
   100 × 100 stamp over a 1000 × 1000 image costs milliseconds.
3. **Blur and rotation search.** The wide view is defocused while the shrunk
   reference is sharp, so a sharp stamp systematically under-matches. The stamp
   is tried at 4 blur levels (0–3 px) × 5 angles (±4°), bracketing the tilt the
   evaluator described in the webinar.
4. **Coarse-to-fine.** The (blur, angle) setting is chosen on a half-resolution
   pass; only the winner is re-correlated at full resolution.
5. **Centre-prior tie-break.** Among local maxima, the winner maximises
   `NCC − λ·dist_to_centre/n_px` with `λ = 0.08`. Quality stays primary — a
   strong off-centre peak still wins — but genuine near-ties resolve toward the
   centre, which is where the stage prior says the true site lands. A hard
   "nearest-centre among near-ties" rule was tried first and was *worse*: it
   dragged correct off-centre plain matches inward.
6. **Sub-pixel refinement.** A 1-D parabola is fitted through the peak and its
   two neighbours per axis. Ground truth is fractional and scoring is at 5 px,
   so the fraction a whole-number peak leaves behind is worth recovering. The
   offset is clamped to ±1 px — a fit wanting to move further indicates a flat
   or double peak and is not trusted.

**Two optimisations worth noting.** The image side of the NCC denominator
depends only on the image and the template *shape*, never the template values,
so it is identical across all 20 (blur, angle) variants and is hoisted out of
the loop — 2–3× fewer FFTs, verified byte-identical output. Combined with the
coarse-to-fine pass this gave a 2× end-to-end speedup with no accuracy change.

### 4.2 DriftMatchNet — the learned matcher (`driftmatch/`)

A fully-convolutional **Siamese cross-correlation network with a centre-point
head** — deliberately *not* a coordinate-regression CNN, which throws away
spatial structure and is brittle on repeated patterns.

```
reference (B,1,100,100) ──encoder──► (B,C,25,25)     the feature filter
search    (B,1,1000,1000)─encoder──► (B,C,250,250)   the search field
                     depthwise cross-correlation
                              ▼
                        (B,C,250,250) response
                              ▼
              heatmap (B,1,250,250) + sub-cell offset (B,2,250,250)
```

Design decisions and the reasoning behind each:

- **Stride 4, shallow encoder.** Classification backbones (ResNet-50, VGG,
  EfficientNet) downsample 32×, which destroys the "where" we are scored on at
  5 px, and are far too heavy for a 4 GB card. A stride-4 residual encoder
  keeps the resolution and stays trainable.
- **Learned *normalised* cross-correlation.** Each channel of the reference
  filter is L2-normalised before correlation. This bounds the response — the
  same principle the classical matcher uses — and it is also what makes fp16
  autocast survive: unbounded feature correlation overflows fp16 (> 65504 → inf
  → NaN). The correlation and shrink run in float32 inside an autocast-disabled
  block for the same reason.
- **Adaptive shrinkage** — parameter-free clutter suppression. Per sample and
  channel, the response is standardised and only the part above the mean is
  kept. Background and periodic decoys produce diffuse, near-mean correlation
  and are pushed toward zero; a true match stands proud. Having no parameters,
  it cannot overfit.
- **Centre-point head** — a heatmap plus a sub-cell offset regression, so the
  stride-4 grid does not cap precision. Focal-loss prior bias (`−2.19`) on the
  heatmap, since positives are rare (1 cell in 62 500).
- **The same centre tie-break** as the classical matcher is applied to the
  heatmap peaks. The multi-match physics prior does not depend on how the
  response was produced, so it carries over unchanged to learned features.

### 4.3 Why a router instead of just the better model

Because they fail differently, and the failures are separable *at inference
time* from a signal we already computed. This is the "you can add classical
methods, you can add deep learning networks also" framing from the webinar,
taken literally: the learned component supplies robustness to warps that
correlation cannot model, and the classical component supplies an explicit,
inspectable prior for the periodic case plus a hard guarantee that the function
runs on any machine.

---

## 5. The dataset generator

`generate_dataset.py` + `driftsense/` — an SEM image-formation simulator.
**No generative image model is used anywhere**: every pixel is computed from
geometry and a documented physical model.

```bash
python generate_dataset.py --style mixed --pairs 30 --out data/mydata
```

The core design choice: **a single vector layout is defined in nanometres, and
each view is rendered by sampling that layout independently** at its own pixel
size, dose, focus and noise realisation.

```
        Layout in nm (pitch, linewidth, phase — analytic)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   sample @ 1 nm/px        sample @ 10 nm/px
   over a 1 µm window      over a 10 µm window
        ▼                       ▼
   reference                 wide search
```

This matters for two reasons:

1. **The wide view is a genuinely coarser acquisition, not a resized copy.** A
   model trained on resize-pairs learns the resize kernel, not the physics, and
   collapses on real data.
2. **The ground truth is analytic.** It comes from where the landmark was
   *placed* in nm — never from matching pixels — so labels carry zero matcher
   bias.

Everything is specified in physical units, which makes the two captures
self-consistently different: an edge effect with a 5 nm escape band is 5 px wide
in the reference and 0.5 px wide in the wide view, and it falls below resolution
there *on its own*, exactly as on a real tool.

The pipeline order follows the **NIST ARTIMAGEN** paradigm — define the sample,
then render it with defined, known amounts of each artefact: edge-effect
(secondary-electron escape), defocus, astigmatism, charging streaks, shot noise,
detector noise, line-edge roughness, vignette, gamma and barrel distortion.

Every parameter, its physical range and its literature source are recorded in
[`docs/GENERATOR_SPEC.md`](docs/GENERATOR_SPEC.md) — a **24-source citation
ledger**. Corrections made after the organizers' webinar (notably that our
stage-rotation range was 10–20× too small) are tracked separately in
[`docs/WEBINAR_CORRECTIONS.md`](docs/WEBINAR_CORRECTIONS.md), with quotes.

### The generated datasets

| Set | Pairs | Where it lives | Purpose |
|---|---|---|---|
| `curated30` | 30 | **in this repo** (`data/curated30/`) | showcase cases, annotated in `CASES.md` |
| `val_resize60` | 60 | **in this repo** (`data/val_resize60/`) | held-out resize-domain generalisation test |
| `eval200` | 200 | [release `datasets-v1`](../../releases/tag/datasets-v1) | primary evaluation set |
| `train` | 4 000 | [release `datasets-v1`](../../releases/tag/datasets-v1) (3 parts) | training pool |

Every archive unpacks to `images/` (1000 × 1000 PNG pairs), `labels.csv`
(ground-truth coordinates plus every sampled parameter), `seeds.txt` and
`dataset_meta.json`.

```bash
# fetch the full sets from the release
gh release download datasets-v1 --repo itsAryan-devop/drift-sense --dir data/
# the train set is split across three parts; unzip all of them into data/
```

The two smaller sets are committed directly so the repo is runnable and
inspectable the moment it is cloned; the two large ones are release assets
because a 4.8 GB git history would make the repo unusable to clone.

### Reproducibility

The generator is deterministic, so none of the above is strictly necessary —
`eval200`, `train` and `val_resize60` each regenerate **byte-identically** from
the seed list tracked in git beside them:

```bash
python generate_dataset.py --seeds-file data/eval200/seeds.txt --out data/eval200
python generate_dataset.py --seeds-file data/val_resize60/seeds.txt --render-mode resize --workers 1 --out data/val_resize60
```

`curated30` is the one exception, by construction: it is three 6-step *ladders*
plus 12 individual cases, so several pairs share one base seed and differ only
in the single variable the ladder sweeps. Its 30 pairs rebuild from the 12
tracked seeds through the script that composes them:

```bash
python scripts/make_curated30.py --out data/curated30
```

That is what keeps every number in this README auditable: the seeds, manifests
and metadata are version-controlled, so a reviewer can rebuild the exact pixels
any figure was computed from.

---

## 6. Repository layout

```
register.py                     ★ PHASE 2 SUBMISSION ENTRY POINT
                                  --input pairs.csv --output predictions.csv
failure_analysis.pdf            ★ Phase 2 failure analysis (2 pages)
route.py                          DriftRoute: the router (classical + learned)
solve.py                          DriftFind: classical matcher core — supplies all
                                  six Phase 2 columns in the shipped configuration
generate_dataset.py               dataset generator (standalone CLI)
requirements.txt                  numpy / pillow / scipy only (torch NOT required)

infer.py                          Phase 1 entry point (superseded by register.py)
predict.py                        Phase 1 CLI — classical only
predict_net.py                    Phase 1 CLI — learned only
predict_router.py                 Phase 1 CLI — router, batch mode
fmt_pose.py                       Fourier-Mellin experiment — measured and rejected

driftmatch/                     the learned matcher
  model.py                        Siamese encoder + learned NCC + centre-point head
  data.py                         dataset, memmap cache, augmentation
  train.py                        training loop (fp16, chunked, resumable)
  infer.py                        heatmap → (x, y)
  checkpoints/
    best.pt                     ★ shipped generalist  (94.5 % / 86 %)
    best_theirs_domain.pt         single-generator specialist (95.5 % / 84 %)
    best_generalist.pt            earlier generalist (epoch 18)
    best_ourdomain.pt             our-generator-only net (ablation)

driftsense/                     the nm-scale SEM renderer
  layout.py                       vector layout in nanometres
  raster.py                       anti-aliased sampling onto any pixel grid
  physics.py                      SEM image-formation model
  sampling.py                     seed → fully-specified pair

scripts/                        evaluation, verification, calibration (24 utilities)
docs/                           problem analysis, generator spec, webinar notes
data/
  curated30/                      30 showcase cases + CASES.md  (images included)
  val_resize60/                   held-out resize-domain set    (images included)
  eval200/                        manifest + seeds  (images: release datasets-v1)
  train/                          manifest + seeds  (4 000 pairs; ditto, 3 parts)
```

Full image sets for `eval200` and `train` are published as **release
[`datasets-v1`](../../releases/tag/datasets-v1)** — see §5.

---

## 7. Reproducing the numbers

```bash
# classical
python scripts/eval_solver.py data/eval200

# learned
python scripts/eval_net.py    data/curated30 driftmatch/checkpoints/best.pt

# router (the submission)
python scripts/eval_router.py data/curated30 driftmatch/checkpoints/best.pt
```

To reproduce the headline cross-generator figures, clone the organizers'
Space, generate their test sets, and evaluate against their manifest:

```bash
git clone https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data ds_ref
pip install opencv-python-headless
python ds_ref/generate_dataset.py --seed 7000 --pairs 200 --out ds_ref/refdata/test
python scripts/eval_manifest.py ds_ref/refdata/test/manifest.csv
```

## 8. Training

```bash
python -m driftmatch.train \
    --data  uniondata \
    --out   driftmatch/checkpoints \
    --epochs 12 --batch 4 --workers 0 \
    --eval1 refdata/test --eval2 rrdata/heldout
```

The shipped checkpoint was trained on a **union of 6 000 pairs** — 4 000 from
the organizers' generator with randomised aberrations, plus 2 000 from ours —
for 30 epochs, keeping the best checkpoint by score on the *noisy held-out* set
(not the default set, to avoid selecting for the easy case).

**Constraints that shaped the training loop**, all learned the hard way on a
16 GB / 4 GB-VRAM laptop:

- `--workers 0` — DataLoader worker processes trigger CUDA OOM on Windows.
- `--batch 4`, fp16 autocast + GradScaler + gradient clipping — the ceiling for
  a 4 GB card.
- The sample cache is a **disk memmap**, not a RAM array. A RAM cache is faster
  but fragments the heap and dies around epoch 16 on pools above ~1 000 pairs.
- Training is **chunked and resumable** (`--resume`): a fixed number of epochs
  per process, then the process restarts with fresh memory. This is what made a
  30-epoch run on 6 000 pairs finish at all.

Held-out accuracy plateaued around 88 % by epoch 30, which is why training was
stopped there — the curve had converged, and pushing to 80 epochs would have
bought overfitting rather than accuracy.

---

## 9. Failure cases and honest limitations

Where this still breaks, and why:

1. **Dense periodic arrays with a weak unique landmark (~6 % of the default
   set).** When every candidate site is genuinely near-identical, the centre
   prior is the *only* discriminating signal, and it is a prior, not evidence.
   If the true site happens to sit far from the search-image centre while a
   decoy sits near it, the answer is confidently wrong. This is the dominant
   residual error mode.
2. **Severe noise (the 300-pair noisy set).** Accuracy is 86.0 % @5 px but
   93.3 % @10 px — meaning most remaining failures are *near-misses*, not
   catastrophic mislocations. The site is found; the sub-pixel refinement is
   swamped by noise.
3. **Geometric warps beyond the trained range.** Barrel distortion and
   astigmatism are absorbed because the network saw them in training. A warp
   materially stronger than the generator's range would degrade the learned
   path — and the classical fallback is *worse* there, not better, since
   correlation cannot model warps at all.
4. **The classical path is the weak link on the organizers' distribution**
   (75.0 % / 56.3 %). It is retained for the multi-match case, for machines
   without a GPU, and as a guarantee against total failure — not because it is
   competitive on its own.
5. **A third, unseen generator is the real open risk.** Our own-generator net
   scored only 71 % on the organizers' data before union training — the
   cross-generator gap is real and was measured, not assumed. Union training
   plus the classical fallback are hedges against it, not proofs.
6. **RGB optical images (the +10 bonus) are not implemented.** The renderer and
   both matchers are grayscale-only. This was a deliberate scope decision, not
   an oversight.

---

## 10. References

Principal sources; the full 24-source ledger with per-parameter attribution is
in [`docs/GENERATOR_SPEC.md`](docs/GENERATOR_SPEC.md).

- J. P. Lewis, *Fast Normalized Cross-Correlation*, Vision Interface, 1995.
- P. Cizmar et al., *Simulated SEM images for resolution measurement*
  (ARTIMAGEN), Scanning, 2008 — NIST artefact-rendering paradigm.
- L. Bertinetto et al., *Fully-Convolutional Siamese Networks for Object
  Tracking*, ECCV Workshops, 2016 — the cross-correlation matching formulation.
- B. Li et al., *SiamRPN++*, CVPR 2019 — depthwise correlation for per-sample
  filters.
- X. Zhou et al., *Objects as Points* (CenterNet), 2019 — centre-point heatmap +
  sub-cell offset head.
- T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 — the
  rare-positive prior bias.
- L. Huang et al., *Joint Anomaly Detection and Inpainting for Microscopy Images
  via Deep Self-Supervised Learning*, ICIP 2021 — the MIIC dataset, used **only**
  to measure calibration statistics. Not redistributed here; see
  [`docs/SEM_CALIBRATION_AND_LITERATURE.md`](docs/SEM_CALIBRATION_AND_LITERATURE.md).

Reference generator and the ZNCC baseline compared against throughout:
`https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data`

---

## 11. Environment

Developed on Windows 11, Ryzen 5 7535HS, RTX 3050 Laptop (4 GB VRAM), 16 GB RAM.

| Path | Requirements |
|---|---|
| Classical + generator | numpy 2.5.1, scipy 1.18.0, pillow 12.3.0 — **no GPU** |
| Learned + router | additionally torch ≥ 2.4 (built against 2.5.1+cu121) |

A CPU-only torch build works and is slower; with no torch at all, `infer.py`
still runs via the classical path.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

The MIIC dataset referenced in §10 is **not** covered by this license, is not
redistributed in this repository, and remains subject to its own
non-commercial research terms.

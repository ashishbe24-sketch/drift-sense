# SEMICON India Hackathon 2026 — PS-02 (Applied Materials)
## "Drift-Sense: Finding a Needle in a Nanoscale Haystack" (Navigation-Error Recovery)

**Status:** Understanding phase only. No solution design in this document.
**Compiled:** 1 Aug 2026. Sources: the Applied Materials PS deck (10 slides) + https://i4c.in/hackathon-2026/

---

## 1. The event (facts, with dates)

| Item | Value |
|---|---|
| Event | SEMICON India Hackathon 2026 — AI Chip Design & Semiconductor Innovation Challenge |
| Organizer | SEMI India; Strategic partner IESA; Implementation partner i4C; Academic partner VIT |
| Industry partners | KLA (Track 1 / PS-01), **Applied Materials (Track 2 / PS-02)** |
| Eligibility | B.E./B.Tech/M.Tech/MCA/PhD, any stream, **teams of 2–4** |
| Registration opened | 24 Jul 2026 |
| Webinar — Applied Materials PS | 31 Jul 2026 (**already past — get the recording**) |
| Technical Knowledge Session | 6 Aug 2026 |
| Solution Development Guidance | 8 Aug 2026 |
| **Round-1 submission + registration deadline** | **16 Aug 2026** |
| Round-1 evaluation | 17–26 Aug 2026 |
| Top 30 announced | 27 Aug 2026 |
| Round-2 submission | 4 Sep 2026 |
| Top 10 announced | 6 Sep 2026 |
| Finalist mentoring | 7–12 Sep 2026 |
| Grand finale / winners | 17 / 18 Sep 2026, at SEMICON India 2026 |
| Prize pool | ₹5,00,000 across both tracks |
| Contact | support@i4c.in, +91 98504 58254 |

**~15 days to the Round-1 deadline.**

### Facts to verify (conflicting or unconfirmed)
- **Finale venue.** The i4c hackathon page says Yashobhoomi (IICC), Dwarka, New Delhi. A separate press article describes an i4C "DeepTech Hackathon" finale at IESA Vision Summit, Bengaluru. Different events — but confirm on the registration portal.
- **PDF file-naming for Track 2.** The only stated example is `TeamName_KLA_PS01.pdf`. Assume `TeamName_AppliedMaterials_PS02.pdf` but confirm in the official Idea Submission Template.
- **Rubric weights sum to 90%** (see §5). Ask what the remaining 10% is.
- Applied Materials said they will later release **(a) a starter GitHub repo, (b) the test dataset, (c) the detailed evaluation rubric**. Watch mail/WhatsApp for these; they change the game.

---

## 2. The physical problem (why this exists)

A wafer-inspection / review tool characterises a site on one die. Later it must return to the **same site on a different die**. Dies are near-identical copies, so "go to die (i,j), offset (x,y)" *should* work — but the motion stage accumulates **thermal drift, vibration, mechanical slack**. The tool lands slightly off target. Nobody can see the error directly; the tool must **re-find the site from imagery**.

The recovery procedure the PS models:
1. Take a **wide, low-magnification** image around where the tool thinks it landed.
2. Search inside it for the **high-magnification reference** image of the site captured earlier.
3. Report where it actually is → the offset feeds back as a stage correction.

That is the entire task: **search-and-localise across a 10× scale gap, in a scene that is deliberately, densely periodic.**

---

## 3. Exact problem specification (from the deck, verbatim intent)

**Inputs — two grayscale images, both exactly 1000×1000 px:**

| | Reference ("100x") | Wide search ("10x") |
|---|---|---|
| Pixel size | **1 nm/px** | **10 nm/px** |
| Field of view | 1 µm × 1 µm | ~10 µm × 10 µm |
| Content | the target site, high detail | a larger area containing that site |

**The single most important consequence:**
The reference covers 1 µm. The search image is 10 nm/px. So the reference's physical footprint occupies exactly

> **100 × 100 pixels inside the 1000×1000 search image — 1% of its area.**

A naive `cv2.matchTemplate(search, reference)` cannot even run (template ≥ image). Correctly handling the 10× ratio is called out explicitly in the deck; it is the first filter on submissions.

**Outputs:**
1. Find the reference pattern inside the search image.
2. Report the **center (x, y)** of the first matching tile.
3. **If more than one region matches, return the one closest to the search image's center.** ← a stated, deterministic tie-break rule. Implement it as a rule; do not just take `argmax`.

**Data:** *"Due IP constraints, No dataset is provided; participants shall generate their own synthetic image-pairs."* You build both the data and the solver. Applied Materials evaluates on **their own** synthetic test set.

---

## 4. What actually makes this hard (critical read)

### 4.1 The 10× gap is not a resize
The obvious move — downscale the reference 1000→100 and template-match — is necessary but is *not* physically what happens. At 1 nm/px you see contact dots, fin edges, corner rounding. Decimating 10× pushes almost all of that structure past Nyquist: it aliases or vanishes. Meanwhile the real 10 nm/px capture is a *physically coarser acquisition* — larger probe/PSF, different dose, different noise realisation — not a clean mathematical average of the fine image.

So there is a **domain gap between "reference resized by 10×" and "reference as it truly appears in the wide image."** How you model that gap is most of the engineering. (Note: Applied Materials' own starter prompt builds the wide image by rendering a bigger layout and resizing it down — so in *their* data the gap is smaller than in reality. But their FAQ warns the test wide-search images will be **noisier** than your training data. Don't fit tightly to a clean resize model.)

### 4.2 Periodicity is the real adversary — this is a disambiguation problem, not a detection problem
DRAM (word-line / bit-line / contact arrays) and FinFET (parallel fins + gate bars) layouts are *designed* to be perfectly periodic. Correlate a periodic template against a periodic scene and the score surface is not one peak — it is a **lattice of near-identical peaks**, one per unit cell. At 10 nm/px with realistic pitches, dozens to hundreds of positions score within noise of each other.

The deck says this outright: the repeating structures are *"what makes this genuinely hard rather than a simple exact-pixel lookup."*

### 4.3 The disambiguating signal is the non-periodic landmark
Look at both example figures in the deck: each reference is a periodic array **with one aperiodic feature in the middle** — a large cross/pad in Example 1, a dark square block in Example 2. That anomaly is the only thing that breaks the translational symmetry. Two implications:
- **Your generator must place such landmarks** (and must vary how salient they are), or your test cases are unsolvable-by-construction and your reported accuracy is meaningless.
- **Your matcher should weight aperiodic energy** rather than treating all pixels equally — the periodic background contributes ambiguity, not information.

### 4.4 Noise asymmetry
*"Independent grayscale sensor noise on each image (don't reuse the same noise on both — they're two separate captures)."* Two separate acquisitions. So no algorithm may assume correlated noise, and the wide image is the noisier of the two — the low-SNR side is also the low-resolution side.

### 4.5 Sub-pixel tolerance
The success criterion is *"within subpixel of the true downsampled location."* One search-image pixel = 10 nm physically, so "sub-pixel" ≈ few-nanometre accuracy. Integer-argmax template matching cannot reach this; sub-pixel refinement (parabolic/centroid fit on the score surface, phase correlation, or a learned offset head) is required. Because the wording is loose, report a **tolerance curve** (≤0.5 px / ≤1 / ≤2 / ≤5 px) instead of one number — it is more honest and it protects you if their rubric uses a different threshold.

### 4.6 Rotation/distortion is implied, not specified
The deck's "Expected Solution" asks you to justify *"distortion, rotation & scaling"* choices. So small in-plane rotation and scale error are expected in your dataset. A pure translation-only matcher is likely to be penalised on their harder cases.

---

## 5. How you are scored (the strategic core)

From the FAQ slide, verbatim weights:

| Weight | Criterion |
|---|---|
| **50%** | Inference results — correct coordinates on **their** test data, *includes computation time* |
| **30%** | **Augmentation code** that produces real-like SEM images of FinFET/DRAM stacks, **based on literature study** |
| **10%** | **Root-cause / explainability on failure cases** |
| Bonus | Same method working on **RGB optical-microscope** images |
| *(10% unaccounted — ask)* | |

### Read this carefully
**40% of the marks are not about accuracy at all.** They are about (a) physically defensible synthetic data with citations, and (b) honestly explaining where and why your method fails. Most teams will spend 95% of their effort on a model and lose those 40 marks. That asymmetry is the single biggest strategic fact in this PS.

The other consequence: because the 50% is graded on *Applied Materials' generator*, not yours, a deep net overfitted to your own generator is a **risk**, not an advantage. Whatever is built must be robust across generators. Note the deck permits *"classical ML or DL-based"* — classical methods are explicitly welcome and generalise across domain gaps far better.

### Hard requirements attached to scoring
- **Justify every augmentation/noise/distortion/rotation/scaling choice against ≥2–3 credible public sources** (papers, textbooks, or patents on device structure or SEM imaging), cited in the final presentation.
- **Only publicly known structural characteristics.** Never proprietary fab data. (Applied Materials is protecting itself; a deck citing real fab numbers is a liability, not a flex.)
- Report **computation time on a single 1k×1k pair**.
- Run **≥30 randomized generated test cases**, report % within a stated tolerance.
- Give **at least one honest failure example** (their suggestion: inside a highly periodic array region) **and why**.

---

## 6. Submission requirements (mandatory checklist)

**A. Presentation** — official Idea Submission Template, **PDF only**, 8–9 slides:
team details · problem statement & relevance · idea description · proposed solution (architecture, training strategy, augmentation, pipeline diagram) · innovation & uniqueness · results (metrics + before/after visuals) · technology & feasibility (stack, hardware, train time, inference time, model size) · GitHub + video links · **references and citations**.

**B. Public GitHub repository (mandatory), Track-2 contents:**
1. `README.md` with complete setup instructions
2. **Dataset generator — standalone `.py`**, parameterised by *architecture style, number of pairs, output directory*, and it must **record ground-truth coordinates**
3. **Localization inference — standalone `.py`**, takes *reference image path + search image path*, outputs predicted **(x, y)**
4. Model weights (if DL), downloadable
5. Training script (`.py` or `.ipynb`, if DL)
6. `requirements.txt` from `pip freeze`
7. Citation / supporting-reference documents (PDF or markdown)

**C. Zip** (per the deck's FAQ) containing the frozen env, the documented generator, the documented inference file, DL notebook if used, and supporting citation docs.

**D. Video** — optional, recommended, ≤5 min.

**Hard rules:**
- **Python only.** C++ and single-notebook submissions are explicitly refused: *"It will be difficult to evaluate different languages."*
- Scripts must run **as-is on a fresh machine, without manual edits**. The organisers state plainly that a script that doesn't run cannot be benchmarked, and unscored submissions cannot win. Treat reproducibility as a scored deliverable, not hygiene.

---

## 7. Literature you will have to cite (the 30% bucket)

Every one of these needs 2–3 public sources. Topics to pin down:

**SEM image formation**
- Shot / Poisson noise from finite electron counts; SNR ∝ √(dose) — Reimer, *Scanning Electron Microscopy*; Goldstein et al., *Scanning Electron Microscopy and X-Ray Microanalysis*
- Detector/read Gaussian noise, and why the wide low-dose scan is noisier
- **Edge brightening**: secondary-electron yield rises at feature edges because the escape volume intersects more surface — the deck explicitly asks for this effect
- Probe size / Gaussian PSF → blur kernel; why the 10 nm/px capture is genuinely blurrier
- Charging artifacts, scan/raster drift distortion (shear along the slow-scan axis), horizontal scan-line streaking

**Device structure (public sources only)**
- DRAM: word lines, bit lines, contacts/vias, 6F² cell layout; pitch values from IRDS / published roadmaps
- FinFET: fin pitch, gate pitch, fin/gate dimensions from IRDS or published papers
- Line-edge roughness (LER) and line-width roughness (LWR) — statistical models and typical magnitudes

**Metrology / motion**
- Thermal drift and vibration in precision stages; why die-to-die navigation error exists at all (this is the PS premise — cite it in the problem-relevance slide)

Keep a `references.md` in the repo from day one, with a one-line justification per augmentation → citation. That file *is* 30% of the marks.

---

## 8. Risks, ambiguities, open questions

1. **Domain gap is risk #1.** They test on their generator. Mitigation principle: make your generator strictly *broader* than theirs (pitches, orientations, noise levels, blur, contrast, landmark saliency) so their distribution sits inside yours.
2. **Tie-break rule must be explicit.** "Closest to search-image center" is a scoring rule, not a heuristic. Any candidate-ranking stage must apply it deliberately.
3. **"First matching tile"** (slide 4, point 2) vs "closest to center" (point 3) — ordering language is loose. Safe reading: enumerate all matches above threshold, then select the one nearest the search image's center. Worth asking the organisers.
4. **Tolerance definition is unstated** — hence report a curve.
5. **Compute time is inside the 50%.** Speed is scored, not just accuracy. A heavy multi-scale exhaustive search will cost marks even if it is accurate.
6. **The later drops** (starter repo, test set, detailed rubric) mean the plan must include a re-tuning window after they land.
7. **Team of 2–4 required** — confirm the roster before 16 Aug.
8. **Bonus RGB optical** is explicitly conditional: *"provided the core SEM-based solution is completed first."* Do not touch it until the grayscale case is finished.
9. **Both webinars are past** (30 & 31 Jul). Get the PS-02 recording — organisers said explanation videos are available.

---

## 9. One-line statement of the problem

> Given a 1000×1000 @1 nm/px SEM reference of a semiconductor site and a 1000×1000 @10 nm/px noisier wide SEM view containing that site as a 100×100-pixel footprint inside a densely periodic die layout, return the sub-pixel center (x, y) of the site in the wide view — fast, on synthetic data you must generate and physically justify yourself, choosing the center-most candidate when the layout's periodicity makes several positions match.

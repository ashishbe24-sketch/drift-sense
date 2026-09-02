# FINAL SESSION HANDOFF — DriftSense Phase 2, PS-02 Applied Materials
### Written 2 Sep 2026. Submission due **3 Sep 2026, 23:59**. Read this file completely before acting.

You are picking up a project that is **technically complete and validated against the organizers'
real data**. The remaining work is almost entirely non-technical packaging. Your job in this session
is to verify everything end-to-end, produce the missing deliverables, and ship. Do not start
new algorithmic work unless this document explicitly flags it as open.

> **UPDATE 2 Sep — read [`docs/ORGANIZER_MATERIALS_DIGEST.md`](ORGANIZER_MATERIALS_DIGEST.md) next.**
> All the mentor-supplied Phase 2 material was re-read end-to-end, including the dataset-generator
> prompt, which had never been distilled into these docs. It closes three open risks, independently
> vindicates two of our decisions, **invalidates one section of the failure-analysis draft**, and
> **adds one deliverable (`README.md`) that this document does not list**. Corrections are
> inlined below and marked `[2 Sep]`.

---

## 0. THE 60-SECOND SUMMARY

| | |
|---|---|
| **Problem** | PS-02 "Drift-Sense", SEMICON India Hackathon 2026, Applied Materials |
| **Team** | "The T Guys" — Aryan Chourasia (lead), Govinda Podder, Ashish Bajaj + Devaansh Gupta (teammate doing GPU work) |
| **Repo** | https://github.com/itsAryan-devop/drift-sense (public), branch `main`, HEAD `e847f17` |
| **Current real-data score** | **Localization 35.60/40 (89%)**, rejection F1 0.968, pose theta 10/10, **scale 8.62/10** |
| **Shipped algorithm** | Classical NCC (`solve.py`) supplies x,y AND theta/scale/found/score. The CNN exists but is **off by default**. |
| **Runtime** | 1.5–2.9 s/pair CPU-only (budget: 5 s median, 20 s hard timeout) |
| **STILL MISSING** | **Phase 2 PPT, failure_analysis.pdf (≤2 pages), demo video, final zip** — all at ZERO |

**The single most important fact:** everything technical has been validated against the organizers'
real 20-pair sample. The risk to qualification is now *missing deliverables*, not model quality.

---

## 1. THE PROBLEM (Phase 2 spec)

Given two 1000×1000 grayscale images — a **reference** (1 nm/px, 1 µm FOV) and a **search/wide**
image (~8–12 nm/px) — locate the reference's site inside the wide image.

**What Phase 2 changed from Phase 1:**

| Axis | Phase 1 | Phase 2 |
|---|---|---|
| Zoom | exactly 10× | **unknown, uniform [8,12]** per pair |
| Rotation | 1–3° noise, unreported | **unknown ±5°, must be reported** |
| Presence | always present | **~20% absent** (Set C) |
| Output | `x,y` | **`x, y, theta, scale, found, score`** |

**Mandatory entry point (literal, not illustrative):**
```
python register.py --input pairs.csv --output predictions.csv
```
`predictions.csv` — one row per `pair_id`, **every pair_id exactly once; a missing row scores zero**.
When `found=0`, write 0 in all pose columns.

**Reference machine:** 4-core x86, 8 GB RAM, **no GPU, no network**, Python 3.11, weights ship inside
the zip. **Runtime: median ≤5 s/pair, hard timeout 20 s = that pair scores zero.**

**Blind test set:** 200 pairs — Set A nominal (70), Set B degraded (70, severity 1–4), Set C absent
(40), Set D optical RGB bonus (20). We never see these.

### Scoring (100 + 10 bonus)

| Pts | Criterion | Detail |
|---|---|---|
| 40 | Localization | Tiered credit: ≤1px=1.00, ≤2px=0.80, ≤3px=0.60, ≤5px=0.40, >5px=0. **Total = 0.45·A + 0.55·B** |
| 20 | Pose | scale 10 + rotation 10. Scored **only where localization credit > 0**. Scale tiers ≤1%/2%/5% → 1.00/0.60/0.30. Theta tiers ≤0.25°/0.5°/1.0° → 1.00/0.60/0.30 |
| 15 | Rejection | F1 on `found` across A+B+C. **Never rejecting scores zero here** |
| 10 | Calibration | AUC of `score` vs per-pair correctness |
| 5 | Efficiency | Relative quartile ranking on median wall-clock. **Exceeding 5s costs ranking, does NOT zero anything** |
| 10 | Generator + citations + failure analysis | Re-judged under Phase 2 conditions |
| +10 | Bonus | +6 Set D (if A–C ≥0.50); +4 if rejection F1 ≥ 0.90 — we are at 0.968, already clear |

**[2 Sep] The bonus is a TIE-BREAKER, not points.** Mentor, verbatim: *"Does the bonus change the
ranking? No, no, it doesn't… it cannot lift a team above 100 for ranking. It's just the best second
tiebreaker after maybe set B credit."* We already clear the F1 gate — nothing to chase. Do not
present it in any deliverable as if it lifts the score.

**[2 Sep] Rejection F1 is computed over the 180 grayscale pairs (A+B+C); Set D is excluded.** FP and
FN weigh equally; the jury breaks them out separately and may use which way a team leans as a
tiebreak.

Only **5 teams per problem statement** advance from the 15 shortlisted.

### DISQUALIFYING — no appeal
- Network access during the scored run
- Hard-coding, filename fingerprinting, reading outside supplied paths
- **A method materially different from our declared Phase 1 approach**
- **Mixing organizer test/validation data into training, in either direction**

---

## 2. CURRENT ARCHITECTURE — read this carefully, it is subtle

**DriftRoute** (`route.py`) is the declared architecture: classical `solve.py` + learned
`driftmatch/` (DriftMatchNet CNN), dispatched by a router. **This is still what we declare, and the
mentor explicitly confirmed extending it is compliant** (see §7).

**But what actually runs on the grading machine is classical-only.** Precisely:

- `route.predict_full(..., use_net_xy=False)` — **default False**. Classical supplies `x, y`.
- Classical *always* supplies `theta`, `scale`, `found`, `score` (the net has no pose head).
- `requirements.txt` ships **torch commented out / optional**, so on the reference machine torch is
  absent, `route.load_net()` catches the ImportError, and the net never loads. Verified in a clean
  torch-free venv: exit 0, valid 20-row output, byte-identical to the torch-present run.

**How to describe this honestly in the PPT/PDF** (do NOT imply a hybrid runs on the graded box):
> "DriftRoute retains both components. Validation against the organizers' data showed the classical
> matcher generalises better on this domain, so it carries localisation in the shipped configuration;
> the learned matcher remains available (`use_net_xy`) for GPU-equipped deployments."

**Key constants (current, validated):**
- `route.py:145` — `FOUND_PEAK = 0.53`
- `route.py:173` — `use_net_xy=False` (default)
- `solve.py` — `PHASE2_SCALES` = 8.0→12.0 step 0.5 (9), `PHASE2_ANGLES` = −5→+5 step 2.5 (5),
  `SCAN_ANGLES = PHASE2_ANGLES` (the p008 fix), `THETA_SIGN = +1.0`
- `register.py:130` — ships `driftmatch/checkpoints/best_phase2_speckle.pt`

---

## 3. THE REAL NUMBERS (organizers' 20-pair sample — the only ones that count)

Everything else in this project is self-graded on our own generator and has historically been
**misleading**. These are measured against their withheld `ground_truth.csv`.

| Metric | Value |
|---|---|
| **Localization** | **35.60 / 40** (Set A mean credit **1.000**, Set B **0.800**) |
| Present pairs within 5px | **15 / 16** |
| **Rejection F1 (present+)** | **0.968** (TP 15, FP 0, FN 1) — clears the +4 bonus gate (a tiebreaker, not points) |
| Pose — theta | **10 / 10** (sign confirmed against their ground truth) |
| Pose — scale | **8.62 / 10** (was 8.00 before the 2 Sep reported-pose clamp) |
| Calibration AUC | 0.725–0.789 |
| Runtime (classical, CPU) | 1462 ms (teammate's box) / 2843 ms (Aryan's box) median |

**How we got here (this is the story for the failure analysis):**

| Config | Localization /40 |
|---|---|
| Original shipped (net x,y, FOUND_PEAK 0.68) | **13.35** |
| + classical x,y instead of net | 23.1 |
| + FOUND_PEAK → 0.53 | 29.7 |
| + p008 scale-scan fix | **35.60** |

### HONEST CAVEATS — carry these into every claim
1. **20 pairs is a tiny sample.** 16 present pairs → one pair ≈ 6 pp. Only large gaps are meaningful.
2. **Their README says the real 200-set is harder** (Set B skews to severity 3–4). Set B is the
   heavier-weighted bucket (55%) and already our weaker one (0.800 vs Set A's 1.000).
   **Expect the real score LOWER than 89%.** Do not plan around 89%.
3. Our own-generator numbers (82–84% historically) were **not predictive** — they were 31% on real
   data before the fixes. Never quote self-generated numbers as real performance.

---

## 4. COMPLETE WORK HISTORY

### Phase 1 (context)
Built physics-based SEM generator (`driftsense/`), classical matcher (`solve.py`), CNN
(`driftmatch/`), router (`route.py`). Scored 94.5% on their Phase 1 generator vs 75.5% ZNCC
baseline. Ranked 8/15, advanced.

### Phase 2 — by Aryan (this machine, D:\semicon)
| Work | Result |
|---|---|
| Scale search [8,12] + golden-section refine | Median scale error ~0.6–0.9% |
| Rotation/theta recovery + `THETA_SIGN` fix | Median 0.19–0.29°; sign was backwards, caught by testing |
| Absent-pair generation + rejection threshold | Original `found` mechanism |
| Full aberration suite in generator | astigmatism, barrel, vignette, gamma, scan distortion |
| `register.py` 6-column contract | Validated end-to-end |
| **Fourier-Mellin estimator** (`fmt_pose.py`) | **Built, measured, REJECTED** — scale err 90% vs grid's 1.1%. Root cause: reference's 1 µm FOV shows only 3–14 lattice periods, too few for a usable spectral peak. Do not revisit. |
| **Competitor recon** | Read all 9 other teams' public repos. Most frozen pre-Phase-2. Findings in §8. |
| **AMP-generator retrain experiment** | See below — proved the overfit diagnosis, classical still won |

### Phase 2 — by Devaansh (teammate, separate box, RTX 3050 6GB)
| Commit | Work | Result |
|---|---|---|
| `7d9089e` | Rotation-aware retrain, 8k pairs | net 83→84%, adopted |
| `0dc42e0` | Epoch-selection verification | epoch 2 confirmed on 100/150 pairs, kept |
| `1b153d4` | **Set C absent-pair generator bug fix** | Absent peak median 0.822→0.533; rejection F1 0.876→0.939. **Reversed the "separability ceiling" conclusion — it was our bug** |
| `15e7478` | Speckle + salt-and-pepper noise categories | +5pp on noisy eval, adopted |
| `61f70bf` | **Classical x,y + FOUND_PEAK 0.53** | **13.3 → 29.7 /40 — the single biggest win** |
| `eda08fb`,`0ae7fe9` | requirements.txt + torch-free clean-venv verification | Proven runs with numpy/pillow/scipy only |
| `6cf0fec` | **p008 fix: `SCAN_ANGLES = PHASE2_ANGLES`** | 29.7 → **35.60/40**, Set A now perfect |
| `79af962` | Set C decoy pitch fidelity | absent median 0.507→0.371, matches their ~0.40 |
| `1031c44`,`e99ecf1` | CPU-latency benchmark | classical 1.46s median; net-on-CPU 2.02s |
| `d28a9c3` | Set B severity ladder | own-data recalibration now *supports* 0.53 |

### The AMP retrain experiment (`e847f17`, completed 2 Sep, NOT adopted)
Ran **their own Phase 2 generator source** (shared in the SharePoint folder) with new seeds and
randomised poses to make 2,996 training pairs, fine-tuned the CNN, evaluated.

| Config | Localization /40 | within 5px |
|---|---|---|
| classical (shipped) | **35.60** | 15/16 |
| net trained on OUR generator | 13.35 | 7/16 |
| net trained on THEIR generator | 28.78 | 15/16 |

**Proved the overfit diagnosis** (13.35→28.78 from a distribution change alone) but **classical still
won** — the net matches on recall (15/16) and loses on *precision* (Set A 0.825 vs 1.000: lands
inside 5px but outside 1px, forfeiting tier credit). **Not adopted. Nothing shipped changed.**
Training curve was healthy (held-out rose then plateaued, no overfit signature).

---

## 5. WHAT REMAINS — THIS IS YOUR JOB

### 5.1 MANDATORY DELIVERABLES (deadline 3 Sep 23:59)

> **CORRECTED 2 Sep after reading the official `Applied Materials_Phase 2_Task.pptx` (slide 5).**
> An earlier draft of this file listed a Phase 2 PPT and a demo video as mandatory. **They are not.**
> Slide 5 ("ALSO IN THE ZIP") lists exactly: `requirements.txt` from pip freeze, `generate_dataset.py`
> documented, and `failure_analysis.pdf` (max 2 pages) — plus the `register.py` entry point and
> weights shipping inside the zip. Slide 9 carries forward the Phase 1 code rules only ("Python only,
> zip with a pip freeze environment, documented generator with cited sources").
>
> **Status: only TWO items are actually outstanding — `failure_analysis.pdf` and the zip assembly.**
> `register.py`, `requirements.txt`, the documented generator and the 30-source citation ledger
> (`docs/GENERATOR_SPEC.md`) are all done.
>
> *Caveat:* the submission **portal** may separately request a deck/PDF the way Phase 1's idea
> submission did. The task deck does not require one. Verify on the portal before building a PPT.

1. **`failure_analysis.pdf`, max 2 pages.** Source material is rich — use
   `docs/PHASE2_FAILURE_ANALYSIS_DRAFT.md` plus these genuinely strong, *measured* findings:
   - **The overfit proof** (AMP experiment): removed the suspected cause, net's real-data score more
     than doubled, classical still won on sub-pixel precision. This is the headline.
   - **p008**: our matcher failed a pair their *naive ZNCC baseline* solved (credit 1.00). Root-caused
     to the scale-scan ranking at angle 0 while the pipeline searched ±5°. Fixed, +5.9/40.
   - **Set C bug**: absent decoys shared the reference's exact lattice → artificially high peaks. We
     had misdiagnosed this as a fundamental "signal separability ceiling" for days. Fixed.
   - **`THETA_SIGN` was backwards** on first guess; caught by building a synthetic test, not assumed.
   - **p011/p012**: genuinely inherent — their own baseline also scores 0 credit on them.
   - Honest limitation: our present-pair degradation is still too mild (4% below 0.55 vs their ~50%).

   > **[2 Sep] DO NOT reuse draft §2 ("a large fraction of failures are unsolvable by
   > construction").** Their dataset prompt §5 mandates a label-verification gate — every present
   > pair's global NCC peak must land within 3 px of the label with a margin ≥0.02, cross-checked by
   > a second renderer, or the pair is resampled/dropped. So **every present pair in the graded set
   > is provably hittable.** That framing is true of our generator and false of theirs; using it
   > would read as an excuse. Two better, *measured* additions are in
   > [`ORGANIZER_MATERIALS_DIGEST.md`](ORGANIZER_MATERIALS_DIGEST.md) §1.2 and §1.6 — the off-grid
   > scale residual, and the decoy large-scale-structure signature. Both are diagnosed, both
   > deliberately not implemented; the prompt says an honest limitations list is where the marks are.
2. **Final zip assembly + fresh-clone verification** (§5.2). Contents per slide 5: `register.py`,
   `generate_dataset.py` (documented), model weights, `requirements.txt` (already regenerated, torch
   optional), `failure_analysis.pdf`, citations (`docs/GENERATOR_SPEC.md`).
3. **[2 Sep] `README.md` — not a slide-5 zip item, but do it anyway; it is cheap and it is asked
   for.** It is still the **Phase 1** README: it opens with *"94.5% accuracy @5px … ~150–430 ms per
   pair on a 4 GB laptop GPU"*, never mentions `register.py`, and §4.3 is titled *"Why a router
   instead of just the better model"* — which contradicts the shipped classical-led, CPU-only
   config. It is the first file a judge opens. The mentor also made one explicit README request we
   have not answered: *"your own confidence… if you can mention that in your readme, that'll be
   really good, because that way we can quickly read the readme on how you have found your
   confidence"* — we document the `score` column nowhere, and it sits next to the 10-pt calibration
   bucket.

**NOT required by the Phase 2 task deck** (build only if the submission portal asks): a Phase 2 PPT,
a demo video. Phase 1's `driftsense-demo.mp4` is stale and untracked; leave it out.

**Runtime figure to quote in any deliverable** (measured on two machines, cite the range — quoting
only the faster box would be cherry-picking): CPU-only, no GPU, **1.3–2.8 s/pair median**; worst
single pair across 100 samples **1.47 s vs the 20 s hard timeout (13.6x margin)**. Latency is fully
settled — do not re-benchmark, and do not trim the search grids for speed.

### 5.2 VERIFICATION CHECKLIST (do this before shipping)
- [ ] `git pull` — confirm synced with `origin/main`
- [ ] Fresh clone into a new folder → `pip install -r requirements.txt` → `python register.py
      --input pairs.csv --output predictions.csv` on the organizer sample → 20 valid rows, exit 0
- [ ] Confirm `data/organizer_sample/` and all `data/amp_*`, `data/p2*` stay **gitignored**
- [ ] Confirm no AI attribution anywhere: `git log --format='%B' | grep -iE "claude|anthropic|co-authored"`
- [ ] Confirm `driftmatch/checkpoints/best_phase2_speckle.pt` is committed (~1 MB) — without it the
      net path breaks on clone (though classical still runs)
- [ ] `scripts/eval_organizer.py --data data/organizer_sample` reproduces 35.60/40

### 5.3 OPEN / DEFERRED (do NOT start unless deliverables are done)
- Present-degradation fidelity: our pairs rarely drop below 0.55 (theirs do ~50% of the time).
  Needs a less-idealized layout model — too big for the remaining time.
- Calibration AUC (~0.73) is our weakest scored bucket.
- Set D optical RGB bonus: not implemented, deliberately.

---

## 6. ENVIRONMENT & HOW TO RUN

```bash
# Classical + generator (NO torch) — this is the shipped path
.venv/Scripts/python.exe register.py --input pairs.csv --output predictions.csv

# Neural / training (torch 2.5.1+cu121, CUDA, RTX 3050)
C:/Users/ARYAN/AppData/Local/Programs/Python/Python312/python.exe -m driftmatch.train ...

# Score against the organizers' real sample
<py> scripts/eval_organizer.py --data data/organizer_sample
# Exact rubric scorer on our own data
<py> scripts/score_phase2.py <dataset> --ckpt <ckpt>
```

**CRITICAL:** `.venv` has **no torch** — that is intentional and correct (it proves the torch-free
path). Any run through `.venv` is classical-only. Historically this caused a false "results dropped"
panic; always state which environment a number came from.

**Dataset seeds (all gitignored, regeneratable):** 900000 train8k · 910000 eval100 · 920000 test150 ·
950000 calib300 · 960000 reject_test300 · 970000 speckle4k · 980000 speckle_eval100 ·
770000/880000/990000 AMP train/train2/holdout.

---

## 7. ORGANIZER-PROVIDED MATERIAL (and the compliance line)

Shared via SharePoint (mentor Sourabh Ubale) — **local at `data/organizer_sample/`, GITIGNORED,
must never be committed to the public repo**:
- 20 real pairs (`reference/`, `search/`), `pairs.csv`, **`ground_truth.csv`**, `manifest_jury.csv`,
  `baseline_calibration.txt`, `README.md`
- **Their full generator source** (`src/phase2_pipeline.py`, `presets.py`, `sem_imaging.py`,
  `patterns/`) + the "Prompt for phase 2 dataset" docx

**The compliance line, precisely:**
- ✅ ALLOWED: running their generator source with NEW seeds to synthesise NEW pairs (Phase 1
  precedent; shared tool). ✅ Extracting conventions/parameters from their spec into our generator.
- ❌ FORBIDDEN: training on, or fitting any parameter to, the **20 sample pairs**. They are a
  validation fold — scored only, never learned from. This was honoured throughout.
- Our own generator (`driftsense/`, `generate_dataset.py`) remains the shipped deliverable.

**Mentor Q&A (answered 1 Sep, recorded in `docs/CONTINUE_HERE.md`):**
1. *"Can we extend the Phase 1 architecture?"* → **"Yes, definitely... adjustments are expected."**
   This covers the classical-x,y switch.
2. *"How is rejection F1 counted?"* → "standard binary classification F1" — ambiguous on positive
   class. `scripts/score_phase2.py` already reports all three (present+, absent+, macro).

**Their own naive ZNCC baseline** scores 0.80 mean credit on present pairs — we are above it.

---

## 8. COMPETITIVE LANDSCAPE (recon done 1 Sep, all public repos)

| Team | Repo | Last activity | Note |
|---|---|---|---|
| NanoBolts | aashishniranjanb/Drift-Sense-SEM-Localization | **30 Aug** | Only other team visibly doing Phase 2. Has `phase2/register.py`. **Their own report: 61% Top-100 candidate recall, "oracle ranking ceiling 63.57%"** — well below us |
| TECHTONICS | DK-A/Techtonics_Drift-Sense... | 19 Aug | Claims 98.3% but self-scored on own data; their failure report shows a **hard centre-tiebreak bug** picking a wrong candidate merely for being nearer centre |
| Volt Visionaries | RHUDHRESH/LatticeRank-SEMICON-2026 | 19 Aug | Most rigorous/honest. Publishes 48.75% internal vs 93% external. **P95 runtime 30.32s — over the 20s hard timeout** |
| SILICOFORGE | marakahansika27-prog/SilicoForge | 18 Aug | Own audit: learned component **never trained** (random init); had a ~230px centroid bug |
| Bhoochadae | avaramahmood/semicon_driftsense | 18 Aug | Infinite streaming generation + cross-encoder re-ranker (interesting, not adopted — DQ risk) |
| Others (SUNRISE, NanoTrace, Black_Pearl, Learning Loop) | — | 18–19 Aug | All frozen pre-Phase-2 |

**Read:** 7 of 9 haven't touched their repos since before the Phase 2 addendum existed. Our position
is genuinely strong.

---

## 9. WORKING RULES (non-negotiable)

- **Commit as `Aryan Chourasia <achourasia_be24@thapar.edu>`** (teammate commits as `DevaanshGupta8`
  — that is fine and agreed).
- **NO AI attribution anywhere** — no co-author trailers, no "generated with", in commits, code
  comments, or docs. Scan before every push.
- **Never `git add -A`** — stage explicitly; the data dirs are huge.
- **Never push without explicit OK from Aryan.**
- **Never commit `data/organizer_sample/`** — it is the organizers' private validation data on a
  public repo.
- Report numbers honestly; always say whether a number is from our generator or their real sample,
  and whether it is classical-only or includes the net.
- When a fix "works", verify it generalises (unrelated dataset) before believing it — this project
  has been burned repeatedly by self-graded numbers.

---

## 10. RECOMMENDED PRIORITY FOR THIS SESSION

**[2 Sep] Rewritten — §5.1 was corrected against the official task deck (no PPT, no video required)
and this list had not been updated to match.**

1. **Fresh-clone verification** (§5.2) — ~15 min, and it is the only item that can silently zero the
   whole submission. A broken `register.py` on a clean checkout scores nothing no matter how good
   the PDF is; everything else merely loses points. Run it first, then write knowing the artifact
   is sound.
2. **`failure_analysis.pdf`** — highest unclaimed point value (part of 10 pts). Material is ready,
   but read the `[2 Sep]` note in §5.1 first: one draft section must not be reused.
3. **Final zip assembly.**
4. **`README.md` refresh** — currently advertises Phase 1 numbers and a GPU runtime; also carries
   the mentor's one explicit README request (document the `score`/confidence column). Cheap.
5. Only if all above are done: touch nothing else. The technical work is finished and validated.
   A Phase 2 PPT and demo video are **not** required by the task deck — build them only if the
   submission portal separately asks.

**Do NOT:** retrain, change `FOUND_PEAK`, re-enable the net, trim the search grids, revisit
Fourier-Mellin, or start new model architectures. Every one of these has been tested and settled,
with numbers, and is recorded in `docs/PHASE2_RESEARCH_NOTES.md`.

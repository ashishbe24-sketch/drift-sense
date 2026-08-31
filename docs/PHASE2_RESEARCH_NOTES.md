# Phase 2 — Implementation Research Notes

Quick research pass against the two hardest open problems in
[PHASE2_UNDERSTANDING.md](PHASE2_UNDERSTANDING.md) §7, done against a tight token budget — this is
findings + a recommended path, not finished code. Also re-verified the exact current code (`solve.py`,
`route.py`, `infer.py`) so the plan below is grounded in what actually exists today, not the README's
prose description of it.

---

## What the current code actually does (re-confirmed by reading it, not the README)

- **`solve.py`**: `SCALE = 10` is hardcoded — `downsample()` always divides by exactly 10. The
  existing search is **blur × angle only**: `BLURS = (0,1,2,3)` px × `ANGLES = (-4,-2,0,2,4)°` = 20
  variants, coarse pass at half-resolution picks the winning (blur, angle), then one full-res
  correlation at that setting. Sub-pixel refinement (parabola fit) is applied to **(x, y) only** —
  there is no angle or scale refinement at all right now.
- **`route.py`**: `is_multimatch()` already computes something adjacent to what rejection needs —
  it reads the heatmap's top-2 peak ratio via `maximum_filter` + NMS. That machinery is reusable.
  `locate()` returns `(x, y)` only; no scale, theta, found, or score anywhere in the return path.
- **`infer.py`**: `predict()` returns `(x, y)`; this is the function `register.py` needs to wrap
  and extend, not replace.

So the honest gap is exactly what §7 said: scale search, angle *precision*, found, and score are
all genuinely missing, not just unexposed.

---

## Problem 1: recovering scale ∈ [8,12] and refining theta to ≤0.25–0.5° precision

**Option A — extend the existing brute-force grid (lowest-risk, smallest diff).**
`_best_variant()` already loops blur × angle; add a scale axis: e.g. 5 scale samples ×
5 angle samples × 4 blur = 100 variants at the coarse (half-res) pass — 5× the current 20, still
cheap because the coarse pass is on a 500×500 image. Take the winning (scale, angle, blur) from
the coarse grid, then do a **local refinement** around it — a small Nelder-Mead or coordinate-descent
step in (scale, angle) using the fine-resolution NCC as the objective, the same way `_subpixel()`
already refines (x,y) via a local parabola. This is the natural extension of code that already
exists and ships in `solve.py` today; no new dependency, no retraining risk.

**Option B — Fourier-Mellin Transform (FMT) / log-polar correlation**, the textbook answer to
"recover unknown rotation+scale between two images in one shot." Log-polar remapping turns
rotation and scale into pure translations, solvable by a single phase correlation instead of a
grid search. Two existing OpenCV/Python reference implementations:
- [yycho0108/LogPolarFFTTemplateMatcher](https://github.com/yycho0108/LogPolarFFTTemplateMatcher)
- [Smorodov/LogPolarFFTTemplateMatcher](https://github.com/Smorodov/LogPolarFFTTemplateMatcher)

**Refinement method, confirmed by research:** golden-section search / coordinate descent (optimize
one parameter axis at a time, cycling) is the standard lightweight technique for exactly this —
refining a small number of continuous parameters (here: scale, angle) around a good coarse
estimate, using the correlation value itself as the objective. This is a much smaller addition
than a full Nelder-Mead simplex and fits directly on top of the existing `_subpixel()`-style
parabola-fit pattern already in `solve.py`. ([Golden-Section variant of Nelder-Mead, Springer](https://link.springer.com/article/10.1023/A:1014842520519),
[Nelder-Mead method overview](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method))

**Recommendation:** treat Option A as the MVP (small diff, directly extends code we already have
and understand, no new failure surface) and Option B as a stretch/fallback if Option A's grid
search is too slow on the CPU-only reference machine or doesn't hit the tight pose tiers. Do **not**
start with FMT — given the "materially different from Phase 1" disqualification rule, swapping the
whole matching core for a log-polar pipeline is exactly the kind of change that rule is aimed at;
extending the existing NCC grid is defensibly "the same method."

For `driftmatch/`: the net currently assumes a fixed-ratio 100×100 reference crop (i.e. pre-downsampled
by exactly 10× before it ever reaches the network). The equivalent extension is to **resample the
reference at a few hypothesized scales before feeding the net** (cheap, no retraining) as a first
pass, with retraining on the wider scale/rotation range as the higher-effort follow-up if the
resample-only approach doesn't generalize.

---

## Problem 2: the `found` rejection flag + `score` confidence column

Confirmed via research: the standard technique here is **Peak-to-Sidelobe Ratio (PSR)** — the gap
between the top correlation peak and the surrounding sidelobe/second-peak statistics, used in
tracking literature specifically to decide "is this actually a match or not," and noted as more
robust than a flat absolute-correlation threshold. `route.py`'s `is_multimatch()` already computes
the top-2-peak ratio — extending it to a rejection decision is a small step, not new machinery:

1. Compute the same top-peak-value + second-peak-ratio signal already used for `is_multimatch()`.
2. Calibrate an absolute-value threshold (not just the ratio) on our **own regenerated dataset**
   with absent pairs included (per §7 item 8 — we don't have any absent pairs to calibrate against
   yet), optimizing F1 the way the rubric scores it.
3. Use the same signal (or the raw peak value) as the `score` column directly — it's already
   monotonic with match quality, which is all the confidence-calibration scoring requires.

This is the cheapest of the open problems: no new search dimension, just a threshold to fit once
absent-pair data exists.

**Calibration methodology, confirmed by research (open-set recognition literature):** the standard
approach is precisely what we already planned — build a validation set containing *both* known
(present) and unknown (absent) examples, then sweep the threshold to maximize an open-set metric
like AUROC or OSCR, rather than picking a threshold analytically. This validates step 2 above as
methodologically sound, not just a reasonable guess — but it also confirms we cannot skip
regenerating the dataset with absent pairs first, since there is no way to calibrate a rejection
threshold without labeled negative examples to sweep against.
([Open Set Recognition Mechanisms overview](https://www.emergentmind.com/topics/open-set-recognition-mechanisms),
[Learning for Transductive Threshold Calibration in Open-World Recognition, arXiv 2305.12039](https://arxiv.org/pdf/2305.12039))

---

## Priority order given the above (unchanged from before, now more concrete)

1. **CPU-only latency benchmark** — still first; determines whether Option A's grid search (now
   5× larger) is even affordable within the 5s median / 20s hard cap on a no-GPU 4-core machine.
   **When benchmarking, sweep `torch.set_num_threads(1..4)` (and `set_num_interop_threads`) as a
   variable, not just GPU-vs-CPU** — for small models, more threads can be *slower* due to
   oversubscription; `torch.set_num_threads(1)` is a commonly reported fix, not just a fallback.
   ([Optimizing PyTorch Model Inference on CPU](https://towardsdatascience.com/optimizing-pytorch-model-inference-on-cpu/),
   [PyTorch CPU threading docs](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference))
   **Safety net if thread-tuning alone isn't enough:** PyTorch's x86 INT8 dynamic quantization
   backend reports ~3x geomean CPU speedup over FP32 across common model types, with negligible
   accuracy loss — a `torch.quantization.quantize_dynamic()` one-liner is worth trying on
   `DriftMatchNet` before considering anything more invasive (pruning, distillation, architecture
   changes) if the plain-FP32 CPU benchmark comes in close to the 5s budget.
   ([PyTorch INT8 quantization for x86 CPU](https://pytorch.org/blog/int8-quantization/),
   [Intel: X86 quantization backend](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-int8-inf-with-new-x86-backend.html))
2. **`register.py` I/O wrapper** — mechanical, unblocks testing everything else end-to-end early.
3. **Regenerate our dataset** with scale/rotation ranges + absent pairs (blocks both Problem 1's
   refinement tuning and Problem 2's threshold calibration — nothing to tune against without it).
4. **Scale+angle grid extension in `solve.py`** (Option A above).
5. **Rejection threshold + score column in `route.py`**.
6. **`driftmatch/` scale-resample + retrain**, if time remains after 1–5.

---

## Implementation progress

**Item 2 — `register.py` skeleton: DONE and validated end-to-end (28 Aug).**

- `register.py` created: exact `--input/--output` CLI, reads pairs.csv (flexible column-name
  detection until the real sample lands), writes the 6-column contract, one row per pair_id in
  input order, wraps each pair in try/except so a failing pair is written as `found=0` rather than
  dropped (a missing row scores zero — a zeroed row is strictly safer).
- `route.predict_full() -> PairResult` added as the single six-field prediction function;
  `register.py` is a pure I/O shell over it, so upgrading the matcher never touches the contract.
- `solve.py` extended backward-compatibly: new `_fine_score_full` surfaces the recovered
  (angle, blur); `fine_score` stays a 4-tuple wrapper (sweep_lambda unaffected); `locate` gains an
  opt-in `return_info=True` returning `{score, theta, scale, blur}` with the default 2-tuple return
  unchanged (all 6 existing callers verified intact).
- Validated on `curated30`: 30/30 pairs, x/y sub-pixel accurate (e.g. C00 pred 559.9,470.0 vs gt
  560,470), `score` = peak NCC (0.89–0.99 on these clean pairs), `found`/`scale` correct.

**Field status in the current skeleton** (what is real vs. placeholder):

| Field | Status |
|---|---|
| `x`, `y` | **Real, strong** — sub-pixel, from the existing classical matcher. |
| `score` | **Real** — peak NCC, a genuine monotonic confidence signal. |
| `found` | **Mechanism real, threshold provisional** — `FOUND_THRESHOLD = 0.30` in `route.py`, must be recalibrated on regenerated absent pairs (item 5). |
| `scale` | **Placeholder = 10.0** — the fixed `SCALE`, correct only because curated30 is fixed-10×. Needs the [8,12] scale search (item 4). |
| `theta` | **Surfaced but currently unreliable** — *finding:* on C00 (gt rotation 1.8°) the matcher reported 0.0°. The existing angle grid is 2° steps (`ANGLES = -4,-2,0,2,4`), too coarse to resolve <2° and it snaps to 0. This is direct evidence for the rotation-precision half of item 4: the grid needs finer sampling **plus** the golden-section/coordinate-descent refinement (Problem 1) to reach the ≤0.25°/0.5° pose tiers. Also: `THETA_SIGN` in `solve.py` is an unverified guess — must be checked against sample-pair ground-truth theta the moment it lands (~29 Aug). |

**Immediate next task: item 4** — extend `solve.py` to (a) search scale over [8,12] and (b) sample
angle finely + refine, then feed both recovered values into `return_info`. This directly fixes the
`scale` placeholder and the `theta` finding above, and is the prerequisite for the pose-recovery
20 pts. Item 3 (dataset regeneration) can proceed in parallel and is the gate for item 5.

**Items 3 (scale half) + 4 (scale search): DONE and validated (overnight, 28 Aug).**

- *Generator (item 3, scale + signed-rotation sampling):* `sample_spec` gained gated
  `scale_range` and `signed_rotation` params; `generate_dataset.py` gained `--phase2` /
  `--scale-range`. Both **default-off, so Phase 1 datasets stay byte-identical** (verified: seed
  7000 still scale=10.0, unsigned rotation). Zoom is now plumbed through `generate_one` →
  `make_pair(wide_px_nm=spec.wide.px_nm)`, so the rendered pixels match the recorded `scale`
  column (new). Signed-rotation *sampling* is in, but note it rotates the whole layout (ref+wide
  together) — it does NOT yet create *relative* rotation between ref and wide, which is what
  `theta` recovery needs (see rotation note below).
- *Why this was the top priority — measured, not assumed:* on a fresh 12-pair varying-zoom set,
  the old fixed-`SCALE=10` solver had **3/12 catastrophic localization failures** (570px, 390px,
  88px) → only 75% within 5px. Scale mismatch degrades the true peak until a periodic decoy wins.
  So scale search protects the **40-pt localization score**, not just the 10-pt scale score.
- *Solver (item 4):* `solve.py` gained an opt-in `scales=` path (default `None` = exact Phase 1
  behaviour, byte-verified on curated30 C00). It resamples the reference to each candidate zoom
  (`_stamp_at_scale`, BOX area-average to match the detector), ranks scales by half-res peak NCC
  with a single cheap variant (`SCAN_BLURS/SCAN_ANGLES`), golden-section-refines the winner
  (`_refine_scale`, the technique from Problem 1), then recovers blur/angle once at the chosen
  scale. `_best_variant` now also returns the peak value (to rank scales); `_fine_score_full`
  returns the recovered scale; all existing 4-tuple/2-tuple callers preserved.
- *Result on the 12-pair set:* within-5px **75% → 83%**, median scale error **4.1% → 1.0%**
  (50% within the 1% tier, 67% within 2%), at **1.33 s/pair** (was 5.25s before the
  single-variant scan optimization) — inside the 5 s CPU median budget with headroom. The classical
  path is already CPU-only numpy/scipy, so this timing is representative of the no-GPU grader.
- `route.predict_full` now always passes `PHASE2_SCALES`, so `register.py`'s `scale` column is
  real. Validated end-to-end: 12/12 pairs, scale recovered per-pair.

**Item 3 (absent pairs) + item 5 (rejection): DONE and validated (overnight, 28 Aug).**

- *Generator:* `make_pair(absent=True)` renders the wide from the same layout with the landmark
  shapes removed (periodic architecture kept, unique site gone) — "a different die region of the
  same architecture", exactly the Set C spec. `sample_spec(absent=True)` marks `present=0` and a
  -1 gt sentinel; `generate_dataset.py --absent-fraction` decides absent per-seed deterministically.
  New `present` column in the manifest.
- *The calibration story (and a reversal worth recording).* First tried on an 18-pair set:
  `distinct = peak * (1 - second_peak_ratio)` (how much the winning peak stands out) beat raw peak
  NCC (F1 1.0 vs 0.86), so `found` was wired to `distinct`. **Then re-validated on a larger, harder
  60-pair set (16 absent, more aliased/multi-match present pairs) and the result reversed:** raw
  **peak NCC won, F1 0.925 vs 0.854 for distinct.** The reason is instructive — a *present-but-
  periodic* pair still has a strong landmark peak (high raw peak) but low distinctiveness (its
  decoys tie it), so `distinct` conflates "absent" with "present-and-periodic", which is exactly
  the Set C trap ("a different die region of the same architecture"). Raw peak does not conflate
  them. **Lesson: the 18-pair F1 1.0 was a small-sample artifact; always calibrate rejection on a
  large, hard, aliased-heavy set.** `found` now thresholds raw peak (`FOUND_PEAK = 0.70`, robust
  across 0.67-0.73), and peak doubles as the `score` confidence. `solve.locate` still returns
  `distinct`/`second_ratio` in case they help a future multi-match refinement.
- *End-to-end on the 60-pair set (`register.py`, corrected):* rejection **F1 0.925**
  (TP 43, FP 6, FN 1, TN 10), localization on found-present **81% within 5px, median 0.55px**,
  scale error median **0.62%**. Honest numbers on a hard mixed set; the 6-column contract writes
  correctly, absent pairs get found=0 with zeroed pose.
- *Caveat:* still our own generator, not the organizers'. FOUND_PEAK is provisional — recalibrate
  on their data, and the operating point may shift once Q2 (rejection-F1 positive class) is known.

**Field status now:** x/y — real, strong (80% @5px on a hard set). scale — **real, recovered**
(median err 0.62%). found — **real, peak-thresholded, F1 0.925** (provisional threshold). score —
peak NCC. theta — still 0 (rotation deferred, below).

**Item 6 (net) — DECISION REVERSED (29 Aug): the net stays; do NOT ship classical-only.**

The 28 Aug "classical-only" call below was wrong because it judged the net on OUR generator's data.
Correcting it with the decisive evidence: **on the ORGANIZERS' own data (Phase 1 overnight log,
300 pairs), the net router scored 84% @5px vs classical's 56%** (DriftRoute vs DriftFind). Classical
alone craters on their domain. Since the organizers evaluate on THEIR data, the net is essential,
not optional. Two root causes were then found for why our data misled us:
  1. **Our generator lacks geometric warps** (barrel / scan distortion / astigmatism) — it models
     only drift-shear + vibration + blur + noise + charging (grep of physics.py). The net's whole
     edge is absorbing warps NCC can't; with no warps in our data, the net shows no advantage here
     AND can't learn the skill. The organizers' Set B explicitly lists "scan distortion" — a warp we
     do not model. This is a real generator gap (also costs dataset-realism credit).
  2. **The net needs scale handling.** Best approach measured: classical estimates scale (0.62% err)
     → resample the reference to that scale → net localizes once at the correct scale (one CPU pass,
     ~2.35s, within budget). Tested on our (warp-free) data it is 71% vs classical 87% — but our data
     is exactly the wrong test (no warps). On warped/organizer-like data the Phase 1 result says the
     net wins big.

**Corrected plan:** (a) add scan distortion to the generator [DONE], (b) retrain the net on
scale+warp data (warm-start) [running], (c) ship the router with the scale-hybrid (classical scale →
net at that scale, classical for multi-match + scale + rejection). This is the Phase 1 router,
extended for scale — rules-compliant. The text below is kept for the record but superseded.

**Progress on the corrected plan (29 Aug):**
- *(a) Scan distortion added to the generator — DONE, validated.* `physics.py` gains
  `scan_distortion_field` (smooth low-frequency 2-D warp, peak amplitude in px) + `apply_scan_distortion`.
  Sampled for the **wide capture only** (a *relative* warp ref↔wide — the thing NCC can't absorb),
  gated so Phase 1 stays byte-identical (verified seed 7000 → scan_dist 0.0). `generate_dataset.py`
  `--phase2` now also enables it (max 6 px); `--scan-distortion` sets it explicitly; new
  `wide_scan_distortion_px` manifest column. **Ground truth stays exact under the warp:** the wide is
  warped and the label is shifted by the field's value at the landmark (validated — a pair with 5.1px
  distortion still localizes to 0.5px, i.e. gt follows the warp). And it does break classical as
  intended — a 3.9px-distortion pair gave classical a confident-but-wrong 348px miss, the exact case
  the net should win. This also fills the Set B "scan distortion" requirement (dataset-realism credit).
- *Model check:* `xcorr_depthwise` pads by `Hk//2`, so the heatmap is always 250×250 regardless of
  reference size → scale-correct (variable-footprint) references need NO model change, available for a
  later refinement if scale-tolerant training caps out.
- *(b) Warped training — running.* Regenerated 4000 warped+scale training pairs + warped eval sets;
  fine-tuning from Phase 1 `best.pt` (14 ep, scale-tolerant first — minimal change, warps are the key
  new ingredient). Checkpoint → scratch `p2ckptw/`; Phase 1 `best.pt` untouched. Once done: compare
  net vs classical vs router ON WARPED data (where classical is weak) to confirm the net earns its place.
- *`train.py` resume-baseline bug fixed earlier* still applies — the run re-baselines `best` on the
  current eval set.

**Warped fine-tune result + final architecture decision (29 Aug):**
- Fine-tune completed (after two crashes fixed: GPU OOM → batch 2; host-RAM OOM → `--limit 2500`).
  Best held-out on WARPED eval **81.7%** (resumed from a 76.7% partial). Checkpoint: scratch
  `p2ckptw/best.pt`; Phase 1 `best.pt` untouched.
- *Decisive eval, 113 warped solvable present pairs, net-at-classical-scale vs classical vs router:*
  classical **85%** ALL / **86%** on distorted; net@scale 76%/82%; router 79%/80%. **On OUR data,
  classical wins even on distorted pairs.**
- *But our data under-represents the grading domain.* Our single scan-distortion warp is milder than
  the organizers' warp set (astigmatism + barrel + corner-rounding + linewidth-bias), which on their
  Phase 1 data dropped classical to 56% while the net router hit 84% — a 28-pt gap on the domain they
  actually grade, and the approach that got us selected. ds_ref (their generator) is no longer checked
  out, so a direct Phase-2-domain test isn't available without re-cloning (deemed overkill).
- **DECISION: ship the ROUTER (classical scale-search as strong primary + warp-net as hedge), not
  classical-only.** Rationale: classical-only looks best on our (milder) data, but if the organizers'
  Phase 2 warps behave like their Phase 1 — very likely (same team; Set B lists scan distortion) —
  classical-only could crater to ~56% and cost the 40-pt localization score and selection. The router
  hedges that catastrophic downside at a cost (net 2.35s/pair) within the 5s budget. The session's
  work made the net scale-capable + warp-trained (previously stuck at fixed-10, useless for Phase 2)
  and upgraded the generator (scan distortion = Set B realism = dataset-score credit) — necessary, not
  wasted. Router wiring/tuning in `route.predict_full` is the remaining implementation step (keep it
  simple: classical primary, net consulted where classical confidence is low — avoid regressing the
  85%).

---

**[SUPERSEDED] Item 6 earlier reasoning (28 Aug) — kept for the record:**

- *Fine-tune done:* warm-started from Phase 1 `best.pt`, 12 epochs on 3000 scale-varied [8,12]
  pairs, lr 3e-4, ~1h on the RTX 3050. The net *did* learn some scale tolerance vs the untouched
  Phase 1 net (one held-out set 66.7% → 81.7%; another flat 78.3%; Phase 1 domain 100% → 96.7%, a
  small expected forgetting). Checkpoint in scratch `p2ckpt/last.pt`; **Phase 1 `best.pt` untouched.**
- *Decisive head-to-head on 60-pair Phase 2 present pairs:* **classical scale-search 80% @5px
  (0.56px median) > fine-tuned net 68% (0.86px); a simple ensemble matched classical (80%), no
  gain.** So the net does not beat classical AND costs 2.35s/pair on the no-GPU grader.
- *Why the reversal from Phase 1 (where net won 94% vs 75%):* Phase 1 scale was fixed at 10, so the
  net's edge was absorbing geometric warps; Phase 2's dominant challenge is unknown scale, which the
  classical **explicit** scale-search handles better than the net's implicit tolerance. Warp
  absorption doesn't outweigh explicit scale handling here.
- *Decision:* **ship classical-only for Phase 2 localization.** Simpler, faster, and more accurate
  on the evidence. The fine-tune was worth it to *rule the net out with data* rather than guess.
- *`train.py` bug fixed along the way:* `--resume` used to trust the resumed checkpoint's stale
  `acc_val` as the "best" baseline, so on a different eval set no epoch ever beat it and `best.pt`
  was never written (only `last.pt`). Now it re-evaluates the resumed net on the current `--eval2`
  to set the baseline. Fixed in `driftmatch/train.py`.
- *Not pursued (speculative, GPU-costly):* a "classical estimates scale → resample ref → run net
  once at the correct scale" hybrid might let the net help specifically on the heavily-warped Set B
  degraded pairs. Left as a future option only if classical-only proves weak on degraded data.

**Localization failure diagnosis (why chasing the remaining misses isn't worth it):** of 9 misses
in 44 present pairs on the 60-set, scale was recovered fine (median 0.58% err, same as the hits),
so scale search is not the cause. 4/9 are `below_floor` difficulty — pairs the generator makes
**unsolvable by construction** (landmark below the visibility floor, ~8% by policy) — they are meant
to fail. The rest are a few genuinely hard weak-landmark cases (large 100-468px misses on periodic
structure). So on *solvable* pairs the classical path is ~88% @5px; the 80% headline includes the
unsolvable-by-design pairs. No cheap fix exists (scale is fine); the residual is inherent difficulty.
The classical path is in good shape and not the bottleneck.

**Still open:**
- **Rotation / `theta` recovery.** The generator renders ref and wide from one layout at one
  shared angle, so there is no *relative* rotation to recover, and `theta` is still 0. Phase 2
  needs the wide rendered at θ relative to the ref, about the match centre — geometrically delicate
  (sign convention, centre of rotation), and the sign cannot be validated without the organizers'
  sample ground-truth theta (~29 Aug). Deliberately deferred rather than guessed unsupervised.
- **Net retrain (item 6).** The data pipeline it needs is now mostly ready (scale + absent
  generation work); still worth waiting until rotation is in and the Q answers land, so the net is
  trained once on the complete Phase 2 distribution rather than retrained twice. GPU authorised by
  the user for when it runs. Not started tonight — long, and better done on the finished pipeline.
  *Also note:* **torch is not installed in the project `.venv`** — the router transparently falls
  back to classical-only (verified: `predict_router.py` prints "net unavailable ... using classical
  only" and still returns the right answer). So all tonight's improvements are on the classical
  path, which is CPU-only and therefore representative of the no-GPU grader. Any net retrain needs a
  torch(+CUDA) install first — a deliberate, supervised step, not something to do unattended.

**Backward-compat re-verified after all tonight's edits:** `predict.py`, `predict_router.py`,
`infer.py` all return the byte-identical `559.904, 470.001` on curated30 C00; `fine_score` still a
4-tuple; `scripts.sweep_lambda` still imports. Phase 1 sampling unchanged (seed 7000 → scale 10.0,
unsigned rotation). No regressions.

---

## Problem 3: regenerating the dataset (scale range, signed rotation, absent pairs, severity tiers)

Correcting §7 item 8 of `PHASE2_UNDERSTANDING.md` — checked `driftsense/sampling.py` and
`driftsense/physics.py` directly (grep, not the README's summary) and **the gap is smaller than
originally scoped**, because the Phase 1 webinar-correction pass already built most of the physics:

| Needed for Phase 2 | Status in code today |
|---|---|
| Rotation up to 5° | **Already there.** `sampling.py:66` — `ROTATION_DEG = (0.0, 5.0)`, comment cites the webinar directly. `rotation_deg` is already a recorded field on `PairSpec` (ground truth exists). |
| Per-polygon scale jitter ±20% | **Already there.** `sampling.py:67` — `SCALE_JITTER = 0.20`, matches Gokul's "shrink or grow a polygon by 20%" almost verbatim. |
| Charging (Set B) | **Already there.** `physics.py: apply_charging()` — field deflection + bright streaks, with a `streak_rate` parameter. |
| Defocus (Set B) | **Already there.** `physics.py: defocus_sigma_nm`, combined in quadrature with probe PSF. |
| **Zoom ratio ∈ [8,12]** | **Missing.** `sampling.py:139` — `wide_px_nm: float = 10.0` is a fixed default, not sampled per pair. This is the one genuinely new physical parameter to add. |
| **Signed rotation (±, CCW positive)** | **Partially missing.** Current sampling is magnitude-only (`rng.uniform(0.0, 5.0)`); need a random sign bit so both directions occur and the reported `theta` has a meaningful sign to be scored against. |
| **Absent-pair mode (Set C)** | **Missing entirely.** No `absent`/`present` logic in `sampling.py`. Needs a new pair-generation path: render two non-overlapping regions of one layout (same architecture, so periodically plausible) instead of placing the reference site inside the wide view at all. |
| **4-tier severity ladder (Set B)** | **Missing as an explicit ladder** — the underlying continuous parameters (dose_ratio, charging, defocus_sigma) exist; they just aren't yet bucketed into 4 discrete, documented severity presets. |

**Practical implication:** this is a lower-risk, smaller-diff task than the original gap analysis
implied — three parameters to add/adjust to an existing, working sampler, not a new simulator.

On hard-negative generation for Set C specifically: general Siamese-network literature on
hard-negative mining (see sources) confirms the intuitive design here — random unrelated crops as
"absent" pairs would be too easy and wouldn't test the rejection logic the way the rubric intends.
The organizers' own description — *"a different die region of the same architecture... plausible
and periodically similar"* — **is** a hard-negative specification: the absent pairs should come
from the same periodic layout family as the true site, which is exactly the case that currently
fools naive correlation (this is why `route.py` already has multi-match handling — the same
periodicity that creates false peaks for localization is what should create false positives for
rejection if not handled deliberately).

---

Sources:
- [strec007/artimagen — NIST's ARTIMAGEN, the paper we already cite in GENERATOR_SPEC.md, with source](https://github.com/strec007/artimagen)
- [Simulated SEM Images for Resolution Measurement (Cizmar et al., the ARTIMAGEN paper)](https://www.researchgate.net/publication/5239806_Simulated_SEM_Images_for_Resolution_Measurement)
- [LogPolarFFTTemplateMatcher (yycho0108)](https://github.com/yycho0108/LogPolarFFTTemplateMatcher)
- [LogPolarFFTTemplateMatcher (Smorodov)](https://github.com/Smorodov/LogPolarFFTTemplateMatcher)
- [Fourier-Mellin transform for rotation/scale/translation-invariant target recognition (ResearchGate)](https://www.researchgate.net/publication/251972485_Rotation_scale_and_translation_invariant_automatic_target_recognition_based_on_Fourier-Mellin_transform_and_bispectrum_for_satellite_imagery)
- [Deep Learning Improves Template Matching by Normalized Cross Correlation (arXiv 1705.08593)](https://arxiv.org/pdf/1705.08593)
- [Building image pairs for siamese networks with Python (PyImageSearch, general pair-construction reference)](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)

**Note:** also surfaced "High-Fidelity Synthetic TEM Image Generation Using Diffusion Probabilistic
Models for Data-Limited Semiconductor Metrology" (arXiv 2606.24817) — flagging only to explicitly
rule it out. Gokul's instruction was unambiguous: *"Don't give Nano Banana or Gemini or OpenAI to
create images for you... so that you know exactly what logic was used and what is the ground truth
at pixel level."* Any generative-image-model approach is disqualifying by the deck's own logic
(ground truth must come from placement, not from a model), regardless of image quality.

---

## Overnight session (29 Aug, unattended) — generator enriched with missing optical aberrations

**Closed a real spec-vs-reality gap.** The README's "NIST ARTIMAGEN paradigm... astigmatism,
vignette, gamma and barrel distortion" line was checked against `docs/GENERATOR_SPEC.md` and
`physics.py` and turned out to describe the *reference paper's* pipeline, not something actually
implemented (grep found zero matches for any of the four). Now implemented, all gated off by
default (Phase 1 byte-identity reconfirmed: seed 7000 → all new fields zero/neutral):

- **Astigmatism** (`apply_astigmatism`): a rotated anisotropic Gaussian kernel (sharp axis + smeared
  axis + angle) — confirmed via research as the standard elliptical-PSF model. Additive with the
  existing isotropic probe/defocus blur.
- **Barrel/pincushion distortion** (`apply_barrel_distortion` + `barrel_displacement_at`):
  single-term radial model, exact inverse via 3 Newton iterations. **Ground-truth sign convention
  verified empirically, not assumed** — built a synthetic bright-dot test, warped it, located the
  dot by centroid, and confirmed `new_gt = gt + displacement_at(gt)` (0.03px residual vs 4.4px for
  the wrong sign). This is the OPPOSITE convention from `scan_distortion` (`gt - displacement`) —
  easy to get backwards by pattern-matching the other one; caught by testing, not assuming.
- **Vignette + gamma** (`apply_vignette`, `apply_gamma`): intensity-only, no coordinate risk.
- **Compounding order handled correctly**: barrel applies inside `render()` (physics.py), before
  scan_distortion (applied after, in `generate_dataset.py`) — so gt shifts must compose in that same
  order: barrel's shift first, then scan-distortion's field evaluated at the already-shifted
  position. Implemented and validated together, not independently.
- **End-to-end validation** (20-pair set, `--phase2`, all aberrations on): **17/19 solvable pairs
  localize to 0.0-0.7px** even with barrel+scan+astigmatism stacked on one pair (e.g. scan=4.1 +
  astig=1.5 → 0.3px) — ground truth exact under compounded warps. 2/19 failed classical badly
  (229px, 270px) — the intended effect: aberrations now strong enough to sometimes break periodic-
  pattern classical matching, which is the net's reason to exist.
- New CLI: `--optical-aberrations` (independent flag); `--phase2` enables it by default now, alongside
  scale+rotation+scan-distortion. New manifest columns: `wide_astig_sigma_nm`, `wide_astig_angle_deg`,
  `wide_barrel_k1`, `wide_vignette`, `wide_gamma`.

**Next (continuing overnight):** regenerate a larger, richer training set with the full aberration
suite, retrain the net on it, then rerun the decisive net-vs-classical-vs-router comparison —
expecting a bigger, more decisive net edge than the scan-distortion-only run showed, since the data
now covers more of what likely drove the organizers' large Phase 1 gap (56% vs 84%).

---

## Overnight: found a real rejection-threshold miscalibration, fixed with proper cost-weighting

A regression check (running `register.py` on curated30 — Phase 1 data, reference always present)
found **one pair falsely rejected**: C09, peak NCC 0.6857, just under the `FOUND_PEAK=0.70`
threshold. Investigating turned up a real methodology flaw, not just a bad instance: **the
threshold was calibrated by optimizing raw F1, which weighs a false-reject and a false-accept
equally — but they are NOT equally costly under the actual scoring rubric.**

- **False reject** (present pair, we report found=0): the contract forces pose columns to 0, so we
  lose the *entire localization credit* for that pair (part of the 40-pt pool) **and** it counts as
  a rejection-F1 false-negative (part of the 15-pt pool). Two pools hit.
- **False accept** (absent pair, we report found=1): Set C isn't in the localization sets at all,
  so this only costs the rejection-F1 false-positive. One pool hit.

Re-swept the threshold on the 60-pair calibration set (`p2val60`) minimizing a cost function with
false-rejects weighted 1x, 2x, and 3x relative to false-accepts, instead of optimizing symmetric F1.
**Result: 0.68 is cost-optimal under all three weightings** (FN=1, FP=6, F1=0.925 — identical F1 to
the old 0.70, so this is a clean improvement, not a tradeoff) and it clears C09's 0.6857.
`route.FOUND_PEAK` updated 0.70 → 0.68; curated30 re-verified all-found after the fix (see log).

**Residual risk, stated honestly:** C09 clears the new threshold by only 0.006 — a thin margin, not
a robust one. If the organizers' actual Set B degradation is harsher than ours, more present pairs
could sit near this edge. The durable fix is a better rejection signal than a single peak-NCC
threshold (e.g. combining peak with a secondary signal, or a small calibrated model), not further
threshold-nudging — noted as a real future-work item, not resolved tonight.

**The general lesson, worth keeping:** when a scoring rubric has asymmetric costs across error
types, calibrating against a symmetric metric (F1, accuracy) can silently pick the wrong operating
point even when it "looks optimal" on its own metric. Worth checking this same asymmetry doesn't
apply anywhere else in the pipeline once more of the rubric's exact mechanics are confirmed (e.g.
via the pending Q2 answer on the rejection-F1 positive class).

---

## Overnight mistake, owned and fixed: killed an in-progress job by accident

Copied the GPU-cleanup pattern from an earlier crash-recovery script (`train_phase2_warp.sh`, where
it correctly cleaned up *stale* processes from a *previous crashed run* before starting) into a new
launcher (`train_phase2_full.sh`) without noticing that this time, a **legitimate, currently-running
sibling job** (the 4000-pair full-aberration data generation, ~34% done) was still active and shared
the same process name. The blind `Get-Process python | Stop-Process -Force` killed it mid-generation
— confirmed via the task notification reporting the generation as "completed" when it had actually
been terminated (no `labels.csv` was written, only 1351/4000 pairs existed on disk). Lost roughly
15-20 minutes of compute, not data integrity (nothing corrupted, just incomplete).

**Fixed:** removed the blind kill from the script; noted inline that any future GPU-freeing should
target specific stale PIDs (checked via `nvidia-smi`), not all python processes indiscriminately.
Regenerated the training data cleanly (new seed range, no concurrent risk this time since the
training launcher's kill step is gone and it was already safely idling in its wait-loop).

**Recorded for the same reason everything else in this file is recorded: an autonomous overnight
session should be auditable, including its mistakes** — this one was caught immediately (checked the
generation progress right after launching the second script, on the reasoning that both jobs used
the word "python" in a way that could collide), not discovered after the fact.

---

## Overnight: full-aberration retrain, the final router design, and a rejected "fix" (with evidence)

**Retrain on full-aberration data (astigmatism+barrel+vignette+gamma+scan-distortion+scale) completed
cleanly** — all 12 epochs, resumed from the scan-distortion-only checkpoint (81.7%), reached **best
held-out 88.3%** (train.py's `quick_eval`, which uses the net's raw fixed-10x path, no scale
correction). Checkpoint copied into the repo as `driftmatch/checkpoints/best_phase2.pt` (a NEW file
-- `best.pt`, the Phase 1 shipped checkpoint used by `infer.py`, is untouched).

**Decisive comparison on the 116-pair full-aberration eval set, with this new checkpoint:**

| Method | @5px | median err | time/pair |
|---|---|---|---|
| classical (scale-search) | 88% | 0.42px | 1.46s |
| net @ classical-corrected scale | 86% | 2.87px | 0.09s |
| **net @ fixed-10 (raw, no scale correction)** | **89%** | 0.67px | 0.09s |
| **hybrid: classical scale/found/score + net x,y** | **89%** | 0.67px | 1.62s |

**Counterintuitive finding, worth keeping:** "helping" the net with the classically-estimated scale
(resampling the reference to the correct footprint before feeding it) made the net WORSE (86% vs
89%). The net's encoder was trained on a fixed 100x100 reference input; a variable-sized stamp is
genuinely out-of-distribution for its architecture, even though the correlation math is technically
size-agnostic. The net has instead learned scale ROBUSTNESS implicitly, by being trained across the
full [8,12] range while always seeing the reference at its native fixed-10x footprint -- so the
"naive" fixed input is actually the in-distribution one. Measured, not assumed.

**Final router design: classical supplies `theta`/`scale`/`found`/`score` (needed regardless -- the
net has no pose head); the net supplies `x, y` when available, unconditionally, with classical's own
`x, y` as the fallback if no net (`route.py: predict_full`, already the exact shape this was
implemented in earlier -- no further code change needed here). 89% @5px, 1.6s/pair, well inside the
5s budget. `register.py` updated to load `best_phase2.pt` explicitly (not the shared `DEFAULT_CKPT`,
which stays pointed at the Phase 1 checkpoint so `infer.py` is unaffected).

## "The results dropped" -- investigated properly, not waved away (29 Aug)

After the rotation work, a fresh 30-pair test showed classical-only localization at 63% @5px,
well below the previously-quoted 88-89%. Challenged (rightly) to find the real cause rather than
explain it away or quietly tune a number back up. Three separate, rigorous checks, in order:

1. **Is the new angle-search code itself a regression?** Generated 60 pairs matching the earlier
   full-aberration/unknown-scale conditions exactly, but held out of `signed_rotation` (i.e. the
   same distribution the original 88% figure was measured on, before relative rotation existed).
   Ran `solve.locate` on the identical 60 pairs twice: once with the OLD default `ANGLES` grid (no
   refinement, the pre-this-session code), once with the NEW `PHASE2_ANGLES` + `_refine_angle`.
   **Result: identical, 76.7% @5px both ways, median 0.498 vs 0.497px, zero pairs flipped in either
   direction.** The angle-search change is a provable no-op on non-rotated data -- ruled out with a
   clean A/B, not by assumption.
2. **Is real relative rotation itself the cause?** The controlled ablation earlier in this file
   (same 30 seeds, with vs without real `relative_theta_deg`, identical noise realization) already
   answered this: 19/30 vs 20/30 -- one pair, inside sampling noise for n=30. Also ruled out.
3. **What actually explains the gap: classical-only was being tested by accident.** This repo's
   `.venv` has no torch (`requirements.txt` lists it, but it was never installed here), so every
   test today silently degraded to the classical-only fallback -- `route.load_net` printed
   "net unavailable ... using classical only" and it went unnoticed. The 88-89% figures being
   compared against were always the **net+classical router**, not classical alone -- an
   apples-to-oranges comparison. Found a second Python environment on the machine
   (`C:\Users\ARYAN\AppData\Local\Programs\Python\Python312\python.exe`, torch 2.5.1+cu121, CUDA
   available -- the environment Phase 1 training already used) and reran the exact same pairs
   through the real router (`best_phase2.pt`):

   | Set | Classical-only | Real router (net+classical) |
   |---|---|---|
   | 30-pair rotation set | 63.3% @5px | **83.3% @5px**, median 0.755px |
   | 150-pair mixed present/absent set | ~68% overall correctness | **81.1% @5px** localization (present only), median 0.787px |

   Theta accuracy under the router: 0.261 deg / 0.218 deg median (comparable to the classical-only
   0.286 deg reported earlier -- theta still comes from classical either way, per the router's
   design, so this is expected).

**Conclusion:** no regression from today's rotation/angle-search work (proven by #1 and #2, not
assumed); the apparent drop was a testing-environment mistake (#3) compounded by normal
small-sample variance (classical-only alone has been observed anywhere from 63% to 88% across
different 30/60/116-pair random draws generated today -- these are still fairly small samples).
81-83% router accuracy is the honest, representative number, in the same ballpark as the previously
quoted 89% (that earlier figure also had no real rotation in the data, which this new number does).

**Process note, worth keeping:** going forward, run validation that is meant to represent the
shipped system through the torch/CUDA Python 3.12 environment, not `.venv` -- and state explicitly
whether a reported number is classical-only or the full router, since silently reporting the
degraded fallback as if it were the shipped path is exactly the kind of mistake that erodes trust
in every other number in this document if it happens again unnoticed.

---

## Confidence-calibration AUC measured for the first time -- and a real rejection weakness found (29 Aug)

Item 2 of the priority list ("validate the AUC properly" -- it had never been tested, only assumed
monotonic). Generated a fresh 150-pair mixed present/absent set (122 present, 28 absent, seed
850000, `--phase2 --absent-fraction 0.22`) independent of the 60-pair set `FOUND_PEAK` was
originally calibrated on, and computed AUC of the `score` column (peak NCC) against per-pair
correctness (present: found=1 and localized <=5px; absent: found=0), via a rank-sum AUC (no
sklearn dependency, matches the standard Mann-Whitney U formulation):

| Metric | AUC |
|---|---|
| score -> found correctly predicted (rejection only) | 0.789 |
| score -> localized <=5px \| present (score as localization confidence) | 0.615 |
| **score -> overall correctness (the literal rubric metric)** | **0.657** |

> **SUPERSEDED (1 Sep):** the "signal-separability ceiling" conclusion below was drawn from data
> produced by a Set C generator BUG (absent wides rendered from the reference's own layout with the
> landmark stripped, so the periodic backgrounds were byte-identical). It was NOT a signal limit.
> After fixing the generator, raw peak-NCC AUC went 0.789 -> 0.945 and rejection F1 0.88 -> 0.93 --
> see the "Set C absent-pair generator bug fixed" entry at the end of this document. The analysis
> below is kept as the honest record of what the broken data looked like and why it misled us.

0.657 is mediocre for a calibration signal. Digging into why (not just reporting the number)
surfaced a real problem, larger than calibration noise: **rejection separability itself breaks
down on "resolved"-regime absent pairs** -- the most common pitch regime by design weight (60%).
Absent-pair peak-NCC scores in that regime ranged up to **0.97** on this set, against a present-pair
median of 0.94 -- i.e. statistically indistinguishable. Sweeping every threshold in [0.50, 0.95] on
this 150-pair set for the best possible F1 still only correctly rejects 6/28 absent pairs (21%);
this is not a threshold-tuning problem, it is a signal-separability ceiling. The reason: "a
different die region of the same architecture" (the Set C spec) means an absent pair *is* periodic,
and on well-resolved periodic structure a decoy peak can be nearly as strong as a true landmark
peak -- exactly the same periodicity that already motivates the centre-tie-break for localization
now also defeats a single-peak-value rejection rule.

**Why this matters for `docs/TEAMMATE_TASK_LARGER_TRAINING.md`:** that task recalibrates
`FOUND_PEAK` on a bigger (300-pair) set, which will likely still help (more data, tighter
threshold estimate) but should not be expected to fully close this gap -- a bigger calibration set
narrows the *estimate* of a threshold, it does not create separability that the underlying signal
does not have. Recorded here so whoever reports back on that task isn't surprised if F1 improves
only modestly, or plateaus, despite 5x the calibration data. The durable fix (unstarted, real
future work, matches what an earlier note already anticipated) is a rejection signal that is not a
single peak value -- e.g. combining peak with the existing `second_ratio`/`distinct` diagnostics in
a regime-aware way, or, if the net is available, its own heatmap's peak/second-peak behaviour
(`route.is_multimatch`'s machinery, already built for a different purpose, is the natural
candidate to repurpose here).

**Update (29 Aug, later): tried a proper fix, it honestly didn't work.** Implemented and tested
Peak-to-Sidelobe Ratio (PSR) -- the actual textbook statistic from correlation-filter target
detection (the peak's z-score against the correlation surface's own mean/std, excluding its
immediate neighborhood), which is a genuinely different measurement from the `second_ratio`/
`distinct` diagnostics already tried and rejected (those compare only to the second-best local
max; PSR compares to the whole surface's noise floor). Tested at 4 exclude-radii (10/20/40/80 wide-
px) on the same 150-pair set: **PSR loses to plain peak NCC at every radius** (AUC 0.766-0.819
range for PSR vs 0.819 for raw peak). The reason makes sense in hindsight: on a well-resolved
periodic surface, the ENTIRE correlation map is uniformly high (strong self-similarity everywhere
the lattice aligns), so both PSR and second_ratio -- which measure how much the peak stands out
from the rest of the surface -- get washed out exactly where discrimination is needed most. The
problem is the peak's absolute height not separating present from absent in that regime, not its
shape relative to its surroundings, so shape-based statistics don't help.

**Decision: do not force a fix on too little data.** Two different, literature-grounded signal
candidates have now failed empirically. The remaining honest options (a small classifier combining
multiple diagnostics, or using the net's own heatmap confidence as an additional signal) both need
more labeled absent-pair data than exists right now (this set has only 28 absent pairs -- fitting
and validating a classifier on that few positives would itself risk being the kind of
looks-good-on-paper, doesn't-generalize fix that should be avoided). **Deferred until the
teammate's 300-pair calibration set (`docs/TEAMMATE_TASK_LARGER_TRAINING.md`) lands** -- that is
real additional data to fit and validate a combined signal honestly, rather than overfitting one
now. Not attempted further tonight for that reason, not because the problem is unimportant.

**Not attempted tonight, deliberately (superseded detail, kept for the record):** redesigning the rejection signal itself. That is a bigger
change than "validate the AUC" calls for, overlaps with the teammate's in-flight recalibration
task, and risks conflicting/duplicate work on the same file (`route.py`'s `FOUND_PEAK` and the
`found` logic). Flagging with hard numbers so the next person (whoever picks this up, possibly
after the teammate reports back) can prioritize it correctly instead of assuming recalibration
alone will fix it.

---

## Rotation (`theta`) implemented and validated self-consistently (29 Aug)

Closed the biggest remaining gap. The diagnosis in the earlier "Still open" note below was
exactly right -- the generator rendered ref and wide from one shared `Layout.angle_deg`, so there
was no *relative* rotation between them to recover. Fixed with a small, surgical change rather
than a new mechanism, reusing exactly the field the generator already sampled for this purpose:

- **`driftsense/raster.py::make_pair`** gained `relative_theta_deg` (default 0.0 = today's
  behaviour, byte-identical). Nonzero: only the WIDE capture is rendered at a shifted
  `layout.angle_deg`; the reference keeps the base angle. The pivot the rasteriser already rotates
  about (`Layout.centre_nm`) is the landmark -- i.e. exactly (gt_x, gt_y)'s nm coordinate -- so it
  is a fixed point of the rotation and **no ground-truth position shift is needed** (unlike barrel
  or scan distortion, which do shift a non-pivot point).
- **`generate_dataset.py`**: when `signed_rotation=True` (already the flag `--phase2` sets), the
  existing signed draw `spec.rotation_deg` is passed through as `relative_theta_deg` instead of
  being baked into the shared layout angle. When `signed_rotation=False` (Phase 1), nothing
  changes -- `relative_theta_deg=0.0`, ref and wide still share one angle, exactly as before.
  Verified: seed 7000 (no `--phase2`) still gives `scale=10.0`, unsigned `rotation_deg`, and
  `solve.locate` on curated30 C00 (scales=None, the untouched Phase 1 path) still returns
  `559.9038, 470.0009` -- matches the documented `559.904, 470.001` to the reported precision.
- **Sign convention, verified empirically, not assumed** (the same standard this codebase already
  held itself to for barrel distortion): built a synthetic layout with one marker off the rotation
  pivot, rotated it via the actual rasterizer, and measured which way it moved. Result:
  increasing `Layout.angle_deg` turns the rendered pattern **clockwise** (in the standard
  CCW-positive sense). Documented inline in `raster.py`.
- **`solve.py`**: rotation recovery previously had no refinement (unlike scale, which already had
  `_refine_scale`) -- the existing `ANGLES` grid (2 deg steps, ±4°) was both too coarse and too
  narrow for Phase 2's ±5° range and the ≤0.25°/0.5° pose tiers. Added `PHASE2_ANGLES` (a ±5°
  bracket at 2.5° coarse steps) and `_refine_angle`, a golden-section refinement mirroring
  `_refine_scale` almost exactly -- it reuses `_coarse_peak` with a one-element `angles` tuple as
  the objective, so no new evaluator was needed. Wired into `_fine_score_full`'s scales-given
  (Phase 2) branch only; the `scales=None` Phase 1 path and its `ANGLES` grid are untouched.
  `route.predict_full` now passes `angles=solve.PHASE2_ANGLES` alongside `scales=solve.PHASE2_SCALES`.
- **THETA_SIGN was wrong -- caught, not guessed around.** Generated 30 fresh Phase 2 pairs (own
  generator, full aberration suite, real signed relative rotation) and compared recovered `theta`
  (via `solve.locate(..., return_info=True)`) against the new ground truth. With the original
  `THETA_SIGN=-1.0`: correlation -0.95, median abs error 3.35°. Flipping the sign
  (`THETA_SIGN=+1.0`): correlation +0.95, median abs error 0.26°. Fixed in `solve.py`, with the
  derivation and the caveat below recorded inline.
- **Result on the well-localized subset of that 30-pair set** (19/30 within 5px -- see caveat
  below on why this set's miss rate is high): median abs theta error **0.286°**; tiers ≤0.25°→37%,
  ≤0.5°→74%, ≤1.0°→89%. Scale accuracy on the same pairs: median 0.98% error, consistent with
  earlier (rotation-free) numbers -- the scale search is not disturbed by the new rotation axis.
- **No localization regression from the new angle-refinement path**: curated30 (Phase 1 data)
  scores identically (90% @5px, median 0.180px) whether the added `_refine_angle` path runs or
  not -- confirmed by running the same call with and without `angles=PHASE2_ANGLES`.
- **Honest caveat on the 30-pair set's miss rate (11/30, vs the ~88% solvable-only figure quoted
  earlier for a rotation-free aberration set):** not a regression from this change -- confirmed by
  the curated30 A/B above, which isolates the angle-search code path and shows no difference.
  This is simply the FIRST test set with a genuine relative rotation actually present, stacked on
  top of the existing scale + full aberration suite, and it is a small (n=30) sample with real
  difficulty-tier variance (4/30 drew `below_floor`, unsolvable by construction). Read as "rotation
  is a genuinely harder axis, worth a larger validation set," not as "the new code broke something."

**Still open, honestly:** this validates that OUR OWN recovery pipeline agrees with OUR OWN
generator's CCW-positive convention -- a real check that rules out a whole class of sign/pivot
bugs, done the same way the barrel-distortion sign was verified (synthetic test, not assumption).
It does **not** yet confirm our convention matches the organizers' definition of "CCW positive,
about the match centre" -- that still needs their sample pairs' ground truth (expected ~29 Aug per
the addendum timeline; not released as of this fix -- checked the WhatsApp transfers folder and
Downloads directly, only the addendum PDF/transcript/CONTINUE_HERE.md are present). The moment
their sample lands, re-run this same comparison against their labelled theta before trusting the
sign on the scored submission.

---

**A "fix" that was tested and correctly REJECTED, with numbers, not intuition:** a regression check
on curated30 (Phase 1 data) showed the hybrid's net path failing badly on one pair (`C26`, 528px
miss -- a case CASES.md documents as a specific drift-shear stress test). The instinct was to add a
safety valve: if net and classical disagree by >20px, trust classical. **Tested before shipping it,
and it made both domains WORSE, not better** -- curated30 dropped 97%->90%, the full-aberration set
dropped 89%->88%, both landing exactly at classical's own baseline. The reason: disagreement between
net and classical does NOT mean net is wrong -- in most of the ~20% of pairs where they disagree by
>20px, **net was the one that was right**, and classical was fooled by a decoy. Blanket-deferring to
classical on disagreement discards every one of those wins along with catching the one C26-style
loss, net negative. **Decision: no valve, ship the unconditional hybrid.** C26's miss is an accepted
residual (classical itself already misses ~10% of curated30, not the ~100% assumed before this was
actually measured) -- not a regression introduced by this design, and not fixable by the naive
mechanism tried. A real fix, if pursued later, would need a signal that actually discriminates who is
right (e.g. the net's own heatmap confidence, or `is_multimatch()` on the net's heatmap the way the
original Phase 1 router did it) -- not raw disagreement distance, which doesn't carry that
information. Noted as a future-work item, not resolved tonight, and not needed to be: the unconditional
hybrid is already a validated improvement over classical alone.

---

## Bigger training set + rotation-aware retrain + 300-pair recalibration (31 Aug)

Executed `docs/TEAMMATE_TASK_LARGER_TRAINING.md` end to end. The headline goal was NOT "more data
for its own sake" -- it was that the shipped `best_phase2.pt` was trained *before* the generator
started baking relative +-5 rotation into `--phase2` pairs, so it had **never seen a rotated pair**.
This retrain uses the rotation-aware generator, so it is the first genuinely rotation-aware net.

**Environment (stated explicitly, per the standing rule about which path a number came from):** all
numbers below are the full router / real GPU path, measured through `.venv` (Python 3.13,
torch 2.11.0+cu128) on an RTX 3050 6 GB Laptop GPU -- NOT the CPU-only fallback. The default PyPI
`torch` wheel installed CPU-only (`2.13.0+cpu`); the CUDA build was installed explicitly from the
cu128 index. Net forward-pass times below are therefore GPU-timed; the CPU-only reference-machine
latency is unchanged from the earlier ~1.6 s/pair hybrid figure because the architecture is
identical (same `DriftMatchNet(C=64)`, same 1.03 MB checkpoint).

**Data generated (deterministic, seeds recorded, NOT committed -- regeneratable):**
- `data/p2train8k` -- 8000 present pairs, full aberration suite, seed 900000 (`--phase2`), ~29 min.
- `data/p2calib300` -- 300 pairs, 22% absent (233 present / 67 absent), seed 950000, ~68 s.
- `data/p2eval100` -- 100 held-out present pairs, seed 910000 (separate from train, used as the
  training `--eval2` held-out set AND the comparison set), ~25 s.

### Retrain (step 3)

`python -m driftmatch.train --data data/p2train8k --resume driftmatch/checkpoints/best_phase2.pt
--epochs 15 --batch 4 --lr 2e-4 --workers 0 --eval1 data/p2train8k --eval2 data/p2eval100
--out driftmatch/checkpoints_new`. On the 6 GB card `train.py` auto-enabled `cudnn.benchmark` +
`pin_memory` (its >=5.5 GB switch) and ran `batch 4`. ~11.5 min/epoch (data-bound: `--workers 0`
single-thread loading, GPU ~87-100%, ~16 s/epoch data-wait). Loss fell monotonically 0.559 -> 0.348.

**Held-out (p2eval100, n=60 quick-eval) peaked EARLY, at epoch 2 (85.0%), then plateaued/declined**
while the on-train `eval1` kept climbing to 94% -- a clean mild-overfit signature. `train.py`'s
"keep best on held-out" logic therefore saved **epoch 2** as `checkpoints_new/best.pt`
(acc_eval 86.0, acc_val 85.0). The resumed `best_phase2.pt`, re-baselined on this same held-out set,
scored **81.7%** (its stored 88.3% was on a different/older eval set -- the re-baseline is why the
resume logic exists, and is the only fair old-vs-new comparison).

### Decisive comparison (step 5), 100 held-out present pairs, real +-5 rotation present

`scripts/compare_checkpoints.py data/p2eval100 <old> <new>`:

| Method | @5px | median err | time/pair |
|---|---|---|---|
| classical alone | 73.0% | 0.539 px | 815 ms (CPU) |
| net alone -- old `best_phase2.pt` | 83.0% | 0.685 px | 49 ms (GPU) |
| **net alone -- new `best_phase2_rot8k.pt`** | **84.0%** | 0.694 px | 43 ms (GPU) |
| hybrid -- new (== `route.predict_full`) | **84.0%** | 0.694 px | 858 ms |

**The new rotation-aware net matches/edges the old on every metric (84 vs 83 @5px here, 85 vs 81.7
on the n=60 quick-eval) and is never worse.** The +1 pt on 100 pairs is within noise on its own, but
it is consistent across both eval slices AND the new net is trained on the correct Phase 2
distribution the old one never saw -- so this is a principled improvement, not just a lucky pair.

**Why 83-84% and not the earlier "89%":** not a regression. This is the FIRST comparison set with
genuine relative rotation stacked on the full aberration suite; the 29 Aug "results dropped" note
already established that once real rotation is in the data the honest localization number is ~81-83%
(the old 89% had no rotation in the eval data). 84% here is consistent with that, and slightly
above it.

**Pose recovery (classical path -- checkpoint-independent, since `theta`/`scale` always come from
classical), 73 well-localized present pairs:**
- scale: median **0.85%** error; tiers <=1% -> 58%, <=2% -> 79%, <=5% -> 92%.
- theta: median **0.190 deg**; tiers <=0.25 -> 62%, <=0.5 -> 79%, <=1.0 -> 90%.

Consistent with (and slightly better than) the 29 Aug rotation figures (median 0.286 deg on a
smaller, harder n=19 subset) -- confirms theta recovery holds on a larger sample. NOTE this task did
NOT touch rotation code (per the task's own "what NOT to do"); this is a re-measurement of the
existing classical pose path on new data, not a change to it.

**Decision -- adopted the new net as the shipped default, kept the old as fallback.** Added
`driftmatch/checkpoints/best_phase2_rot8k.pt` (a copy of `checkpoints_new/best.pt`) and pointed
`register.py`'s `_phase2_ckpt` at it. `best_phase2.pt` is untouched -- reverting the one changed line
in `register.py` rolls back instantly. Rationale: not-worse on every metric + rotation-aware +
identical CPU latency, and leaving the retrain unused would defeat the task. Verified `register.py`
runs the full 6-column contract end to end on the new checkpoint (net loads, no classical fallback).

### Recalibration (step 4) -- 300-pair set, `scripts/recalibrate_found.py data/p2calib300`

peak-NCC separation: present min 0.123 / median 0.933; absent **max 0.967** / median 0.822.

| Threshold | F1 | prec | rec | FN | FP | cost (2xFN) |
|---|---|---|---|---|---|---|
| 0.53 -- cost-optimal (route.py's own 2x-FN methodology) | 0.878 | -- | -- | 6 | 57 | **69** |
| **0.68 -- current, KEPT** | 0.876 | 0.808 | 0.957 | 10 | 53 | 73 |
| 0.73 -- plain-F1-optimal | 0.882 | -- | -- | 13 | 46 | 72 |

> **SUPERSEDED (1 Sep):** this table and the "separability ceiling" reasoning below were measured on
> the SAME broken Set C data. After the generator fix, F1 at 0.68 rose 0.876 -> 0.939 on this very
> seed-950000 set and 0.68 became the cost-optimum outright -- see the final entry. Kept as record.

**`FOUND_PEAK` kept at 0.68.** This is the exact outcome the 29 Aug calibration note predicted:
a bigger set narrows the *estimate* of a threshold, it does not create separability the signal
lacks. The absent MAX (0.967) exceeds the present MEDIAN (0.933), so no single cutoff separates the
classes; F1 sits within 0.006 across the whole plausible range, and the "optimum" swings 0.53-0.73
depending only on whether you weight cost or raw F1. 0.53 (the strict cost-optimum) is too permissive
-- it rejects only 10/67 absent pairs, and "never rejecting scores zero." 0.68 is the validated middle
ground; chasing a 0.006 F1 gain to 0.73 (at the price of 3 more falsely-rejected present pairs, each
of which also zeros its localization + pose credit) is not worth it. Updated the `FOUND_PEAK` comment
in `route.py` to record this. **The +4 bonus (rejection F1 >= 0.90) is out of reach for any
single-threshold rule on this signal** (best F1 here 0.882); the durable fix remains a multi-signal
rejection rule, which 300 labeled pairs (67 absent) still do not comfortably support fitting.

### Artifacts / repo hygiene

- New files: `scripts/recalibrate_found.py`, `scripts/compare_checkpoints.py` (both reads-only,
  reproducible from the seeded sets), `driftmatch/checkpoints/best_phase2_rot8k.pt`.
- Edited: `register.py` (one-line checkpoint pointer + comment), `route.py` (`FOUND_PEAK` comment
  only -- value unchanged).
- NOT committed: `data/p2train8k`, `data/p2calib300`, `data/p2eval100`, `driftmatch/checkpoints_new/`
  -- all regeneratable from the recorded seeds; added to `.gitignore` to prevent accidental staging.

**Standing caveat, unchanged:** every number here is on OUR generator, not the organizers' data, and
the theta sign convention is still only validated self-consistently against our own generator -- the
organizers' sample ground-truth theta (not yet released as of this work) is still required before
trusting the sign on the scored submission.

---

## Epoch-selection verification: was epoch 2 really the best checkpoint? (31 Aug)

Executed `docs/TEAMMATE_TASK_EPOCH_SELECTION.md`. This is a *measurement-correctness* check, not a
tune: the retrain kept epoch 2 because held-out accuracy "peaked" there, but that peak was read off
the training loop's **n=60 quick-eval** -- small enough that "epoch 2 is best" could be a noise
wobble rather than a real peak (loss fell monotonically all the way to epoch 15, so a later epoch
could plausibly be better). Re-decided the question on 100- then 150-pair samples, through the
shipping pipeline, with the exact-rubric scorer (`scripts/score_phase2.py --ckpt`). No retrain
needed -- `checkpoints_new/best.pt` (= epoch 2, = `best_phase2_rot8k.pt`) and
`checkpoints_new/last.pt` (= epoch 15) were both still on disk.

**Step 2 -- full `p2eval100` (100 pairs), localization only (the only component the net affects;
pose/rejection come from the checkpoint-independent classical path):**

| Checkpoint | @5px | credit /40 | median |
|---|---|---|---|
| old `best_phase2.pt` | 83.0% | 31.84 | 0.69 px |
| **epoch 2 (shipped) `best_phase2_rot8k.pt`** | **84.0%** | **32.40** | 0.69 px |
| epoch 15 `last.pt` | 83.0% | 32.08 | 0.63 px |

Epoch 2 is marginally *ahead* of epoch 15 here (+1 pp @5px, +0.32 /40), not behind -- so no later
epoch overtakes it. That already answers the question, so **Step 4 (the ~3 GPU-hour per-epoch
re-run to map epochs 3-14) was skipped** per the doc's own guidance ("skip it if Step 2 gives a
clear answer -- it usually will").

**Step 3 -- fresh, unused `p2test150` (seed 920000, non-overlapping with all train/eval/calib
seeds), the honest number since `p2eval100` is now a selection set:**

| Checkpoint | @5px | credit /40 | median |
|---|---|---|---|
| old `best_phase2.pt` | 82.0% | 31.25 | 0.74 px |
| **epoch 2 (shipped)** | **82.0%** | 31.47 | 0.69 px |
| epoch 15 `last.pt` | 82.0% | 31.84 | 0.66 px |

**All three tie at 82.0% @5px on the fresh set** (0 pp apart). Epoch 15's /40 credit is trivially
higher (31.84 vs 31.47) and its median trivially tighter (0.66 vs 0.69 px), but the decision rule is
defined on @5px, where the gap is zero.

**Decision -- KEEP epoch 2 (`best_phase2_rot8k.pt`), no change.** The rule: on n=150 (binomial noise
~+-4 pp) a difference is real only at >=5 pp @5px; everything here is 0 pp, a tie, and a tie means
the shipped checkpoint stands -- do not manufacture a winner from a 0.4-in-40 credit wobble. The old
net does NOT beat both (it is last on credit, tied on @5px), so no surprise-flag branch triggers.
`register.py` is unchanged -- it still ships `best_phase2_rot8k.pt` on the same one line as before.

**What this verification bought:** it retires the "epoch 2 might be a 60-pair noise artifact" risk.
On 100 and 150 pairs epoch 2 is tied-or-better than epoch 15 everywhere, so shipping it was the
correct call, now decided on a trustworthy sample size rather than 60 pairs. `FOUND_PEAK` left at
0.68, untouched, as the task required (the 300-pair recalibration above already settled it -- a
separability ceiling, not a tuning problem).

Repo hygiene: only `docs/PHASE2_RESEARCH_NOTES.md` and `.gitignore` changed (added `data/p2test150/`
to the ignore list). No code change. `data/p2test150` not committed -- regeneratable from seed
920000.

---

## Fourier-Mellin invariant scale+rotation estimator: built, measured, and rejected (31 Aug)

Executed `docs/TEAMMATE_TASK_FOURIER_MELLIN.md` -- the "invariant formulation" Gokul named on the
orientation call as the other acceptable way to handle unknown scale+rotation, as an independent
cross-check against the existing grid search. **Verdict: it loses decisively, with a diagnosed,
verified root cause -- not a bug, not a sign-convention miss, a real information-availability limit
of this specific problem's geometry.** Built as isolated new files only (`fmt_pose.py`,
`scripts/eval_fmt.py`), per the task's scope rule -- `solve.py`/`route.py` untouched.

**The idea, and why it looked promising going in:** textbook FMT registers two images that are
literally the same content differing by rotation/scale/translation, via log-polar phase correlation
of their FFT-magnitude spectra. Our reference and wide don't share a field of view, so this doesn't
directly apply -- but the die layouts are periodic (DRAM/FinFET arrays), and a periodic lattice's
spatial frequency is a shared, translation-invariant signature present in *both* images: a physical
pitch of P nm appears at P px in the 1 nm/px reference and P/scale px in the wide view, and the whole
layout (lattice included) shares the same relative rotation as the landmark. In principle, the
lattice's frequency peak should shift radially by log(scale) and angularly by theta between the two
spectra -- recoverable by log-polar phase correlation, driven by the periodic background instead of
a shared foreground.

**Built:** Hann-windowed 2D FFT magnitude, tight Gaussian high-pass (deliberately narrow -- a wide
one was tried first and turned out to suppress exactly the low-radius band where a large-pitch
reference's lattice peak lives), log-polar remap (720x720, log-spaced radius), per-ring
whitening (subtract/divide by each radius's across-angle mean/std, to remove the shared broadband
1/f-ish envelope any textured image has, which otherwise dominates a naive correlation), then FFT
phase correlation with a parabolic sub-bin refine -- the same technique `solve.py`'s `_subpixel`
already uses, applied to the log-polar bin index instead of the image plane.

**Diagnosed empirically before trusting any accuracy number (direct spectral inspection, not just
an aggregate score):** on a sample pair, the WIDE image showed a strong, sharp, individually
identifiable frequency spot at radius 41 -- matching the physics prediction (`n/pitch_nm * scale` =
41.0) almost exactly. But the REFERENCE image showed **no corresponding standout peak** at its
predicted radius (~4.3) -- the values there (13.6-14.1) were statistically indistinguishable from
the surrounding broadband floor (also 13.6-14.1). Root cause: the reference's 1 um field of view at
a 70-320 nm pitch (the dominant "resolved" regime, 60% of pairs by generator design) shows only
**3-14 lattice periods** -- too few for a sharp spectral peak (a periodic signal needs many cycles
to produce a narrow, tall frequency-domain spike; the wide view's much larger ~8-12 um FOV gives it
25-170+ periods of the same pitch, hence its clean peak). Unlike textbook FMT, where both images are
rich in the shared periodic content, one side of this correspondence is information-starved by the
problem's own geometry -- the reference is *supposed* to be a small, tight crop.

**Quantitative confirmation, not just one example (60-pair set, seed 810600):** correlation between
the FMT estimator's raw signal and ground truth scale/theta, by regime:

| regime | n | scale corr | theta corr | mean confidence |
|---|---|---|---|---|
| resolved | 39 | -0.186 | -0.276 | 0.0137 |
| aliased | 7 | +0.216 | -0.317 | 0.0133 |
| coarse | 14 | +0.202 | -0.053 | 0.0131 |

All \|r\| < 0.32, inconsistent sign across regimes, confidence uniformly tiny (~0.013, versus
`solve.py`'s peak-NCC routinely 0.7-0.99 on real matches) -- this is noise, not a flipped sign to
fix. Confirmed head-to-head on the full spec'd 200-pair set (seed 810000, `scripts/eval_fmt.py`),
restricted to the 140/200 pairs the classical grid search already localizes <=5px (so both methods
are judged on the same solvable pairs):

| Method | scale median err | scale tiers (<=1/2/5%) | theta median err | theta tiers (<=0.25/0.5/1.0 deg) | runtime/pair |
|---|---|---|---|---|---|
| **grid search (shipped)** | **1.127%** | 44% / 69% / 93% | **0.208 deg** | 58% / 76% / 86% | 1932 ms |
| **FMT** | 90.237% | 0% / 0% / 0% | 44.412 deg | 1% / 1% / 1% | 453 ms |

FMT is ~4x faster (453 ms vs 1932 ms/pair, both well inside the 5 s budget either way) but the
accuracy gap is total, not marginal -- 0% of pairs land in any scale or rotation tier. This is a
clean loss, not a close call needing more tuning.

**Decision: do NOT integrate FMT.** The task's own framing anticipated this as a legitimate outcome
("a tie or a loss is a perfectly good outcome to report") -- reporting it. The root cause (reference
field-of-view too small for the primary pitch regime to produce a usable spectral peak) is a property
of the problem's geometry, not an implementation defect fixable by more tuning of this approach; a
genuinely different design (e.g. estimating the wide image's own periodicity independently, without
needing a matching reference-side peak, then cross-checking candidate scales some other way) would
be a materially different, larger undertaking than "one shot invariant formulation" and was not
pursued further given the remaining time before submission and the higher-priority open items
(rejection/calibration ceiling, CPU-latency benchmark, organizers' sample data still unconfirmed).

**Not committed:** `data/fmt_calib20/`, `data/fmt_calib60/`, `data/fmt_test200/` -- regeneratable
from seeds 810500 / 810600 / 810000, added to `.gitignore`. Committed: `fmt_pose.py`,
`scripts/eval_fmt.py`, this entry -- `solve.py`/`route.py` untouched, per the task's isolation rule.

---

## Set C absent-pair generator bug fixed -- the "separability ceiling" was ours, not the signal's (1 Sep)

Executed `docs/TEAMMATE_TASK_ABSENT_PAIR_FIX.md`. **This reverses the 29 Aug "signal-separability
ceiling" conclusion**: it was never a signal limit, it was a generator defect. Applied Materials'
own Phase 2 generator spec (Section 4, shared 31 Aug) warns verbatim that cutting the decoy from a
canvas with the *same zone geometry* makes absent pairs score *higher* peaks than present ones. Our
generator did exactly that.

**The bug.** `driftsense/raster.py::make_pair(absent=True)` rendered the absent wide from the
reference's OWN `Layout`, only stripping the landmark shapes (`layout.shapes = []`). The periodic
arrays -- pitch, phase, roughness, everything -- were byte-identical to the reference's background,
so the reference's periodic surround matched the decoy *perfectly at a shifted position*. That, not
any property of NCC, produced the artificially-high absent peaks.

**The fix (generator only; localization/pose/net untouched).** Absent wides now render from an
**independently instantiated decoy `Layout` of the same architecture family**, built in
`generate_dataset.py`'s absent branch and passed to `make_pair` via a new `wide_layout=` argument
(when given, the reference renders from `layout`, the wide from `wide_layout`, which keeps its OWN
landmark lattice -- a landmark-stripped render is itself a detectable signature the spec warns
against). Decoy design:
- **Same family:** `style` forced equal to the reference (`dram`/`finfet` never switched).
- **Same pitch regime/band:** decoy `regime = spec.regime`, pitch redrawn within that band -- so the
  decoy is the same feature scale (a genuine hard negative, no trivial pitch-outlier tell).
- **Different lattice:** the redrawn pitch differs from the reference's, so the two lattices do NOT
  co-register. Critical subtlety: a phase-only shift at *identical* pitch would still correlate
  cleanly at the shifted position -- the pitch itself must differ. Phase, roughness and jitter also
  differ (decoy layout seed = `spec.seed ^ 0xDEC0`, pitch draw = `spec.seed ^ 0xDEC0BA5E`), fully
  deterministic so the set still reproduces from `--seed`.
- **Same rotation distribution:** the decoy wide is rendered at the reference's tilt minus the same
  relative-theta a present wide uses, so absent and present wides share one +-5deg relative-rotation
  distribution (otherwise a wider absent rotation would itself leak).

**Systematic-signature self-audit (spec Section 4 requires this).** After the fix, what could a
solver exploit to detect absent pairs *other than* the intended low correlation? Family, pitch band,
noise/aberration channel (applied via `spec.wide`, identical to present), rotation range, and the
presence of a centre landmark are all matched between present and absent. The one remaining
difference is the intended, physical one: the decoy's periodic phase and exact pitch are uncorrelated
with the reference's -- i.e. it is genuinely a different die region. There is no artificial tell such
as "landmark missing from centre" or "decoys are noisier." Residual honest hardness: a few decoys
whose redrawn pitch lands close to the reference's still score high peaks (absent max 0.877 on calib,
0.933 on the fresh set), which is exactly why F1 is ~0.93, not a saturated 0.99.

### Step 1 -- before/after peak distribution, same seed-950000 set (apples-to-apples)

| | present min / median / max | absent min / median / max | absent max < present median? |
|---|---|---|---|
| **before (bug)** | 0.123 / 0.933 / 0.989 | 0.053 / 0.822 / **0.967** | NO -- 0.967 >= 0.933 |
| **after (fixed)** | 0.123 / 0.933 / 0.989 | 0.034 / **0.533** / **0.877** | **YES -- 0.877 < 0.933, CLEAN** |

The present distribution is **identical** before and after (min/median/max unchanged) -- the fix
touched only absent pairs, as intended. Absent peaks collapsed (median 0.822 -> 0.533).

### Step 2 -- re-swept `FOUND_PEAK`: KEPT at 0.68

On the corrected seed-950000 set, `scripts/recalibrate_found.py`: at 0.68, F1 **0.939** (was 0.876
on the broken data), FN 10, FP 19, cost 39 (was 73). **0.68 is the cost-optimum outright now** (2x
FN weight), so the value is unchanged -- but it is now a genuinely good operating point, not the
provisional middle-ground it was on the broken data. (Plain-F1-optimum 0.71 gives 0.940, +0.001 --
noise.) No `FOUND_PEAK` value change; comment updated to record this.

### Step 3 -- per-signal AUC on corrected data: raw peak alone now wins

| signal (higher = more present) | AUC before (bug) | AUC after (fixed) |
|---|---|---|
| **raw peak NCC** | 0.789 | **0.945** (calib) / **0.947** (fresh) |
| distinct (`peak * (1 - second_ratio)`) | 0.830 | 0.781 / 0.799 |
| 1 - second_ratio | 0.793 | 0.705 / 0.711 |

On the broken data `distinct` beat raw peak (0.830 > 0.789) -- which is *why* a multi-signal rule
looked necessary. On the corrected data **raw peak is decisively best and the auxiliary signals are
worse**, so the multi-signal rule is NOT needed -- ship raw peak (which is exactly what `route.py`
already does). The 29 Aug PSR/second_ratio negative results were correct *about the broken data* and
are now moot.

### Step 5 -- full rubric score, FRESH unused set (seed 960000, 219 present / 81 absent)

`scripts/score_phase2.py data/p2reject_test300 --ckpt driftmatch/checkpoints/best_phase2_rot8k.pt`:

| rubric line | before (bug, ~) | after (fixed, fresh set) |
|---|---|---|
| **Rejection F1 (15)** | ~0.88 | **13.97 / 15**, F1(present+) **0.932**, macro 0.860, absent+ 0.789 |
| Calibration AUC (10) | ~0.66 | 6.12 / 10, AUC 0.612 |
| Localization (40) | -- | 33.06 / 40 (85.8% @5px) -- unchanged, checkpoint-independent |

Fresh-set peak separation is CLEAN too (absent max 0.933 < present median 0.942, raw peak AUC 0.947).

**+4 bonus (rejection F1 >= 0.90) verdict:** **reachable under the present+ convention (0.932), on a
fresh set** -- it was declared out of reach on the broken data. It is NOT yet reachable under macro
(0.860) or absent+ (0.789), because a handful of same-band decoys still slip through and FP hurts the
smaller absent class hardest. Reporting all three per the Q2 ambiguity. This is a real improvement
(F1 0.88 -> 0.93, AUC 0.79 -> 0.95), **not** a saturated/leaky 0.99 -- the red-flag check passes.

**Calibration AUC** did NOT improve (0.66 -> 0.61, roughly flat/slightly down). Honest read: `score`
is raw peak, which now separates present/absent superbly (great for rejection) but is a weak
localization-confidence on present pairs, and the calibration metric mixes both. Improving `score`
as a calibration signal is separate future work, out of this task's Set C scope.

### Phase 1 byte-identity regression gate -- PASSED

- seed 7000 (no `--phase2`) still renders `scale=10.0` (present path untouched; the fix is gated on
  `absent and wide_layout is not None`).
- `solve.locate` on curated30 C00 still returns **559.904, 470.001** (exact match to the documented
  Phase 1 recovery). `solve.py` was not touched.

### Files

- Edited: `driftsense/raster.py` (`make_pair` gains `wide_layout=`), `generate_dataset.py` (absent
  branch builds the decoy layout), `route.py` (`FOUND_PEAK` comment only; value unchanged at 0.68).
- New: `scripts/measure_reject_signals.py` (before/after distribution + per-signal AUC).
- Old "separability ceiling" entries annotated as superseded (not deleted -- honest record).
- NOT committed: `data/p2calib300` (regenerated), `data/p2reject_test300` (seed 960000) -- gitignored.

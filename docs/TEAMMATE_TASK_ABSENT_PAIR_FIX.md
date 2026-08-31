# Task: fix the Set C absent-pair generator, then fully re-validate rejection + calibration

Read [`docs/CONTINUE_HERE.md`](CONTINUE_HERE.md) first, then this file. **`git pull` before
you start.** This is a **generator correctness fix** with a mandatory, rigorous re-validation
afterwards — not a threshold tune. Needs `torch` only for the final full-rubric score (step 6);
steps 1–5 are pure classical (numpy/scipy/PIL), so most of the work runs in the plain `.venv`.

> **Ownership / no collision:** this task edits `driftsense/raster.py`, `driftsense/sampling.py`,
> possibly `generate_dataset.py`, and the `FOUND_PEAK` value in `route.py`. **No other session may
> touch those files while you hold this task.** If `git pull` shows someone already changed the
> absent-pair path, stop and report back instead of merging blind.

---

## Why this task exists (read this — it reverses an earlier conclusion)

Applied Materials shared their own internal Phase 2 generator specification (mentor doc, 31 Aug).
Its **Section 4** warns, verbatim:

> "If you cut the decoy reference from a canvas with the **same zone geometry**, you will find that
> absent pairs score **higher** correlation peaks than present pairs... a generic periodic crop
> matches *somewhere* in any periodic image, and it matches there cleanly, whereas a true instance
> has to match at one specific place through noise and degradation."

**Our generator has exactly this bug.** `driftsense/raster.py::make_pair(absent=True)` renders the
absent wide from the **same `Layout` object** as the reference, only stripping the landmark shapes
(`layout.shapes = []`, line ~208). The periodic arrays — pitch, phase, everything — are byte-identical
to the reference's background. So the reference's periodic surround matches the wide's periodic
surround *perfectly*, producing the artificially-high absent-pair peaks we measured (absent max 0.97
vs present median 0.94 — statistically indistinguishable).

**This invalidates the 29 Aug "signal-separability ceiling" conclusion.** We concluded rejection was
a fundamental limit needing a multi-signal classifier and more data. That conclusion was drawn from
data produced by this bug. It is very plausibly **not** a ceiling — it is a generator defect with a
known fix. All current rejection/calibration numbers (`FOUND_PEAK = 0.68`, F1 ~0.88, AUC ~0.66) were
measured against the broken distribution and are now **provisional**, not settled.

The theta-sign and scale conventions were separately confirmed correct against the same mentor doc
(0.03 px agreement) — those are done, do **not** revisit them. This task is only Set C.

---

## The fix (principle first, then the invariant that must hold)

**Principle:** an absent pair's wide image must be rendered from an **independently instantiated
layout of the SAME architecture family** (dram stays dram, finfet stays finfet) — a genuinely
different die region with a different pitch/phase realization — **not** the reference's own layout
with the landmark erased.

**Invariants (all must hold):**

1. **Different instantiation.** The absent wide's layout must draw its own `pitch`, `phase`, and
   per-array jitter from a *different* seed than the reference's layout, so the two periodic lattices
   do **not** align identically. This is the whole fix.
2. **Same family.** Same `style` (`dram`/`finfet`) as the reference. The mentor doc is explicit:
   *"keep decoys within the same architecture family — cross-family decoys are trivially detectable,
   and are not acceptable."* Do not switch dram↔finfet.
3. **Still a hard negative.** The decoy wide stays periodically plausible (same family, similar pitch
   *band*, realistic noise/aberrations). The goal is "a different die region," not "an obviously
   unrelated image." Do not overcorrect into a trivially-separable negative (see the failure mode in
   step 5's decision rule).
4. **Determinism preserved.** Derive the decoy layout's seed deterministically from the pair seed
   (e.g. `decoy_seed = spec.seed ^ 0xDEC0`), so the whole set still reproduces byte-identically from
   `--seed`. Do not introduce an unseeded RNG.
5. **Phase 1 byte-identity untouched.** The absent path is only reachable when `--absent-fraction > 0`
   (Phase 2 only). Present-pair rendering must not change. **Verify** seed 7000 (no `--phase2`) still
   produces `scale=10.0` and `solve.locate` on curated30 C00 still returns `559.904, 470.001` — the
   standing regression gate.

**Recommended implementation shape** (you have the code loaded — choose the cleanest, but this is the
intended structure): build a second `Layout` for the decoy in `generate_dataset.py`'s absent branch
(a fresh `build_layout` on a perturbed spec/seed of the same `style`), and pass it into `make_pair`
as a new optional `wide_layout=` argument. When `wide_layout` is given, `make_pair` renders the
*reference* from `layout` and the *wide* from `wide_layout` (with its own shapes — a real die region
has its own features; do **not** strip shapes, since a landmark-stripped render is itself a detectable
signature the mentor doc's Section 4 warns about). Keep the current same-layout path only if you can
show it is never hit for absent pairs after the change (it shouldn't be).

Whatever shape you pick, **write down** in the report what systematic signature your decoys now carry
and how a solver could exploit it (mentor doc Section 4 requires this self-audit — e.g. "decoy pitch
is drawn from the same band, so the family is undetectable, but the phase is uncorrelated with the
reference"). This is graded generator-quality material, not busywork.

---

## Validation protocol (non-negotiable — same rigor as every other entry in the research notes)

### Step 1 — regenerate the calibration set with the SAME seed, measure before/after separation

Regenerate the exact existing calibration configuration so the comparison is apples-to-apples:

```
python generate_dataset.py --phase2 --absent-fraction 0.22 --pairs 300 --seed 950000 --workers 5 --out data/p2calib300
```

Then measure present-vs-absent peak-NCC separation the same way `scripts/recalibrate_found.py` does.
**The headline number:** does the absent-pair peak **max** now drop below the present-pair
**median**? Before the fix it did not (absent max 0.967 > present median 0.933). Report both
distributions (present min/median/max, absent min/median/max) before and after in one table.

### Step 2 — re-sweep `FOUND_PEAK` cost-optimally

```
python scripts/recalibrate_found.py data/p2calib300
```

This sweeps the threshold minimizing cost (false-reject weighted 2× a false-accept, since a
false-reject also zeros localization+pose credit — the methodology is already in the script and in
`route.py`'s comment). Record the new cost-optimal threshold, its F1, and the FN/FP counts. Update
`route.FOUND_PEAK` **only if** the new value differs meaningfully **and** step 5's sanity check passes.

### Step 3 — re-test the previously-rejected signals on the CORRECTED data

The 29 Aug work rejected `second_ratio`/`distinct` and PSR as rejection signals — but on the *broken*
distribution. Re-check, cheaply, whether raw peak NCC alone now suffices, or whether a combined signal
now helps on the corrected data. `solve.locate(..., return_info=True)` already returns `score`,
`second_ratio`, and `distinct`. Report AUC of each candidate signal against the absent/present label
on `p2calib300`. **If raw peak alone now cleanly separates, we do not need a multi-signal rule — that
is the win.** If a combined signal still helps, note it but do not build a classifier on 67 absent
pairs (that overfit risk was real and still is).

### Step 4 — honest final numbers on a FRESH, unused set

`p2calib300` is now a *selection* set (you tuned the threshold on it). Generate a genuinely fresh
mixed set with a new seed and report the honest rejection/calibration numbers there:

```
python generate_dataset.py --phase2 --absent-fraction 0.22 --pairs 300 --seed 960000 --workers 5 --out data/p2reject_test300
```

(Seed 960000 does not overlap any train/eval/calib seed: 900000–908000 train, 910000 eval, 920000
test, 950000 calib.)

### Step 5 — full rubric score, before vs after

Run the exact-rubric scorer on the fresh set with the shipped checkpoint (this needs `torch`):

```
python scripts/score_phase2.py data/p2reject_test300 --ckpt driftmatch/checkpoints/best_phase2_rot8k.pt
```

Record the **Rejection F1 (15)** and **Calibration AUC (10)** lines. `score_phase2.py` already reports
F1 for both positive-class conventions (`present+`, `absent+`, and macro), so the Q2 ambiguity about
which class is positive is covered — report all three.

**The two questions this answers:**
- Did Rejection F1 and Calibration AUC actually improve vs the pre-fix numbers (F1 ~0.88, AUC ~0.66)?
- **Is the +4 bonus (rejection F1 ≥ 0.90) now reachable?** It was declared out of reach on the broken
  data. Re-decide it on the corrected data.

### Decision rules (explicit, so there is no "tune until it looks good")

- **Clean separation now (absent max < present median), F1/AUC materially up:** update `FOUND_PEAK`,
  update `route.py`'s comment and the research notes to record that the "separability ceiling" was a
  generator bug, now fixed. This is the outcome we expect.
- **Suspiciously perfect (F1 ≥ 0.98, AUC ≥ 0.98):** treat as a red flag, not a triumph — it likely
  means the decoys became *trivially* separable (a different distribution a solver can detect, which
  the mentor doc warns is unacceptable). Check: are decoy pitches drawn from the same band as present
  pairs? Is any other property (noise level, aberration range) accidentally different between present
  and absent? Fix the leak and re-measure. Honest target is a real improvement (F1 ~0.88 → ~0.92+, AUC
  ~0.66 → ~0.80+), not a saturated 0.99.
- **Little or no improvement:** then the 29 Aug ceiling conclusion may actually have been right after
  all — report that honestly with the numbers, and the multi-signal-rule path stays open as real
  future work. Do not force the number.

---

## What to report back

Append a dated section to [`docs/PHASE2_RESEARCH_NOTES.md`](PHASE2_RESEARCH_NOTES.md), same style as
the existing entries, with:
- the before/after present-vs-absent peak distribution table (step 1),
- the re-swept `FOUND_PEAK` and whether you changed it (step 2),
- the per-signal AUC comparison on corrected data (step 3),
- the fresh-set Rejection F1 (all three conventions) and Calibration AUC (step 5), before vs after,
- an explicit verdict on **+4 bonus reachability**,
- your decoy design and its systematic-signature self-audit (mentor doc Section 4),
- the Phase 1 byte-identity regression check result (seed 7000 + curated30 C00).

Also update `route.py`'s `FOUND_PEAK` comment and, if the fix worked, correct the "separability
ceiling" framing in the earlier research-notes entries with a one-line forward-reference (do not
delete the old entry — it's part of the honest record; annotate it as superseded).

## What NOT to do

- **Do NOT touch theta/scale/localization code** — those are confirmed correct; unrelated to Set C.
- **Do NOT retrain the net** — this is a generator + rejection-threshold task; localization is
  checkpoint-independent for rejection.
- **Do NOT build a learned rejection classifier** on 67 absent pairs — overfit risk, explicitly out
  of scope. If raw peak now works, ship raw peak.
- **Do NOT commit the generated datasets** (`p2calib300`, `p2reject_test300`) — large, regeneratable
  from seeds; they are already gitignored, keep them so.
- **Do NOT `git add -A`** — stage explicitly.

## Commit / push rules (strict)

- Commit under your own git identity (same as your previous commits — no change).
- **No co-author or trailer lines of any kind, in commit messages, code comments, or docs.** Everything
  reads as your own hand-written work. Scan the diff and strip any auto-generated attribution before
  pushing.
- `git status` before committing; stage explicitly:
  `git add driftsense/raster.py driftsense/sampling.py generate_dataset.py route.py docs/PHASE2_RESEARCH_NOTES.md`
  (only the files you actually changed).
- `git pull --rebase` right before pushing.

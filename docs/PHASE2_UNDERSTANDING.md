# SEMICON India Hackathon 2026 — PS-02 Phase 2 (Applied Materials)
## Registration under Unknown Pose

**Status:** Understanding phase only. No code changes made yet.
**Compiled:** 28 Aug 2026. Sources: `Semicon_India_Hackathon_Problem_Statement_with_Phase2.pdf` (12 slides,
addendum) + the Phase 2 orientation call, 27 Aug 2026, 1:30 PM, presenter **Gokul Ganesan**
(Computer Vision Engineer, Applied Materials India), 49m45s — full transcript extracted to
`scripts`-adjacent scratch, key quotes below.

We qualified: **team "The T guys", rank 8/15, ID 1050**, for PS-02 (DriftSense).

---

## 1. What changes in Phase 2 (deltas over Phase 1 — everything else still applies)

| | Phase 1 (as issued) | Phase 2 (this addendum) |
|---|---|---|
| Zoom ratio | Exactly 10×, given | **Unknown — uniform in [8×, 12×], different per pair.** You're told the range, not the value. |
| Rotation | Injected as noise, 1–3° (survives, you don't report it) | **Unknown, ±5° either way — and must be reported.** |
| Reference presence | Always present | **~20% of pairs (Set C) contain no true instance.** |
| Required output | `x, y` | **`x, y, theta, scale, found, score`** — 6 columns |

Direct quote on why these three and not something else (slide 3): *"magnification calibration
drifts and the wafer sits with a small residual tilt — the exact-10× premise is the one Phase 1
assumption a real tool cannot guarantee."* Gokul reiterated this three times on the call: **zoom
[8,12], rotation ±5°, presence/absence — those are the only three things added.**

Critical framing from the call (repeated when a participant — our own Aryan, joining as
`achourasia_be24`, asked if a completely different approach was allowed): **"Do we rewrite our
Phase 1 method? Highly recommend not... it's just an extension."** A method "materially different
from your Phase 1 declared approach" is explicitly a **disqualifying, no-appeal** condition (slide 9).
**DriftRoute (classical DriftFind + learned DriftMatchNet, router-dispatched) is the approach we
extend — we do not redesign it.**

---

## 2. Phase 2 dataset — 200 blind pairs, organizer-generated (slide 4)

Same geometry as Phase 1: grayscale, 1000×1000 px, [0,0] top-left. **Teams never see the images —
only their scores.**

| Set | Pairs | Description | Feeds |
|---|---|---|---|
| **A — Nominal** | 70 | Reference present, noise ≈ Phase 1 sample, full [8,12]× and ±5° range | Localization + pose |
| **B — Degraded** | 70 | Reference present. Charging, scan distortion, defocus, elevated shot noise, polygon scaling ±20% — **4 undisclosed severity levels** | Localization + pose |
| **C — Absent** | 40 | No true instance. A *different die region of the same architecture* — plausible, periodically similar. Correct answer: `found=0` | Rejection F1 |
| **D — Optical (bonus)** | 20 | RGB 3-channel optical-microscope analogue, reference present | Bonus only, +6 pts, **only unlocks if grayscale score is already strong** |

Noise model *categories* are disclosed; exact parameters/severity ladder are **not**, deliberately —
quote: *"otherwise Phase 2 collapses back into the self-scored loop that Phase 1 already is."*
**We must regenerate our own dataset** (transcript, repeated twice) — our Phase 1 generator only
produces fixed-10×, always-present pairs, which gives nothing to tune the rejection threshold
against.

---

## 3. Output contract (slide 5) — replaces our current `infer.py` signature

```
python register.py --input pairs.csv --output predictions.csv
```

One entry point, exact signature — not a notebook, not interactive (they run 15 teams
back-to-back on one reference machine). **The slide itself is headed "ONE ENTRY POINT, EXACT
SIGNATURE" — `register.py`, `--input`, `--output`, `pairs.csv`, `predictions.csv` are all literal,
mandatory names, not illustrative.** (On the call Gokul hedges verbally — *"I'm just saying a
name, register.py would be your Python file"* — but the printed slide overrides that: it is the
written spec, and it says "exact.") Rename our current `infer.py` CLI to this exact invocation
rather than adding a same-purpose script under a different name.

`predictions.csv`, one row per `pair_id`, **every pair_id exactly once — a missing row scores zero**:

| Column | Meaning |
|---|---|
| `pair_id` | as supplied in `pairs.csv` |
| `x, y` | match centre in wide-search coords, float, sub-pixel allowed |
| `theta` | rotation in degrees, **CCW positive**, about the match centre |
| `scale` | recovered downscaling factor, nominally in [8, 12] |
| `found` | 1 or 0. **When 0, write 0 in all pose columns** |
| `score` | our own confidence, any monotonic scale (not compared cross-team — only checked for internal monotonicity/AUC against our own correctness) |

**Reference machine:** 4-core x86 CPU, 8 GB RAM, **no GPU**, no network, Python 3.11, weights ship
inside the zip. **Runtime budget: median ≤5 s/pair, hard timeout 20 s = zero for that pair.**

This is the most consequential engineering fact for us: our shipped path is GPU-timed
(150–430 ms/pair on an RTX 3050). **We have never benchmarked DriftMatchNet forward-pass latency on
CPU-only torch, and the grading machine has no GPU at all.** This must be measured before
anything else — if it's slow, the router needs a CPU-aware branch, not just a CUDA-unavailable
fallback to classical (which itself was ~770 ms/pair, fine, but check the router doesn't call the
net anyway if it silently detects no CUDA and still eats the runtime cost of loading a bigger model).

Also: model weights via a state dict/checkpoint, not a live `torchvision.models(pretrained=True)`
call (would attempt a download → network access → disqualification). Confirm none of our code paths do this.

---

## 4. Scoring — 100 pts + 10 bonus (slide 6, tiers on slide 7)

| Pts | Criterion | Detail |
|---|---|---|
| 40 | **Localization** | Sets A+B, present pairs only. Tiered credit on Euclidean error: ≤1px=1.00, ≤2px=0.80, ≤3px=0.60, ≤5px=0.40, >5px=0.00. Set score = mean credit. **Total = 0.45·A + 0.55·B** (degraded weighted higher — "holding up under bad conditions is the interesting part"). |
| 20 | **Pose recovery** | Scale 10 pts + rotation 10 pts. **Scored only where localization credit > 0** — a pose attached to the wrong tile is noise, not signal (explicit design choice, stated twice). Scale tiers: ≤1%→1.00, ≤2%→0.60, ≤5%→0.30. Rotation tiers: ≤0.25°→1.00, ≤0.5°→0.60, ≤1.0°→0.30. |
| 15 | **Rejection** | F1 on `found` across all 180 grayscale pairs (A+B+C). **Never rejecting anything scores zero here** — an always-`found=1` policy cannot reach top 10 on localization alone; rejection is mandatory, not optional. |
| 10 | **Confidence calibration** | AUC of our `score` column vs. per-pair correctness on the blind set. Not compared between teams — only checked for internal monotonicity. |
| 5 | **Efficiency** | Relative quartile ranking on median wall-clock/pair. Not expecting production-grade code, but it counts. |
| 10 | **Generator, citations, failure analysis** | Carried forward from Phase 1, **re-judged under Phase 2 conditions** — new failure modes (scale/rotation extremes, absent case) need their own honest write-up, not a copy-paste of the Phase 1 one. |
| **+10 bonus** | +6 if Set D credit ≥0.40 **and** Sets A–C ≥0.50; +4 if rejection F1 ≥0.90. **Cannot lift a team above 100 for ranking** — it's a tie-breaker only (2nd, after Set B credit; then rejection F1; then median error; then median runtime). |

Confirmed on the call: only **5 teams per problem statement** advance to the grand finale (10 total
across both tracks), from the current 15 shortlisted for PS-02.

---

## 5. Allowed vs. disqualifying (slide 9, no appeal on the right column)

**Allowed:** extending our Phase 1 method to search the disclosed [8,12]/±5° ranges, or moving to
an invariant formulation; regenerating our own dataset with the new ranges + absent pairs (needed
to tune the rejection threshold — we have nothing to tune it against otherwise); hard-coding the
disclosed bounds; data augmentation, retraining, hyperparameter/threshold changes; classical,
learned, or hybrid — judged equally, no bias toward deep learning.

**Disqualifies, no appeal:** any network access during the scored run; hard-coding, filename
fingerprinting, or reading outside the supplied paths; a method materially different from our
declared Phase 1 approach; proprietary/non-public fab layout data in the generator; **mixing
organizer test/validation data into training, in either direction** — when they release the ~15–20
pair validation set (promised "by Monday", i.e. ~31 Aug), it is a validation fold, not training data.

---

## 6. Timeline (T = 27 Aug, Phase 1 results / Top-30 announcement)

| When | What |
|---|---|
| T+0 (27 Aug) | Addendum released to the 30 shortlisted teams |
| T+2 (~29 Aug) | Sample `pairs.csv` format + **3 unscored sample pairs with full ground truth** published — validates our I/O contract, not scored |
| ~31 Aug ("by Monday", per call) | Organizer-generated validation set, ~15–20 pairs, with ground truth — for us to self-check, not for training |
| T+3 (~30 Aug) | Questions close — deadline for I/O-contract clarifications |
| **T+7 = 3 Sep 2026, 23:59** | **Submission due. Code frozen — no resubmission.** |
| T+8–9 | Organizers execute every team's submission on the reference machine |
| T+10–11 (~6 Sep) | Top 10 finalists announced, Phase 3 begins |

**Note the email/PDF say 3 Sep 2026, not the hackathon site's general "4 Sep" — the PS-02-specific
addendum date governs.** Updating memory to reflect this.

**Discrepancy worth flagging:** slide 5's footnote says the 3 sample pairs *"ship with this
addendum"* (present tense, implying now), while slide 10's timeline explicitly schedules them for
**T+2**. As of 28 Aug the local WhatsApp transfers folder has only the addendum PDF, the
transcript, and two unrelated screenshots — no `pairs.csv`, no sample images. Treat T+2 (~29 Aug)
as the real date and re-check the WhatsApp/email attachments then; don't assume they're late if
they land on the 29th, and don't assume they're missing if slide 5's wording sends you looking for
them today.

---

## 7. Gap analysis — current repo vs. what Phase 2 needs

Current submission (`README.md` §3–4): `infer.py` → `predict(reference_path, search_path) -> (x, y)`,
served by `route.py` (DriftRoute), dispatching between `solve.py` (DriftFind, classical NCC with a
fixed 10× block-average downsample + blur×angle sweep at ±4°) and `driftmatch/` (DriftMatchNet,
trained for a fixed-10× reference crop). 94.5%/86% @5px on the organizers' Phase 1 generator.

Concrete deltas needed, roughly in the order they block each other:

1. **New entry point `register.py`** wrapping the existing `predict()` — reads `pairs.csv`, writes
   `predictions.csv` with the 6-column contract, one row per `pair_id`, never crashes on a bad row
   (a missing row scores zero, so a crash is strictly worse than writing a zero/`found=0` row).
2. **CPU-only runtime benchmark, first, before any algorithm work** — measure `driftmatch/infer.py`
   forward-pass latency with CUDA unavailable, on something close to 4-core/8GB. If it blows the
   5 s median or 20 s hard cap, the router needs a machine-aware branch (or the classical path
   becomes primary on CPU-only submission).
3. **Scale search**: `solve.py`'s block-average downsample is hardcoded to 10×. Needs to become a
   3D coarse-to-fine search (scale × blur × angle) over scale ∈ [8,12], with a refinement step
   (parabolic fit, like the existing sub-pixel-on-(x,y) refinement) to hit the ≤1%/2% scale tiers —
   a handful of coarse scale samples won't reach that precision on their own.
4. **Rotation search precision**: current ±4° at 5 samples (~2° steps) is coarse. Pose-recovery
   scoring wants ≤0.25°/0.5° tiers — needs the same coarse-then-refine treatment we already do for
   (x,y), applied to θ.
5. **`driftmatch/` retraining**: the net was trained on fixed-ratio 100×100 reference crops. It
   needs exposure to scale ∈[8,12] and rotation ∈±5° (up from whatever Phase 1's noise-survival
   range was) — likely via an added pre-resample-to-canonical-100×100 step at several hypothesized
   scales (feeding the existing architecture unchanged) or an augmentation-based retrain. Also
   needs **negative (absent) pairs in training** so the heatmap peak height is a meaningful
   rejection signal, not just noise on inputs it's never seen not-matching.
6. **Rejection (`found`) logic**: no existing mechanism. Needs a calibrated threshold on the
   router's confidence (NCC peak / heatmap max, or a blend) tuned on our own regenerated dataset
   (with absent pairs) to optimize F1 — remembering "never rejecting" scores zero here, so the
   threshold must actually fire sometimes.
7. **`score` column**: expose a monotonic confidence value (candidate: the same signal used for
   `found`, e.g. peak NCC/heatmap value) and empirically sanity-check its AUC against our own
   correctness labels before submitting.
8. **`generate_dataset.py` / `driftsense/`**: add uniform scale sampling in [8,12], rotation in
   ±5°, an absent-pair mode (different die region, same architecture, periodically similar —
   probably: render two non-overlapping regions of one layout and pair them), and a documented
   4-tier severity ladder for the degraded set. Must NOT copy organizer parameters (undisclosed by
   design) — only the disclosed categories.
9. **Docs carried forward but re-judged**: `GENERATOR_SPEC.md` citations extend to the new
   parameters; failure analysis needs new honest cases from the new axes (extreme scale/rotation,
   absent-pair false positives/negatives) — the old Phase 1 failure list is necessary but not
   sufficient.
10. **Validate against the 3 sample pairs** the moment they land (~29 Aug) — pure I/O-contract
    check, not a scoring signal.

---

## 8. Open items / watch list

- Sample `pairs.csv` + 3 ground-truth pairs: not yet released (checked local WhatsApp transfers
  folder as of 28 Aug — only the addendum PDF, transcript, and two unrelated screenshots present).
- Validation set (~15–20 pairs): promised "by Monday" (~31 Aug), not yet released.
- Aryan asked on the call whether Phase 1 per-category scores/weak points could be shared; organizers
  declined for now ("you've already crossed that stage... focus on scoring this round") but said
  they'd discuss internally and get back — no commitment.
- "Materially different approach" wording is a real constraint on us: any temptation to rip out
  DriftRoute for something cleaner is explicitly against the rules here, not just inadvisable.

# Organizer Materials — Full Digest (read 2 Sep 2026)

Everything Applied Materials / i4C actually gave us for Phase 2, read end-to-end and
distilled here with **what each item implies for our shipped system**. Written because the
most valuable document in the pack — the dataset-generator prompt — had never been read into
the repo docs, and it answers several questions we had been treating as open.

**Sources digested** (all shared via WhatsApp / SharePoint by mentor Sourabh Ubale):

| Source | What it is | Repo status before this digest |
|---|---|---|
| `Applied Materials_Prompt for phase 2 dataset.docx` | The assignment used to BUILD the Phase 2 dataset generator | **Never read into docs** — the big gap |
| `Applied Materials PS Phase 2 Session _transcript.docx` | 49-min mentor session, 27 Aug, Gokul Ganesan | Captured accurately in `PHASE2_UNDERSTANDING.md` |
| `Semicon_India_Hackathon_Problem_Statement_with_Phase2.pdf` | Official PS + Phase 2 addendum | Captured in `PHASE2_UNDERSTANDING.md` |
| Slide 33 "Output Contract and Run Environment" | The I/O contract slide | Captured; `register.py` conforms |
| `data/organizer_sample/` | 20 real pairs + `ground_truth.csv` | Used, gitignored, correct |

---

## 1. THE BIG ONE — the dataset-generator prompt

This document is the assignment that produced the data we are graded on. It was also handed
to teams ("we have already given you the prompt" — transcript, ~31 min), so it is
simultaneously (a) a description of how their data was built and (b) guidance for our own
generator deliverable. **It tells us more about the blind 200-pair set than anything else we
have.**

### 1.1 Conventions — both of ours are CONFIRMED CORRECT

Two conventions we had resolved empirically are now confirmed against their written spec.
Both were previously listed as residual risk; they are now closed.

**`scale` = `z`, the search pixel size in nm/px, in [8,12] — NOT the ratio `1/z`.** Their §2.3
fixes this by fiat and says explicitly: *the two differ by a factor of ~100, and a solver that
guesses wrong fails every pose comparison*. `solve.PHASE2_SCALES` is `8.0 → 12.0`, i.e. we
report `z`. **Correct.**

**Rotation sign.** Their §2.2 pins the transform exactly:

```
p_search = (1/z) * R(theta) * (p_canvas - c_canvas) + c_search
R(theta) = [[ cos t, sin t],
            [-sin t, cos t]]        t = radians(theta)
```

positive `theta` turns the pattern **counter-clockwise as displayed**. Our `THETA_SIGN = +1.0`
([solve.py:417](../solve.py#L417)) matches, and it now has four independent confirmations: the
synthetic sign test, their ground truth (theta 10/10), their generator source, and now their
written spec. **Closed.**

### 1.2 NEW — the grid-alignment trap (their §2.4)

Not previously in any repo doc, and it bears directly on our search grids.

Their naive baseline searches **0.5 steps in `z` and 1.0° steps in `theta`**. The prompt warns
the dataset builder that a pair landing exactly on that grid is *measurably easier* —
"the matcher gets the pose for free instead of paying interpolation error for it" — and
instructs them to **anti-correlate grid-alignment with severity** so the severity ladder stays
monotone in real difficulty. Integer `z`, whole-degree `theta`, and the mandatory `theta=0.00`
pair are all grid-aligned.

**What this means for us:** `PHASE2_SCALES` is `8.0→12.0 step 0.5` — *exactly the naive
baseline's z-grid*. `PHASE2_ANGLES` is `-5→+5 step 2.5`, coarser than their 1.0°. So on
off-grid poses we do not get the pose for free either; we depend entirely on the golden-section
refinement to buy it back.

Evidence the refinement is doing its job: on their real sample, median scale error ~0.6–0.9%
and theta 10/10. But scale scored only **8.00/10** — and the tier is 1.00 at ≤1%, 0.60 at ≤2%.

**[Resolved 2 Sep — this hypothesis was tested and partly confirmed.]** The prompt's §2.4 requires
the graded set to *reach* the endpoints exactly (z = 8.00 and 12.00). Checking the per-pair scale
against ground truth showed the golden-section refinement **overshooting past those bounds**:
p001 → 7.882 (truth 8.00, 1.48% error) and p013 → 12.174 (truth 12.00, 1.45%), each forfeiting a
whole tier for leaving a range we were told. A clamp of the **reported** `scale` and `theta` to the
disclosed bounds took scale **8.00 → 8.62/10**, with localization (35.60/40) and rejection F1
(0.968) byte-identical — see `route.SCALE_BOUNDS` / `THETA_BOUNDS`.

*The remaining ~1.4 pts is genuine off-grid refinement residual (e.g. p008 11.58 vs 11.90, p019
9.88 vs 10.30) and was NOT pursued — that would require touching the search itself, which can move
x,y. Still the best-motivated future lead, and a genuine line for the failure analysis.*

### 1.3 NEW — every present label is provably hittable (their §5)

Their label-verification gate is strict, and it is described as **non-negotiable**:

- The **global** NCC peak must land within **3 px** of the label.
- Margin over the best competing peak must be **≥ 0.02**, preferring **≥ 0.12**; thin-margin
  pairs must be resampled, not shipped. *"A pair that peaks on its label by 0.002 over the next
  lattice site is a coin flip."*
- Verified on the **re-read PNGs**, not an in-memory render.
- Cross-checked with a **second, deliberately different renderer**.
- Failures are resampled or dropped — *"never ship an unverified label."*

**This is the most reassuring thing in the pack for a correlation-based matcher.** It means
every present pair in the graded set was certified findable by template matching, with margin.
It also removes an excuse: a >5 px miss on a present pair is **our** failure, not an
unhittable label.

**It also invalidates one section of our failure-analysis draft.** Draft §2 argues "a large
fraction of failures are unsolvable by construction." That is true of *our* generator
(`below_floor` pairs, ~8% by design) and explicitly **not** true of theirs. Do not carry that
framing into the PDF as if it applied to the graded set.

### 1.4 NEW — the difficulty lever, and why FOUND_PEAK=0.53 was right (their §5.1)

The prompt tells the builder that the verification gate and the difficulty target pull against
each other, and resolves it with a rule that matters to us:

> treat **reference ambiguity** as the difficulty lever and **severity** as the
> presence-detection lever […] at the top severity the baseline peak collapses while its
> *localisation* stays sub-pixel, so noise moves your presence-detection numbers and barely
> touches centre error.

So in the graded set, heavy degradation is expected to **collapse the peak while localisation
stays good**. That is precisely the failure mode a high `found` threshold punishes: it throws
away a well-localised pair because its peak dropped. Our 1 Sep change `FOUND_PEAK 0.68 → 0.53`
was argued from a cost asymmetry on our own data; **their design document independently
predicts the same thing.** Strong, quotable support for a decision we already made — and worth
one line in the PPT.

### 1.5 NEW — the delivered sample is EASIER than their own design target (their §5.1)

Target: naive-baseline mean credit on present pairs **between 0.30 and 0.55** — *"above that
your set is too easy to separate a field of teams."*

Their baseline scores **0.80** on the 20-pair sample we hold. That is well above their own
band. Two readings, both pointing the same way:

- the 20-pair sample is a gentle validation slice and the blind 200 will be harder, which is
  what their README already warns; or
- the delivered set missed the band.

Either way this is **independent, quantitative support for the handoff's core caution: do not
plan around 89%.** Our 35.60/40 was measured on a set their own rubric calls too easy.

### 1.6 NEW — Set C decoys carry a deliberate signature (their §4)

The prompt anticipates exactly the bug we hit and fixed. It warns that cutting the decoy from
the same zone geometry makes absent pairs score *higher* peaks than present ones — *"A dataset
with that property teaches solvers to reject confident matches — the exact opposite of the
intended lesson."* **That is precisely the bug commit `1b153d4` fixed in our own generator**,
which we had misdiagnosed for days as a fundamental "separability ceiling."

It then requires the decoy to carry **large-scale structure the search canvas does not
contain**, and requires the builder to **audit the signature this leaves** and state how a
solver could exploit it. Constraint: decoys stay in the **same architecture family**.

So their absent pairs have an exploitable large-scale-structure signature by construction. A
richer rejection signal keyed on large-scale structure mismatch would likely beat raw peak NCC.
**Not attempted — one day left, and we are already at F1 0.968.** Recorded as a real lead.

### 1.7 Sample composition — our scorer is correctly aligned

Their §2.4 mandates exactly **8 A / 6 B / 4 C / 2 D** for a 20-pair set.
[scripts/eval_organizer.py:85](../scripts/eval_organizer.py#L85) buckets A = `p001–p008`,
B = `p009–p014`, and excludes `p019/p020` as Set D. **Matches their spec exactly** — our
35.60/40 is computed over the right 14 pairs, and the "15/16 within 5 px" figure counts all 16
present pairs including the two optical ones. Set D is handled without special-casing because
`register.py` does `Image.open(...).convert("L")`.

### 1.8 Where our own generator falls short of the prompt's bar

If the 10-pt generator bucket is judged against this prompt's expectations, we are missing:

- **No label-verification gate.** `generate_dataset.py` has a manifest but no
  peak-within-3px / margin-≥0.02 gate. The prompt calls this non-negotiable and weights it 20/100
  in its own rubric.
- **No contact sheet**, no `REPORT.md`, no resampling-quality (MAE/PSNR/spectral) evidence.

**Caveat on how much this matters:** the mentor said the 10 points are *"carried forward from
phase one… same expectations as before, re-judged only under [Phase 2] conditions"* — i.e.
judged on Phase 1's expectations, not this prompt's 100-point generator rubric. So this is a
modest risk, not a crisis, and there is no time to close it. **The honest move is to name these
as known limitations** — the prompt itself says an empty or cosmetic limitations list scores
zero and *"this section is where honest engineering shows."*

---

## 2. From the mentor session — corrections to our own summary

`PHASE2_UNDERSTANDING.md` captured the session accurately. `FINAL_SESSION_HANDOFF.md` is
slightly lossy in two places:

**The bonus is a TIE-BREAKER, not points.** Handoff §0/§3 frames "+4 if rejection F1 ≥ 0.90 —
we are at 0.968, this is in reach" as points to chase. The mentor was explicit:
*"Does the bonus change the ranking? No, no, it doesn't… it cannot lift a team above 100 for
ranking. It's just the best second tiebreaker after maybe set B credit."* Our F1 already clears
it; nothing to do. Just don't present it in the PPT as if it lifts the score.

**Rejection F1 is over 180 grayscale pairs (A+B+C), and FP/FN weigh equally.** The jury reports
them separately and may use the lean as a tiebreak. Set D is excluded from F1.

Confirmations worth carrying into the deck:

- **Deadline 3 September** — *"28th to 3rd, the 3rd of September is the submission last date."*
- **"Subpixel is allowed and it is given extra weightage."** Said twice. This is the exact axis
  on which classical beat the AMP-trained net (Set A credit 1.000 vs 0.825 — same recall, worse
  precision). Our architecture choice is aligned with the thing they weight highest.
- **Weights must ship in the zip/git**, explicitly, so they can reproduce without guessing
  versions. `best_phase2_speckle.pt` is committed. ✓
- **Classical / learned / hybrid are all equally acceptable** — *"we do not judge you for that."*
  No penalty for shipping classical-led.
- **Follow the format.** Both the mentor and i4C stressed that Phase 1 submissions were hard to
  evaluate: *"for the phase one, it was not clear and so we struggled at… checking the
  evaluation."* Legibility of the repo is worth real goodwill.

---

## 3. THE ONE ACTIONABLE GAP THIS DIGEST FOUND

**`README.md` is still the Phase 1 README, and no handoff document lists it as a deliverable.**

It currently opens with *"Result: 94.5% accuracy @5px … at ~150–430 ms per pair on a 4 GB laptop
GPU"* — Phase 1 numbers, fixed 10× scale, and a **GPU** runtime. The shipped Phase 2 system is
classical, CPU-only, and scores 35.60/40 on their real sample. It never mentions `register.py`.
Section 4.3 is titled *"Why a router instead of just the better model"*, which now contradicts
the shipped configuration (net off by default).

Two reasons this outranks the demo video:

1. **The mentor explicitly asked for one thing in the README** — *"your own confidence… If you
   can mention that in your readme, that'll be really good, because that way we can quickly read
   the readme on how you have found your confidence."* We currently document the `score` column
   nowhere. That is a direct, unanswered request from the evaluator, and it sits next to the
   10-pt calibration bucket.
2. It is the first file a judge opens, and right now it advertises the wrong system.

**Recommend adding it to the deliverables list as item 0** — it is cheap, and it is the only
one where we know exactly what the evaluator asked to see.

---

## 4. Net effect on the plan

Nothing here changes the algorithm, and nothing here is a reason to reopen a settled decision.
What it changes:

- **Three risks closed** (scale convention, theta sign, sample composition) — all confirmed
  correct against their written spec.
- **Two decisions independently vindicated** (`FOUND_PEAK=0.53`; classical-led for sub-pixel
  precision) — now quotable from *their* documents, not just our own measurements.
- **One draft section invalidated** (failure-analysis §2 "unsolvable by construction" does not
  apply to their data).
- **One deliverable added** (README), ranked above the demo video.
- **Two genuine leads logged and deliberately not pursued** (off-grid scale refinement;
  large-scale-structure rejection signal) — both are real, both are out of time, and both belong
  in the failure analysis as "known, diagnosed, not implemented" rather than as silent gaps.

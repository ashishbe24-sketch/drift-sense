# Phase 2 — how much do these findings affect our chances?

**Date:** 2 Sep 2026 · **Author:** DevaanshGupta8 · Honest impact assessment, not a pep talk.

Competition shape (from the deck / handoff): PS-02, a field of teams of which only a
handful advance, so **margin matters**. Score = 100 core (Localization 40 · Pose 20 ·
Rejection 15 · Calibration 10 · Efficiency 5) + 10 generator/citations/failure + an RGB
bonus. I can quantify our score; I **cannot** see the other teams' scores, so "chance of
advancing" is a score band plus judgement, never a hard percentage.

---

## The three things that actually move our chances, ranked

**1. Shipping the deliverables — dominates everything else.**
`failure_analysis.pdf` + the zip + a clean-env fresh-clone verify. A missing or non-running
submission scores **0**, no appeal. This is worth ~100 points of swing on its own; every
technical nuance below is worth 2–6. **Nothing else changes our chances as much as simply
finishing and verifying the submission.** Status: PDF drafted, zip + verify still to do.

**2. The RGB (Set D) bonus — the most likely margin-maker.**
Reachable at **zero matcher change** (0.84–0.97 localization credit through the luma path,
validated end-to-end). With only a few teams advancing, a clean **+bonus could be the
difference** between 6th and 3rd. Status: works today; needs Aryan's sign-off.

**3. The localization-accounting finding (this one) — a real but bounded estimate correction.**
Quantified below.

---

## The localization finding: how much does it actually cost?

`score_phase2.py` credits the internal predicted x,y even for pairs we reject; the real
`register.py` writes 0,0 when `found=0`. If the organizers score the submitted file, our
false-rejected present pairs earn **0 localization**, not ~0.9. On the sev-heavy mixed 200:

| | Localization /40 | Effect on 100-pt total |
|---|---|---|
| What we'd been reporting | 35.4 | — |
| Real submission, worst reading | 31.9 | **−3.5 points (~3.5% of total)** |

**Key framing: this is a correction to our *estimate*, not a new loss.** Our actual
submission was always going to score whatever it scores; we were mis-measuring it. So this
does not lower our real standing — it lowers our *confidence interval's* midpoint, in one of
three possible rubric readings.

**How much it bites depends on two things we don't fully control:**
- **Which rubric reading is real.** Only the most literal reading (rejected present → 0,0 →
  0 credit) costs the full 3.5. Under the other two readings the cost is ~0. So the expected
  hit is **somewhere between 0 and 3.5 points**, not the full 3.5.
- **Where it bites.** Localization is weighted 0.45·A + 0.55·B. Set A (nominal) has almost no
  false-rejects, so the hit lands only on the degraded Set B. On the organizers' actual
  20-sample (mild Set B) the effect is **small** — this is a projection for the harder real
  200-set, whose Set B skews to severity 3–4.
- **Everyone faces the same rubric.** Any team using a rejection threshold and writing 0,0
  for rejects loses the same way. Our localization (13/14 on the real sample, sub-pixel) is a
  *strength* relative to the naive baseline, so even the corrected ~32/40 (~80%) is likely
  still competitive, not a differentiator against us.

**Possible recovery:** lowering `FOUND_PEAK` 0.53 → 0.45 claws back ~2.8 of those points on
our data (see [[FOUND_PEAK_SWEEP_NOTE]]) — but it carries the same overfit risk that sank the
net, so it's Aryan's call and only after a sanity-check on the 20-sample. **Net expected value
of that lever: small and uncertain; not worth destabilizing a validated threshold pre-deadline
unless the deliverables are already locked.**

---

## Bottom line on chances

| Scenario | Core est. (/85) | Advancing? |
|---|---|---|
| Deliverables NOT shipped / verify fails | 0 | **No — this is the only real risk** |
| Deliverables shipped, RGB not claimed, worst loc reading | ~70 | Contender |
| Deliverables shipped, RGB claimed (+bonus), typical reading | ~73 + bonus | **Strong contender** |
| Above + FOUND_PEAK lever pays off | ~75 + bonus | Strong |

- **This localization finding changes our chances by little** — at most ~3.5% of total, likely
  less, and it's a sharpened estimate rather than a lost position. It does **not** threaten
  whether we advance; it refines our expectation from "~86% core" toward a more honest
  "~82–86% core," which was always the truth.
- **What changes our chances a lot is execution:** ship and verify the submission (#1), and
  claim the RGB bonus (#2). Those are worth far more than any threshold tuning.
- **Biggest remaining risk is still not model quality — it's an unshipped or unverified
  submission.** The model is competitive; finish the package.

*(Chances of advancing can't be a number without the other teams' scores. The honest read:
with the deliverables shipped and RGB claimed, we are a strong contender; the one thing that
would sink us is failing to ship or verify.)*

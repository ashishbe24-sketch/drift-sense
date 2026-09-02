# Set D (optical RGB) — luminance shim: status and measurements

**Date:** 2 Sep 2026 · **Author:** DevaanshGupta8 · **Status:** validated, no code change made

Set D is the optical-RGB analogue bonus track (+bonus, ~5 of 15 teams advance, so the
margin can matter). This note records what it takes to capture it and what it measures.

---

## Headline: the "minimal luminance shim" is already the shipped behavior

The proposed shim was: *convert Set D RGB inputs to luminance, then run the existing,
validated grayscale pipeline unchanged.* That is **already what `register.py` does** —
`_load_gray` (register.py:79) loads every input with:

```python
np.asarray(Image.open(path).convert("L"))
```

`convert("L")` applies ITU-R 601 luma (`0.299R + 0.587G + 0.114B`) to any RGB/RGBA
input. So RGB Set D frames are luma-converted and fed to the same classical solver as
Sets A/B/C — **no matcher change, and grayscale behavior is untouched** (an already-`L`
image passes through `convert("L")` unchanged). This is the near-zero-risk path: it
cannot destabilize the core 100 because it adds nothing to the grayscale path.

---

## Measurements (organizers' own generator, `to_optical_rgb`, new seeds)

### Set D standalone (40 optical-RGB present pairs, seed base 5000000)

| Metric | Value |
|---|---|
| Localization | **0.965 credit → 38.60/40**, %@5px **100%**, median 0.82 px |
| Found (rejection) | **40/40 found**, F1(present+) 1.000, 0 false-reject |
| Pose — scale / rotation | 7.72 / 10 · 8.20 / 10 (median 0.87% · 0.196°) |
| Runtime | 2.73 s/pair median (vs ~1.3 grayscale; still 7× under the 20 s cap) |

**Shipped `register.py` end-to-end on the RGB set:** 40 pairs → 40 found, 0 rejected,
valid six-field rows (e.g. `x=711.905, y=653.469, theta=0.1393, scale=8.3197,
found=1, score=0.935`). The entry point handles RGB with no error.

### Set D inside the real-composition mixed 200-set (A70/B70/C40/D20)

| Metric | Set D (n=20, mixed in) |
|---|---|
| Localization credit | **0.840**, %@5px 95.0%, median 1.11 px |

Mixed right in with grayscale A/B/C, Set D still localizes at ~0.84 credit — the optical
blur (blur_px 1.6–2.4) does not destroy the lattice the classical matcher keys on.

---

## Conclusion / recommendation

- **The RGB bonus is reachable at zero matcher change.** The luminance path already
  captures Set D at 0.84–0.97 localization credit, 95–100% within 5 px, with no false
  rejects, validated end-to-end through the shipped `register.py`.
- **No code was changed.** `FOUND_PEAK` and the classical solver are untouched;
  grayscale output is provably unchanged (the shim is a load-time luma conversion that
  already existed).
- **Two open items before this ships:**
  1. **Aryan's sign-off** — capturing Set D reverses his documented decision to skip RGB
     (FINAL_SESSION_HANDOFF §5.3 / §10). The evidence now argues for capturing it; the
     risk is near-zero. His call.
  2. **Optional explicit hardening (minimal, behavior-identical):** make the RGB→luma
     handling explicit and logged in `_load_gray`, and robustly handle RGBA/palette PNGs,
     so Set D support is visibly intentional to a reviewer rather than an implicit side
     effect of `convert("L")`. Grayscale output stays byte-identical (to be proven by a
     before/after diff on the 200-set). **Not done yet — awaiting the go-ahead.**
- **Deliverables still outrank this.** `failure_analysis.pdf` + the zip + the clean-env
  fresh-clone verify come first; Set D is the highest-value cheap add-on after them.

---

## Reproduction

```bash
# 40 Set D optical-RGB pairs (present only; Set D is a present-pair bonus track)
genenv/Scripts/python gen_rubric_eval.py \
    --gen-dir "AMP_Phase 2 material/generator" --out organizer_gen_setD \
    --n-a 0 --n-b 0 --n-c 0 --n-d 40 --seed-base 5000000 --workers 3

# score through the shipped classical pipeline (converts RGB -> L)
.venv/Scripts/python scripts/score_phase2.py organizer_gen_setD \
    --ckpt driftmatch/checkpoints/best_phase2_speckle.pt --out perpairD.csv

# prove the shipped entry point handles RGB end-to-end
.venv/Scripts/python register.py --input organizer_gen_setD/labels.csv \
    --output predD.csv --root organizer_gen_setD
```

(`gen_rubric_eval.py` gained a Set D branch that renders both frames through the
generator's own `to_optical_rgb`. Script + datasets live in the session scratchpad —
regeneratable, not committed.)

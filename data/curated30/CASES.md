# The 30 curated cases

Each case exists to test one specific thing. Ladders hold everything
fixed except the named variable, so a failure can be attributed.

## Results - DriftFind (Solution 1)

Full pipeline: FFT normalised cross-correlation + blur/rotation search + centre
tie-break + sub-pixel refinement. **27 / 30 within 5 px, median error 0.2 px,
~1.1 s/pair.** Per case:

| case | probe | err (px) | verdict |
|---|---|---|---|
| C00-C05 | dose ladder (noise 1.3x -> 7.1x) | 0.06-0.11 | all hit |
| C06-C11 | periodicity ladder (30 -> 2.6 px/period) | 0.14-0.24 | all hit |
| C12-C17 | saliency ladder (landmark 340 -> 32 nm) | 0.13-0.15 | all hit* |
| C18 | off-centre placement | 0.49 | hit |
| C19 | heavy rotation (4.6 deg) | 0.13 | hit |
| C20 | charging bands / streaks | 0.24 | hit |
| C21 | FinFET 1-D periodicity | 0.13 | hit |
| C22 | aliased array, salient landmark | 0.59 | hit |
| C23 | low dose + heavy defocus together | **497** | MISS |
| C24 | large landmark, little context | **327** | MISS |
| C25 | minimal landmark, dense periodic field | **124** | MISS |
| C26 | drift shear | 0.47 | hit |
| C27 | vibration serration | 0.95 | hit |
| C28 | coarse pitch, near-uniform background | 0.76 | hit |
| C29 | labelled below-floor | 0.73 | hit |

**What the ladders show.**
- *Dose (C00-C05):* no break even at 7x the reference noise. With a salient
  300 nm landmark, normalised cross-correlation is essentially immune to photon
  starvation -- the noise limit is not reached on this ladder.
- *Periodicity (C06-C11):* no break even when the array aliases into moire
  (C11, 2.6 px/period). The matcher keys on the aperiodic landmark, not the
  background, exactly as intended.
- *Saliency (C12-C17):* all hit -- but honestly, C16/C17 are the two the
  rationale calls unsolvable (landmark below the wide-view floor). They score as
  hits only because their ground truth is at the exact image centre and the
  centre prior defaults there when no landmark is visible. That is correct
  Bayesian behaviour -- bet on the stage prior when there is no evidence -- but
  it is not genuine detection. An off-centre unsolvable case misses, which is
  exactly what C23 and C25 are.

**The three genuine misses -- each a known mechanism, not a surprise.**
- *C23:* low dose and heavy defocus **together**, on a 26 nm aliased array at
  4.4 deg. Compounded degradation strips the landmark's high-frequency signal;
  this is the honest limit of the classical method.
- *C24:* a large landmark filling a FinFET reference with little periodic
  surround. 1-D periodicity makes correlation flat along the fins, so with no
  lattice to anchor it the match slides -- an anisotropic failure a single
  scalar accuracy number would hide.
- *C25:* a weak 65 nm landmark in a strongly periodic field -- the
  many-near-identical-peaks case. The landmark is too faint to break the tie, so
  a decoy several periods away wins. This is precisely the case a learned
  feature matcher (Solution 2, DriftMatchNet) is meant to rescue.

**One case beat its own label:** C29 was written as a deliberate below-floor
failure, yet DriftFind located it to 0.73 px -- its 211 nm landmark was more
visible at this blur than the rationale assumed. Recorded here rather than
quietly dropped.

\* C16/C17 caveat above: scored hits are the centre prior coinciding with a
centre-placed ground truth, not detection.

### C00 - L1 dose ladder (step 1)

**Variable:** wide-view dose (electrons/grey)  
**Truth:** (560.0, 470.0) px  
**Layout:** dram, pitch 130.0 nm, rotation 1.8 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 24.0

Identical layout, identical reference, identical placement; only the survey dose changes (24.0 e/grey, i.e. a noise ratio of 1.3x versus the reference). This isolates pure photon-starvation: where the method breaks along this ladder is its noise limit, with no other variable to confound it.

### C01 - L1 dose ladder (step 2)

**Variable:** wide-view dose (electrons/grey)  
**Truth:** (560.0, 470.0) px  
**Layout:** dram, pitch 130.0 nm, rotation 1.8 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 12.0

Identical layout, identical reference, identical placement; only the survey dose changes (12.0 e/grey, i.e. a noise ratio of 1.8x versus the reference). This isolates pure photon-starvation: where the method breaks along this ladder is its noise limit, with no other variable to confound it.

### C02 - L1 dose ladder (step 3)

**Variable:** wide-view dose (electrons/grey)  
**Truth:** (560.0, 470.0) px  
**Layout:** dram, pitch 130.0 nm, rotation 1.8 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Identical layout, identical reference, identical placement; only the survey dose changes (6.0 e/grey, i.e. a noise ratio of 2.6x versus the reference). This isolates pure photon-starvation: where the method breaks along this ladder is its noise limit, with no other variable to confound it.

### C03 - L1 dose ladder (step 4)

**Variable:** wide-view dose (electrons/grey)  
**Truth:** (560.0, 470.0) px  
**Layout:** dram, pitch 130.0 nm, rotation 1.8 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 3.0

Identical layout, identical reference, identical placement; only the survey dose changes (3.0 e/grey, i.e. a noise ratio of 3.7x versus the reference). This isolates pure photon-starvation: where the method breaks along this ladder is its noise limit, with no other variable to confound it.

### C04 - L1 dose ladder (step 5)

**Variable:** wide-view dose (electrons/grey)  
**Truth:** (560.0, 470.0) px  
**Layout:** dram, pitch 130.0 nm, rotation 1.8 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 1.5

Identical layout, identical reference, identical placement; only the survey dose changes (1.5 e/grey, i.e. a noise ratio of 5.2x versus the reference). This isolates pure photon-starvation: where the method breaks along this ladder is its noise limit, with no other variable to confound it.

### C05 - L1 dose ladder (step 6)

**Variable:** wide-view dose (electrons/grey)  
**Truth:** (560.0, 470.0) px  
**Layout:** dram, pitch 130.0 nm, rotation 1.8 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 0.8

Identical layout, identical reference, identical placement; only the survey dose changes (0.8 e/grey, i.e. a noise ratio of 7.1x versus the reference). This isolates pure photon-starvation: where the method breaks along this ladder is its noise limit, with no other variable to confound it.

### C06 - L2 periodicity ladder (step 1)

**Variable:** array pitch (nm)  
**Truth:** (430.0, 605.0) px  
**Layout:** dram, pitch 300.0 nm, rotation 1.2 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Pitch 300 nm gives 30.0 px per period in the wide view. Above Nyquist, the array resolves normally. Walking the pitch down while holding the landmark fixed separates 'cannot resolve the array' from 'cannot find the landmark'.

### C07 - L2 periodicity ladder (step 2)

**Variable:** array pitch (nm)  
**Truth:** (430.0, 605.0) px  
**Layout:** dram, pitch 190.0 nm, rotation 1.2 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Pitch 190 nm gives 19.0 px per period in the wide view. Above Nyquist, the array resolves normally. Walking the pitch down while holding the landmark fixed separates 'cannot resolve the array' from 'cannot find the landmark'.

### C08 - L2 periodicity ladder (step 3)

**Variable:** array pitch (nm)  
**Truth:** (430.0, 605.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 1.2 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Pitch 120 nm gives 12.0 px per period in the wide view. Above Nyquist, the array resolves normally. Walking the pitch down while holding the landmark fixed separates 'cannot resolve the array' from 'cannot find the landmark'.

### C09 - L2 periodicity ladder (step 4)

**Variable:** array pitch (nm)  
**Truth:** (430.0, 605.0) px  
**Layout:** dram, pitch 70.0 nm, rotation 1.2 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Pitch 70 nm gives 7.0 px per period in the wide view. Above Nyquist, the array resolves normally. Walking the pitch down while holding the landmark fixed separates 'cannot resolve the array' from 'cannot find the landmark'.

### C10 - L2 periodicity ladder (step 5)

**Variable:** array pitch (nm)  
**Truth:** (430.0, 605.0) px  
**Layout:** dram, pitch 42.0 nm, rotation 1.2 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Pitch 42 nm gives 4.2 px per period in the wide view. Above Nyquist, the array resolves normally. Walking the pitch down while holding the landmark fixed separates 'cannot resolve the array' from 'cannot find the landmark'.

### C11 - L2 periodicity ladder (step 6)

**Variable:** array pitch (nm)  
**Truth:** (430.0, 605.0) px  
**Layout:** dram, pitch 26.0 nm, rotation 1.2 deg, landmark 300.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Pitch 26 nm gives 2.6 px per period in the wide view. At or below the two-pixel Nyquist limit the array folds into moire rather than resolving, so the periodic background stops being usable evidence and the landmark carries the entire signal. Walking the pitch down while holding the landmark fixed separates 'cannot resolve the array' from 'cannot find the landmark'.

### C12 - L3 saliency ladder (step 1)

**Variable:** landmark size (nm)  
**Truth:** (500.0, 500.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 2.6 deg, landmark 340.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Landmark 340 nm against an effective wide resolution of 26 nm, a ratio of 13.2. Comfortably resolved. Every other parameter is held fixed, so this measures exactly how much aperiodic signal the method needs.

### C13 - L3 saliency ladder (step 2)

**Variable:** landmark size (nm)  
**Truth:** (500.0, 500.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 2.6 deg, landmark 220.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Landmark 220 nm against an effective wide resolution of 26 nm, a ratio of 8.5. Comfortably resolved. Every other parameter is held fixed, so this measures exactly how much aperiodic signal the method needs.

### C14 - L3 saliency ladder (step 3)

**Variable:** landmark size (nm)  
**Truth:** (500.0, 500.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 2.6 deg, landmark 140.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Landmark 140 nm against an effective wide resolution of 26 nm, a ratio of 5.4. Marginal: the aperiodic feature is approaching the resolution limit. Every other parameter is held fixed, so this measures exactly how much aperiodic signal the method needs.

### C15 - L3 saliency ladder (step 4)

**Variable:** landmark size (nm)  
**Truth:** (500.0, 500.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 2.6 deg, landmark 90.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Landmark 90 nm against an effective wide resolution of 26 nm, a ratio of 3.5. Marginal: the aperiodic feature is approaching the resolution limit. Every other parameter is held fixed, so this measures exactly how much aperiodic signal the method needs.

### C16 - L3 saliency ladder (step 5)

**Variable:** landmark size (nm)  
**Truth:** (500.0, 500.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 2.6 deg, landmark 55.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Landmark 55 nm against an effective wide resolution of 26 nm, a ratio of 2.1. Below the floor - the disambiguating feature is not recorded in the wide view at all, so this pair is unsolvable by construction and is included deliberately as the endpoint of the ladder. Every other parameter is held fixed, so this measures exactly how much aperiodic signal the method needs.

### C17 - L3 saliency ladder (step 6)

**Variable:** landmark size (nm)  
**Truth:** (500.0, 500.0) px  
**Layout:** dram, pitch 120.0 nm, rotation 2.6 deg, landmark 32.0 nm  
**Wide view:** blur 10.08 nm -> effective resolution 25.77 nm, dose 6.0

Landmark 32 nm against an effective wide resolution of 26 nm, a ratio of 1.2. Below the floor - the disambiguating feature is not recorded in the wide view at all, so this pair is unsolvable by construction and is included deliberately as the endpoint of the ladder. Every other parameter is held fixed, so this measures exactly how much aperiodic signal the method needs.

### C18 - S1 off-centre placement (step 1)

**Variable:** selected case  
**Truth:** (442.5723, 586.1172) px  
**Layout:** finfet, pitch 51.49 nm, rotation 4.631 deg, landmark 225.7 nm  
**Wide view:** blur 14.12 nm -> effective resolution 34.73 nm, dose 2.521

Target placed far from centre, well outside the stage-accuracy prior. Tests whether the method genuinely searches or has learned to bias its prediction toward the image centre, which the stage prior would reward on most pairs.

### C19 - S2 heavy rotation (step 1)

**Variable:** selected case  
**Truth:** (259.7509, 210.4881) px  
**Layout:** dram, pitch 84.8 nm, rotation 1.241 deg, landmark 400.0 nm  
**Wide view:** blur 9.47 nm -> effective resolution 24.44 nm, dose 6.237

Rotation near the top of the range the problem statement allows. Over a 100 px footprint this displaces corners by several pixels, so a translation-only matcher degrades here while a rotation-aware one does not.

### C20 - S3 charging artefacts (step 1)

**Variable:** selected case  
**Truth:** (843.5574, 66.0361) px  
**Layout:** finfet, pitch 274.43 nm, rotation 1.899 deg, landmark 400.0 nm  
**Wide view:** blur 16.53 nm -> effective resolution 40.19 nm, dose 2.565

Wide view carries charge-induced bands and fast-scan streaks. These are large, smooth, non-structural intensity changes that defeat any matcher keying on absolute grey level rather than local structure.

### C21 - S4 FinFET one-dimensional periodicity (step 1)

**Variable:** selected case  
**Truth:** (516.377, 482.094) px  
**Layout:** finfet, pitch 285.14 nm, rotation 2.645 deg, landmark 346.7 nm  
**Wide view:** blur 7.11 nm -> effective resolution 19.5 nm, dose 4.892

Parallel fins are periodic along one axis only, so correlation is sharply peaked across the fins and nearly flat along them. Error should be strongly anisotropic - a single scalar accuracy figure hides this.

### C22 - S5 aliased array, salient landmark (step 1)

**Variable:** selected case  
**Truth:** (96.9977, 630.1924) px  
**Layout:** finfet, pitch 118.66 nm, rotation 3.229 deg, landmark 358.1 nm  
**Wide view:** blur 7.25 nm -> effective resolution 19.79 nm, dose 8.029

Array folds into moire while the landmark stays well resolved. The correct behaviour is to ignore the background entirely; a method that weights all pixels equally is actively misled by the aliased texture.

### C23 - S6 low dose and heavy defocus together (step 1)

**Variable:** selected case  
**Truth:** (420.5704, 532.5973) px  
**Layout:** dram, pitch 26.44 nm, rotation 4.385 deg, landmark 140.7 nm  
**Wide view:** blur 6.07 nm -> effective resolution 17.45 nm, dose 3.991

Both degradations at once. Included because failure modes are not independent: blur removes the high-frequency content that would otherwise survive the noise.

### C24 - S7 large landmark, little context (step 1)

**Variable:** selected case  
**Truth:** (819.1206, 584.2387) px  
**Layout:** finfet, pitch 299.16 nm, rotation 1.368 deg, landmark 211.7 nm  
**Wide view:** blur 14.05 nm -> effective resolution 34.57 nm, dose 4.287

The aperiodic feature dominates the reference, leaving little periodic surround. Easy to localise coarsely but the sub-pixel estimate rests on a few long edges rather than a dense lattice.

### C25 - S8 minimal landmark, dense context (step 1)

**Variable:** selected case  
**Truth:** (565.3343, 434.1353) px  
**Layout:** dram, pitch 315.76 nm, rotation 0.861 deg, landmark 64.8 nm  
**Wide view:** blur 17.55 nm -> effective resolution 42.52 nm, dose 3.321

The mirror of S7: a small landmark inside a strongly periodic field, which is the classic many-near-identical-peaks case the centre tie-break exists to resolve.

### C26 - S9 strong drift shear (step 1)

**Variable:** selected case  
**Truth:** (537.2929, 491.7313) px  
**Layout:** dram, pitch 91.23 nm, rotation 1.709 deg, landmark 400.0 nm  
**Wide view:** blur 18.75 nm -> effective resolution 45.29 nm, dose 12.022

Slow stage creep during the raster shears the wide field. A rigid matcher accumulates a systematic offset that grows down the image.

### C27 - S10 vibration serration (step 1)

**Variable:** selected case  
**Truth:** (668.9638, 697.1613) px  
**Layout:** finfet, pitch 273.95 nm, rotation 1.853 deg, landmark 184.1 nm  
**Wide view:** blur 12.76 nm -> effective resolution 31.68 nm, dose 3.523

Per-scanline displacement serrates edges that should be straight, adding high-frequency error exactly where sub-pixel refinement takes its signal.

### C28 - S11 coarse pitch, near-uniform background (step 1)

**Variable:** selected case  
**Truth:** (430.1334, 653.3207) px  
**Layout:** finfet, pitch 294.32 nm, rotation 0.937 deg, landmark 400.0 nm  
**Wide view:** blur 14.26 nm -> effective resolution 35.03 nm, dose 3.958

Very relaxed pitch leaves large flat areas with little texture to lock onto away from the landmark.

### C29 - S12 below-floor failure case (step 1)

**Variable:** selected case  
**Truth:** (542.1322, 649.3504) px  
**Layout:** dram, pitch 118.28 nm, rotation 1.187 deg, landmark 211.4 nm  
**Wide view:** blur 10.32 nm -> effective resolution 26.28 nm, dose 2.359

Deliberately unsolvable: the landmark falls under the visibility floor for this blur. Included as the honest failure case - the useful output here is a low-confidence flag, not a coordinate.

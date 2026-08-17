# Webinar Corrections to the Generator Spec
### Source: Applied Materials PS-02 webinar, 1 Aug 2026, 1:20:32.
### Speaker: **Aayush Raina**, Deputy Director, Algorithm Group, eBeam Metrology, Applied Materials — 14 years at AMAT, and **the subject-matter expert, mentor and evaluator for this problem statement**.

Full transcript (first 52 min) in `webinar_transcript.md`. This file records what the webinar **changes** in [GENERATOR_SPEC.md](GENERATOR_SPEC.md), with quotes.

---

## A. CORRECTIONS — things my spec had wrong

### A.1 Rotation was 10–20× too small ⚠ biggest correction

My spec derived **0–5 mrad (0–0.3°)** from stage specs and wafer-aligner tolerance. The evaluator says:

> "you can have only 1 to 3°. So when you're creating synthetic data set maybe this line when you're drawing you can move either one of them by 3° 2° 5°."

**1–3°, occasionally 5° = 17–87 mrad.** My physics reasoning was about *residual stage rotation on a re-visit*; he is describing the augmentation they actually want in the data. **The sponsor's number governs the dataset.**

→ **Change:** rotation sweep **0–5°**, typical 1–3°. Keep the mrad derivation in the deck as the *physical* argument for why small rotations dominate, but generate to his range. At 3° the 100 px footprint corner moves ~5 px — this is no longer a sub-pixel effect, it materially changes the matching problem.

### A.2 Tolerance is 1–5 **pixels**, not sub-pixel

The deck said "within subpixel of the true downsampled location." The evaluator repeatedly says otherwise:

> "you can have uh 1 to 5 pixel of error. You can even have bigger error depending upon how much noise you're adding. But you should be able to justify it."
> "slight errors like 1 to 5 pixel is a good good model"

→ **Change:** the sub-pixel refinement stage drops in priority. **Disambiguation and robustness matter more than the last 0.3 px.** Our "report a tolerance curve" plan was right — see A.3.

### A.3 Evaluation is a PR curve over pixel thresholds, using **their** scoring utility

> "we'll give a utility to basically produce such matrices… given a two pair of images and your algorithm it will create a confusion metrics at different pixels and that's how the PR study comes"
> "you will give a list of pair of images which has CSV of wide image search path, reference image path, ground truth on the wide search image and this is your input to the scoring utility. it will publish the metric … will give you some plots and that plot is something you can add to your submission so that everybody is presenting in the same format"
> "You'll get the scoring utility code also."

The threshold in the PR study **is the pixel error** — exactly analogous to a classifier score threshold.

→ **Change:** our `labels.csv` must match their CSV contract: **`wide_path, ref_path, gt_x, gt_y`**. Keep our richer parameter columns, but those four must be present and named plausibly. Plan to swap our own scorer for theirs the moment it lands.

### A.4 Charging and defocus move **into v1** (I had deferred charging)

> "then there is a charging — charging comes because your edges will become uh you'll have some different noise around the edges"
> "you should assume that the wide search image you're taking it from a zoom out, it will not be very clear… it is more noisy because it doesn't know where to focus. So it may focus something on the immediate… General tendency of cameras will be to focus on something nearest block."

→ **Change:** charging is an **explicitly requested** augmentation, not optional. And the wide view gets a **defocus** term, not only a dose term. My dwell-time/dose derivation stays (it's more rigorous and still true), but **focus error is the mechanism the sponsor named** — model both.

He also ties this directly to ranking:

> "someone who is like assuming this kind of assumptions and creating data they will have better real life scenarios and with proper justification they will be much more appreciated or ranked."

### A.5 The scoring split is confirmed — and the missing 10% is resolved

The deck's FAQ summed to 90%. The evaluator gives the full breakdown:

| Weight | Criterion |
|---|---|
| **50%** | Inference results — confusion matrix / PR curve over pixel error, across noise levels, on **our data and theirs** |
| **30%** | Augmentation — *"if you read more and learn more everybody can use and have a clear shot at 30%"* |
| **10%** | **Explainability** of the cases and the approach |
| **10%** | **Failure cases** |
| +10 bonus | RGB / optical-tool images |

Note he splits explainability and failure cases into **two separate 10% buckets**. My spec treated them as one.

---

## B. NEW REQUIREMENTS I had not scoped

### B.1 The 30 samples must be **curated and argued**, not sampled

> "you give us top 30 samples which you feel like where algorithm was very uniquely tested… it is very important that you justify your top 30 sites with your augmentation and your results on it. **We don't care that the results are average or bad.** But if you've generated very good data it is also a very good property… a very much needed skill for somebody who is working in algorithm domain to create a data set which is very challenging"

→ This is a **writing deliverable** attached to each of 30 pairs: *why is this case hard, what does it test.* Our eval set needs a `rationale` column and a companion document. Previously I planned 200 random-seed pairs — that's still right for measuring, but **30 of them must be hand-picked and defended.**

### B.2 The same reference may be reused across different wide images

> "you can reuse the same image on a different pattern… You can have same reference, change the right side. You try to find the same pattern on some other different layout. So this freedom you have."

→ Lets us build **controlled difficulty ladders**: one fixed reference, N wide images with monotonically increasing noise / periodicity / rotation. That is a far more persuasive 30-case set than 30 unrelated pairs, and it isolates one variable at a time.

### B.3 Function-name contract — they will dictate the signature

> "you give a Python file with a given function name. We will throw the function name and uh which will give the xy coordinates in pixels on a wide search image for a given reference image."

→ Our inference entry point must be trivially adaptable. Keep the core as a pure function and let the CLI wrap it, so renaming to their signature is a one-line change.

### B.4 Assert the image size

> "whatever images you are creating you have a validation block that it should have 1,000 pixel by 1,000 pixel… to verify that your algo works on 1000 you can have an assert also."

→ Explicit `assert img.shape == (1000, 1000)` in both generator and inference. He raised it because **their automated test modules will break otherwise.**

### B.5 Explicit ban on generative-image models — our approach is exactly what he asks for

> "**Don't give Nano Banana or Gemini or OpenAI to create images for you.** Ask it to read the literature, give you the Python code with all these things and then generate the files — so that you will know exactly what logic was used to create these images and also you know what is the ground truth at pixel level."

→ Vindicates option (b) completely. Ground truth must come from **placement, not from an image model.** Worth one explicit line in the deck saying no generative image model touched the data.

### B.6 Model size and inference time are graded

> "model size… it should not be so big… inference time it should not take several minutes."
> "the notebook will help us understand how heavy your model is, how many parameters it has, what is the architecture and how much training time does it need and what data you have used for training."

→ Reinforces the small, patch-based model — and your 4 GB GPU stops being a constraint and becomes a design fit.

### B.7 A GPU batch variant may be submitted **as a bonus**, separately

> "computation time… it should be measured per single image, we keep it simple, not GPU CPU… but if you want to have a GPU based algorithm you want to split it, you can mention your approach, have a separate submission for that… you're doing with 10 pairs and you want to use a GPU and you can bring it down. That is also a very good solution. We will respect the computation time in that."

→ Primary = single-pair CPU timing. Optional extra = batched GPU throughput version.

---

## C. CONFIRMATIONS — spec decisions the webinar validates

| Our decision | His words |
|---|---|
| Manhattan geometry (axis-aligned rects + circles) | *"the patterns are Manhattan pattern — circles, squares, polygons… everything is crisscross 90° at a predefined angle. It is not like non-linear pattern"* |
| Deliberately generate repeated/ambiguous regions | *"when we will also evaluate we will also give repeated areas in the image. So that we see that how your algorithm works"* |
| Centre tie-break is a real rule | *"If two of them have same score give me the one which is closest to the center"* — with a worked example at 1:14:46 |
| Downsample the reference, don't upsample the search | *"you want to create a image and then a down sampled image, or you create a big image find a pattern and upsample it. So up to you but **I'll suggest the first way**"* |
| Independent noise on the two captures | *"add more grayscale noise on each image. Don't use same noise on both — have different noises"* |
| Edge brightening | *"mimic it to how SEM images should have brighter contrast around the edges"* |
| Classical methods are fully acceptable | *"It's a pure image processing problem… You can solve it with pure mathematics. You can add some complexity by adding classical methods. You can add deep learning networks also."* |
| Distortion as an augmentation | *"Google distortion in images — it will show some polygon edges are distorted. It's like old televisions… shapes getting misarranged"* |
| Failure-case analysis with root cause | *"justify one good failure case… maybe you added too much noise, too much distortion, too much scaling — where your elbow starts breaking. If you have understanding about what your algo cannot do, that is also one of the key strengths"* |
| Python only, no notebook for inference | *"pure Python code not a Jupyter notebook, because the computation time also matters"* |

**New parameter he named:** per-polygon **scaling ±20%** — *"I can reduce the scale of the circle by 20%, increase it by 20%."*

---

## D. Honest assessment of our spec against his expectations

He expects most teams to prompt an LLM to write a generator from a few papers. **Our spec is considerably deeper than the bar** — measured edge profiles from real SEM, an over-dispersed Poisson model with a cited Fano factor, a sampled LER PSD, IRDS-traceable dimensions.

That depth is not wasted — 30% is explicitly for literature-justified augmentation and he says justified assumptions rank higher. But the rebalance is real:

- **Under-weighted by me, explicitly wanted by him:** rotation (1–3°), charging, defocus on the wide view, per-polygon scaling ±20%, geometric distortion.
- **Deeper than required, keep but don't lead with:** Fano 1.9, LER PSD model, noise colour vs frame count. These are differentiators in the write-up, not prerequisites.
- **Newly critical:** the 30 curated-and-argued cases, and matching their CSV/scoring-utility contract.

---

## E. Watch list

1. **Scoring utility + validation datasets + GitHub repo of prompts** — promised, not yet released.
2. **Another Q&A session** — *"we'll be taking one more Q&A session in next week."*
3. **Updated PPT** on the landing page — *"the updated PPT will again be made available."*
4. **Prize discrepancy:** video description says **₹4,00,000**; the i4c site says ₹5,00,000.
5. Registration is as **team captain**, team of 2–4, Phase-1 deadline **16 Aug 2026**.

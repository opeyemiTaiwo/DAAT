# DAAT — Density-Adaptive Auditable Tracking

*Density-Adaptive Association with Auditable Identity Tracking for Forensic
Multi-Object Tracking*.

DAAT is a tracking-by-detection pipeline (YOLOX detections, BYTE-style
two-stage association) with two additions aimed at forensic deployment.

**1. A frozen, train-only protocol for per-sequence association.**
A rule fit on the seven MOT17 training scenes maps detector-measured scene
density to the track threshold. It is frozen before any test run and
consults no test ground truth and no leaderboard feedback
(`rules/perseq_rule.json`, `rules/test_density.json`).

*We present this as a protocol contribution, not an accuracy result.* Under
leave-one-scene-out cross-validation the rule does **not** outperform a
well-chosen constant threshold — see `results/` and the validation notebook.

**2. Auditable identity tracking.** Every identity-relevant decision —
revivals, re-associations and *refusals* — is logged with its basis: IoU,
appearance similarities, gates and margins, plus the complete
track x detection affinity matrix at event frames. Format:
[`docs/AUDIT_LOG_FORMAT.md`](docs/AUDIT_LOG_FORMAT.md).

## Results

### MOT17 test server (private detections)

The paired A/B below differs **only** in the threshold policy:

| Entry | HOTA | MOTA | IDF1 | IDs |
|---|---|---|---|---|
| density rule (Submission A) | **60.6** | **77.9** | **75.2** | **2,355** |
| fixed tau=0.5 (Submission B) | 59.8 | 77.3 | 73.3 | 2,865 |

The rule contributes +0.85 HOTA, +1.91 IDF1 and -18% identity switches
**over the development default tau=0.5**. The submitted system adds two
further association refinements and scores 60.8 / 77.8 / 75.9.

**A protocol disclosure about that last sentence.** Those two refinements
were chosen with test-server feedback already visible. The 0.2 HOTA between
60.6 and 60.8 is therefore not a clean frozen-then-tested result, and it is a
departure from the Calibrate-Freeze-Audit discipline this work advocates.
Submission A at 60.6 is the figure that was frozen before any test run.

### The accuracy claim does not survive cross-validation

tau=0.5 is not the strongest constant available. The per-scene optima are
0.50, 0.55 and 0.60, so the training-optimal global constant is tau=0.60.
Refitting the whole procedure leaving out one scene at a time:

| Threshold policy (LOSO, n=7) | HOTA | vs tau=0.6 | Wins |
|---|---|---|---|
| oracle per-scene (ceiling) | 68.05 | +0.57 | — |
| **fixed tau=0.60 (best global)** | **67.47** | 0.00 | — |
| density rule (as described) | 66.80 | -0.67 | 0/7 |
| density rule (as implemented) | 66.34 | -1.13 | 2/7 |
| fixed tau=0.50 (dev default) | 65.92 | -1.55 | — |

A per-scene oracle gains only +0.57 HOTA, bounding any rule of this form on
MOT17. No difference is significant (Wilcoxon, p >= 0.31). We report this
negative result rather than the favourable tau=0.5 comparison alone.

### Ambiguity-flag calibration

Matching refusal events against true GT identity switches, under a spatial
gate: a refusal counts as flagging a switch only when it concerns the same
object, at IoU 0.3, within the stated window. Over the seven MOT17 training
scenes, 1,206 switches and 1,072 flags:

| Window (frames) | Recall | Silent | Precision | Chance precision | Lift |
|---|---|---|---|---|---|
| 5 | 0.333 | 66.7% | 0.259 | 0.095 | 2.7x |
| 10 | 0.458 | 54.2% | 0.303 | 0.136 | 2.2x |
| 30 | 0.684 | **31.6%** | 0.402 | 0.233 | 1.7x |

"Chance precision" is what randomly placed flags firing at the same rate
achieve; "lift" is the ratio to it. Reporting it was a reviewer's request and
it changed the result.

**An earlier version of this section was wrong and is corrected here.** It
reported recall 0.896 and a 10.4% silent-error rate over 701 switches. Three
things were wrong with that. The switch count came from a third, inconsistent
counting; independent CLEAR matching finds 1,206. The matching criterion
required only that a refusal fall within the association window, with no
requirement that it concern the same object, and under that criterion
randomly placed flags score *as well as or better than* the real flag, so the
figure measured flag density rather than informativeness. And switches
created by the refusal itself were counted in the numerator, which is
circular.

The corrected reading: the flag carries real information, beating chance by
1.4 to 2.7 times, but it is a weak triage aid. Roughly a third of true
switches raise no flag at a 30-frame window, and two thirds at 5 frames.

The DanceTrack half of this table behaves the same way once measured on the
same criterion, which withdraws a second claim: we had reported the flag as
near-complete on pedestrian footage and much weaker under costume, and that
contrast was an artifact of measuring the two domains differently. See the
paper's Table 13.

Reproduce with `reproduction/scripts/flag_calibration.py` (needs MOT17 GT) or
`reproduction/reproduce.py` for the DanceTrack half, which needs nothing you
do not already have here.

### Differential error

Identity-switch rate is **not** uniform across subgroups. Recomputed for the
revision, the visibility ratio is **4.4x** and the height ratio **3.8x**.

The figures previously given here (2.7x and 1.3x) are withdrawn, and so is
the conclusion drawn from them. We wrote that occlusion rather than apparent
scale drives identity error. The height column is **not monotonic** across
buckets (11.67, 3.54, 1.72, 3.06 per 1,000), so neither variable orders the
error rate on its own, and reporting two marginals concealed that. The paper
now reports the visibility panel with counts and bootstrap intervals plus the
full 4x4 joint grid, which spans 0.40 to 14.46 per 1,000 — a 36x range that
both marginals understate. Visibility and height correlate at only +0.168, so
they are separable, but they interact. See the paper's Tables 4 and 7.

MOT17 carries no demographic annotation, so demographic differential error is
unmeasurable on this benchmark and none should be inferred from these
results.

## A note on sample size

MOT17 ships seven training scenes, each with three public-detector labels.
Under the **private** protocol we supply our own detections, so the three
variants of a scene are byte-identical runs. The effective sample size is
**n = 7**, not 21. The minimum detectable difference at that size is
approximately **1.73 HOTA**; smaller differences in the ablation tables do
not support a component ranking.

## Repository layout

```
code/         DAAT_MOT17_test_submission.ipynb — full Phase-2 pipeline:
              detector -> tracker (with audit hooks) -> MOT-format output ->
              submission packaging. Set DATA_ROOT / OUTPUT_ROOT at the top.
              DAAT_paper_analysis.ipynb — aggregates experiment metrics,
              renders figures, re-scores raw outputs (TrackEval).
              DAAT_validation_analyses.ipynb — reproduces the validation
              results above: effective sample size, LOSO cross-validation,
              paired significance tests, audit-log schema audit, the
              deployment-observable event taxonomy, flag calibration and
              differential error.
rules/        perseq_rule.json    — the frozen density->threshold rule
              train_density.json  — measured train-scene densities
              test_density.json   — measured test-sequence densities
docs/         AUDIT_LOG_FORMAT.md — schema of the evidence logs
sample_logs/  Complete, unmodified audit logs for MOT17-12-DPM and
              MOT17-14-FRCNN from the official test submission.
results/      One folder per experiment, each with its
              ALL_SEQUENCES_metrics.csv exactly as produced at run time.
verify.py     Re-derivation audit over the evidence logs (see below)
eval/         trackeval_rescore.py — TrackEval rescoring
              flag_calibration.py  — ambiguity-flag calibration vs. GT
reproduction/ Self-contained package that rebuilds the paper's numbers and
              reports whether each matched. Event logs, DanceTrack ground
              truth, tracker output, the analysis scripts, the log-format
              specification, and an independent second verifier.
```

## Reproducing

1. Obtain MOT17 from motchallenge.net and set `DATA_ROOT` to its parent.
2. Detector weights: `bytetrack_x_mot17.pth.tar` from the public ByteTrack
   model zoo (the notebook downloads them).
3. Run `code/DAAT_MOT17_test_submission.ipynb` top to bottom. Phase 2
   measures each test sequence's density from raw detector output, looks up
   its bucket in `rules/perseq_rule.json`, tracks, and writes the flat
   21-file submission zip plus per-sequence evidence logs.
4. Validation results: run `code/DAAT_validation_analyses.ipynb`. Sections
   1-8 need only `results/`; the flag-calibration and differential-error
   sections additionally need MOT17 train ground truth (`GT_ROOT`).

## Known limitations of this release

- **Affinity matrices are keyed by frame index only** (`frame_NNNNN_iou.csv`)
  and one directory serves all sequence-detector pairs, so a matrix cannot
  be attributed to a sequence from its filename alone. Provenance is
  recoverable by matching the logged `track_box` values against the emitted
  output. Namespacing by sequence is the obvious fix.
- **The pre-assignment gate mask is not recorded**, so a verifier cannot
  reproduce which entries were suppressed before the solve.
- Matrices are retained at logged event frames rather than every frame.

## Requirements

Python >= 3.10, numpy, scipy, pandas, opencv-python, torch/torchvision (for
YOLOX), TrackEval
(`pip install git+https://github.com/JonathonLuiten/TrackEval.git`).
Developed and run on Google Colab (single GPU).

## Data availability and reproduction

`reproduction/` is a self-contained package that rebuilds the paper's numbers
from the artifacts and reports whether each matched. It needs no network
access and no dataset download.

```
cd reproduction
pip install -r requirements.txt
python3 reproduce.py --smoke     # ~30 seconds, checks the environment
python3 reproduce.py             # the full run, a few minutes
```

It ends with a verdict line such as `26 of 26 claims reproduced`. Every number
it checks is written down in `reproduce.py` alongside the value the paper
prints, so nobody has to read the paper to judge whether a run succeeded.

It contains 467 per-frame audit event logs (35 from the frozen DanceTrack test
submission, 7 from the frozen MOT17 test submission, 425 from 17 DanceTrack
validation runs), 111 tracker output files, ground truth for 15 DanceTrack
validation sequences, and the analysis scripts.

Five claim groups cannot be checked from it, and it says so with the reason
rather than skipping them silently: four need MOT17, which is CC BY-NC-SA 3.0
and not redistributed here, and one needs the serialized cost matrices, which
are not included. Pass `--gt` to the relevant scripts with your own MOT17
copy for those.

`reproduction/SPEC.md` is the log format and decision rules written out in
full, sufficient to write a verifier without reading ours.
`reproduction/verify_independent.py` is a second verifier built from that
specification alone, by an implementer with no access to `verify.py`, and
`reproduction/REPORT.md` is what they found — including three defects our own
verifier passed. `reproduction/CREDITS.md` lists every third-party component
and its license.

## Re-deriving logged decisions

`verify.py` is the re-derivation tool. It reads an evidence log and, for each
logged identity decision, checks two things this work keeps distinct:

1. **Decision re-derivation.** Given the inputs the log records, does the
   recorded outcome follow from the tracker's deterministic rule?
2. **Input provenance.** Can each logged input value itself be recomputed from
   the raw detections and embeddings held in this release?

A decision is reported as FULLY re-derived when both hold, and as DECISION-ONLY
re-derived when the rule reproduces but one input (an appearance term inside a
fused cost) is not independently recomputable, because per-detection embeddings
are not serialized.

```
python verify.py sample_logs/
python verify.py --legacy sample_logs/   # the version submitted with the paper
```

**What changed, and why `--legacy` exists.** The version submitted with the
manuscript did not re-derive the three threshold decisions. For
`byte_second_assoc`, `unconfirmed_update` and `re_activate_from_lost` — 9,612
of the 12,013 DanceTrack association decisions — it tested only that the
logged cost lay in [0, 1], which asks whether a number is a valid number, not
whether the outcome follows from it. Two reviewers observed that
re-derivability was self-certified, and on those categories they were right
in a stronger sense than either described.

This version applies the actual rule. Every record passes, with the maxima
sitting just inside the thresholds, so the conclusion is unchanged and better
supported. It also stops counting a record whose operand is absent as a
passing record. `--legacy` reproduces the submitted behavior so the two can
be compared on the same logs.

**A warning about `track_box`.** The field is written after the track has
absorbed the detection, not at the predicted position the association used.
IoU recomputed from `det_box` and `track_box` therefore does not reproduce
the logged `iou_overlap`: it matches on 36 of the 10,056 records of the four
IoU-driven reasons, and reads higher almost always. On the three
appearance-driven reasons, where no absorption occurs, it matches on 83 of
84. The logged `iou_cost` is the operand the decision used; the boxes beside
it are not a way to check it. Serializing the predicted box is the fix.

The log schema is documented in [`docs/AUDIT_LOG_FORMAT.md`](docs/AUDIT_LOG_FORMAT.md).

## Authors

Opeyemi T. Adeniran, Peter Taiwo, Hashmath Fathima, Temitope Ajibola,
Derrick Cook, Emmanuel Masa-Ibi, Kelechi Nwachukwu, Kofi Nyarko

Center for Equitable Artificial Intelligence and Machine Learning Systems
(CEAMLS), Morgan State University, Baltimore, MD 21251, USA

Correspondence: opade7@morgan.edu

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The audit logs, frozen rules, metric records, and documentation are released
under CC BY 4.0.

## Corrections in this revision

This release corrects figures and claims that did not survive reconciliation
against the run artifacts. Each is stated where it occurs above; collected
here for anyone comparing against an earlier copy of this repository.

| Was | Now |
|---|---|
| Flag recall 0.896, 10.4% silent, 701 switches | Recall 0.684, 31.6% silent, 1,206 switches, under a spatial gate and against a chance baseline |
| The flag is near-complete on pedestrians, weak under costume | Withdrawn; the contrast was two matching criteria, not two domains |
| Differential error 2.7x visibility, 1.3x height | 4.4x and 3.8x; the height column is not monotonic |
| Occlusion rather than scale drives identity error | Withdrawn; neither variable orders the rate alone |
| 60.8 HOTA presented alongside the frozen protocol | Disclosed as chosen with test feedback visible |
| 4 authors | 8, matching the manuscript |
| `verify.py` re-derives logged decisions | It range-checked three categories covering 9,612 of 12,013 decisions; now re-derives them, with `--legacy` for the old behavior |
| `track_box` is the predicted box | It is written post-update and does not reproduce the logged IoU |

Two code defects are fixed in `code/DAAT_MOT17_test_submission.ipynb`.
`run_tracking` read a bare `seq_name` global that is assigned only inside a
later function, so evidence logs could be written under whichever sequence
ran last. And the re-identification embedder fell back silently to
ImageNet-pretrained weights when the MSMT17 download failed, changing the
frozen configuration without recording it; it now refuses unless
`ALLOW_IMAGENET_REID_FALLBACK` is set explicitly, and records which weights
were used.

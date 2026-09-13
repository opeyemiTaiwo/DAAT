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

Matching refusal events against true GT identity switches over the seven
training scenes:

| | value |
|---|---|
| precision | 0.673 (726 of 1,078 refusals near a true switch) |
| recall | 0.896 (628 of 701 switches raise >= 1 flag) |
| F1 | 0.769 |
| **silent errors** | **73 of 701 (10.4%) raise no flag** |

The flag is a triage aid, not a completeness guarantee: roughly one flagged
link in three is a false alarm, and one switch in ten escapes entirely.

### Differential error

Identity-switch rate is **not** uniform across subgroups:

| GT visibility | rate | | box height | rate |
|---|---|---|---|---|
| <0.3 | 0.514 | | small | 0.357 |
| 0.3-0.6 | 0.453 | | med-small | 0.400 |
| 0.6-0.8 | 0.320 | | med-large | 0.383 |
| >0.8 | 0.188 | | large | 0.312 |
| **spread** | **2.7x** | | | 1.3x |

Occlusion, not apparent scale, drives identity error. MOT17 carries no
demographic annotation, so demographic differential error is unmeasurable
on this benchmark.

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
eval/         trackeval_rescore.py — TrackEval rescoring
              flag_calibration.py  — ambiguity-flag calibration vs. GT
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

## License and anonymity

Released for anonymous review. A public repository with authorship and a
permanent license will replace this on acceptance.

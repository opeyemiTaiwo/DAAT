# Reproduction package

**Auditable Multi-Person Tracking for Forensic Video**
MDPI *Information*, manuscript `information-4636481`

This bundle rebuilds the numbers the paper prints, from the artifacts the paper
describes, and tells you whether they matched. You should not need to read the
paper to use it.

---

## If someone asked you to verify this paper

Three commands.

```bash
pip install -r requirements.txt
python3 reproduce.py --smoke      # ~30 seconds, checks the environment
python3 reproduce.py              # the full run, a few minutes
```

The last line is the verdict, and it looks like this:

```
26 of 26 claims reproduced
```

Send that line to whoever asked. If any claim disagrees, the run prints what
the paper says, what it computed, and which script produced it, so the
disagreement is reportable without further digging.

You do not need to judge whether a number looks right. Every figure the paper
prints is written down in `reproduce.py` and compared automatically.

### What you are attesting to

That the released code, run on the released artifacts, produces the numbers
printed in the paper. That is all, and it is deliberately narrow. You are not
attesting that the tracker is accurate, that the logs are complete, or that
the conclusions follow. Reviewer 4 asked for the verifier to be "independently
implemented, or at least run by someone outside the author team," and this is
the second of those.

---

## What is in here

```
reproduce.py              the runner; every claim and its expected value
scripts/                  the analysis scripts, one per claim group
data/logs/dancetrack/     35 event logs, frozen DanceTrack test submission
data/logs/mot17/           7 event logs, frozen MOT17 test submission
data/logs/dancetrack_val/ 425 event logs, 17 DanceTrack validation runs
data/dancetrack_gt/        15 DanceTrack validation ground-truth sequences
data/dancetrack_tracks/   111 tracker output files across validation runs
out/                      written by the run; reports and a JSON summary
CREDITS.md                every third-party source and its license
```

Roughly 90 MB. No network access is needed at run time.

---

## What this bundle cannot check, and why

A reproduction package that quietly skips things is worse than one that
refuses. `reproduce.py` prints these with its results and counts them
separately, so a partial bundle can never look like a clean pass.

| Claim | Why not |
|---|---|
| Table 2, matrix coverage 23.0% | The 1,172 serialized cost matrices of the frozen submission are not included |
| Table 13, the MOT17 half | Needs MOT17 training-split ground truth and the MOT17 tracker output |
| Tables 4 and 7, differential error | Needs MOT17 ground truth, which carries the visibility and height annotation |
| Tables 8 and 12, accuracy comparisons | Needs MOT17 ground truth and TrackEval |
| Figure 2 | Needs TrackEval; install it and run `scripts/make_fig_dancequal.py` |

MOT17 is licensed CC BY-NC-SA 3.0, whose ShareAlike term would propagate to
anything bundled with it, so it is not included here. Download it from
MOTChallenge and pass `--gt` to the scripts above if you want those rows too.

### One limitation to know before you run

The DanceTrack calibration runs on a subset. Validation ground truth is held
for 15 of the 25 annotated sequences, and the surviving tracker output covers
3 to 5 sequences per run across 8 runs. The paper states this; the package
reports the counts rather than implying the full split.

---

## For the reader rebuilding the whole paper

Each script is runnable on its own and documents what it computes in its
module docstring. `SPEC.md` is the log format and the decision rules written
out in full, which is what `verify_independent.py` was built from by an
implementer with no access to `verify.py`. If you want to write a third
verifier, start there and do not read either of ours.

`verify.py --legacy` reproduces the verifier submitted with the original
manuscript, which range-checked three decision categories rather than
re-deriving them. The difference between the two is discussed in Section 4.6.

---

## Citing

If you use the logs or the scripts, cite the paper. If you use the DanceTrack
ground truth included here, cite DanceTrack as well; `CREDITS.md` has the
reference and the license for every third-party component.

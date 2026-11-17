#!/usr/bin/env python3
"""
transfer_equivalence.py -- equivalence testing for the cross-domain transfer
claim, replacing an absence-of-evidence statement with a stated bound.

Reviewer 2, comment 3: "Table 9: +0.11/-0.27 and -0.54/-1.01 HOTA, with n=7
and n=4 and a minimum detectable difference of ~1.7. 'Transfers without
regression' and 'embedders do not transfer in either direction' are
absence-of-evidence claims. (Use CIs or equivalence testing)"

He is right. Failing to reject a null is not evidence for it, and at this
sample size the test could not have detected anything smaller than the effects
that matter. The fix is a two one-sided tests (TOST) procedure, which asks a
different and answerable question: can we rule out a difference larger than
some bound? That turns "we found nothing" into "we can exclude anything bigger
than X", which is a claim the data can actually support.

This scores both arms of the transfer comparison per sequence against ground
truth, pairs them by sequence, and reports:

  * the paired mean difference with a bootstrap confidence interval
  * the smallest equivalence bound the data can establish at alpha=0.05
  * a TOST verdict against the minimum detectable difference the paper quotes

Usage:
    python3 transfer_equivalence.py --gt <dir of MOT17-XX-YYY.txt> \\
        --arm reference=<dir> --arm dance_reid=<dir> [--out transfer_out]
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np


def score_per_sequence(gt_dir, arms):
    """Score every arm on every sequence with TrackEval, at IoU 0.5."""
    import trackeval

    seqs = sorted(set.intersection(*[
        {f[:-4] for f in os.listdir(d) if f.endswith(".txt")} for d in arms.values()
    ]))
    seqs = [s for s in seqs if os.path.exists(os.path.join(gt_dir, s + ".txt"))]
    if not seqs:
        sys.exit("no sequence is present in every arm and in the ground truth")

    root = tempfile.mkdtemp(prefix="te_")
    gt_root = os.path.join(root, "gt")
    tr_root = os.path.join(root, "trackers")
    seq_root = os.path.join(root, "seqmaps")
    os.makedirs(seq_root, exist_ok=True)

    for seq in seqs:
        d = os.path.join(gt_root, seq, "gt")
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(gt_dir, seq + ".txt"), os.path.join(d, "gt.txt"))
        last = 0
        with open(os.path.join(gt_dir, seq + ".txt")) as fh:
            for line in fh:
                if line.strip():
                    last = max(last, int(float(line.split(",")[0])))
        with open(os.path.join(gt_root, seq, "seqinfo.ini"), "w") as fh:
            fh.write("[Sequence]\nname=%s\nseqLength=%d\n" % (seq, last))

    with open(os.path.join(seq_root, "x.txt"), "w") as fh:
        fh.write("name\n" + "\n".join(seqs) + "\n")

    for name, src in arms.items():
        d = os.path.join(tr_root, name, "data")
        os.makedirs(d, exist_ok=True)
        for seq in seqs:
            shutil.copy(os.path.join(src, seq + ".txt"), os.path.join(d, seq + ".txt"))

    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False,
                       TIME_PROGRESS=False, OUTPUT_SUMMARY=False,
                       OUTPUT_DETAILED=False, PLOT_CURVES=False,
                       DISPLAY_LESS_PROGRESS=True))
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(dict(GT_FOLDER=gt_root, TRACKERS_FOLDER=tr_root,
                       BENCHMARK="MOT17", SPLIT_TO_EVAL="x",
                       SEQMAP_FILE=os.path.join(seq_root, "x.txt"),
                       SKIP_SPLIT_FOL=True, DO_PREPROC=False,
                       TRACKERS_TO_EVAL=list(arms),
                       SEQ_INFO={s: None for s in seqs}))

    ev = trackeval.Evaluator(ev_cfg)
    res, _ = ev.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)],
                         [trackeval.metrics.HOTA({"THRESHOLD": 0.5})])
    block = res["MotChallenge2DBox"]

    out = {}
    for name in arms:
        out[name] = {s: 100.0 * float(np.mean(block[name][s]["pedestrian"]["HOTA"]["HOTA"]))
                     for s in seqs}
    shutil.rmtree(root, ignore_errors=True)
    return seqs, out


def scene_of(seq):
    """MOT17-02-DPM -> MOT17-02. Detector variants of a scene are not
    independent observations, so they must be averaged, not pooled."""
    parts = seq.split("-")
    return "-".join(parts[:2])


def tost(diffs, bound, n_boot=20000, seed=0):
    """Two one-sided tests by bootstrap, which needs no normality assumption."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    # p for H0: mean <= -bound   and   H0: mean >= +bound
    p_lower = float(np.mean(boot <= -bound))
    p_upper = float(np.mean(boot >= bound))
    return max(p_lower, p_upper), boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path, repeatable; the first is the reference")
    ap.add_argument("--mdd", type=float, default=1.73,
                    help="minimum detectable difference the paper quotes")
    ap.add_argument("--out", default="transfer_out")
    a = ap.parse_args()

    arms = {}
    for spec in a.arm:
        k, _, v = spec.partition("=")
        arms[k] = v
    ref = list(arms)[0]

    seqs, scores = score_per_sequence(a.gt, arms)

    # Average detector variants within a scene: 21 sequences are 7 scenes.
    scenes = sorted({scene_of(s) for s in seqs})
    by_scene = {name: {sc: float(np.mean([scores[name][s] for s in seqs
                                          if scene_of(s) == sc]))
                       for sc in scenes} for name in arms}

    lines = []
    w = max(len(x) for x in arms)
    lines.append("Scored per sequence with TrackEval at IoU 0.5, then averaged")
    lines.append("within scene: %d sequences are %d independent scenes."
                 % (len(seqs), len(scenes)))
    lines.append("")
    lines.append("  %-10s " % "scene" + " ".join("%*s" % (w, n) for n in arms))
    for sc in scenes:
        lines.append("  %-10s " % sc + " ".join("%*.2f" % (w, by_scene[n][sc]) for n in arms))
    lines.append("  %-10s " % "mean" +
                 " ".join("%*.2f" % (w, np.mean(list(by_scene[n].values()))) for n in arms))
    lines.append("")

    rng = np.random.default_rng(0)
    for name in arms:
        if name == ref:
            continue
        d = np.array([by_scene[name][sc] - by_scene[ref][sc] for sc in scenes])
        n = len(d)
        boot = rng.choice(d, size=(20000, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        lines.append("=" * 68)
        lines.append("%s  vs  %s" % (name, ref))
        lines.append("=" * 68)
        lines.append("  paired per-scene differences (HOTA):")
        lines.append("    " + "  ".join("%+.2f" % x for x in d))
        lines.append("  mean difference      : %+.3f" % d.mean())
        lines.append("  95%% bootstrap CI     : [%+.3f, %+.3f]" % (lo, hi))
        lines.append("  wins / losses        : %d / %d" % (int((d > 0).sum()),
                                                           int((d < 0).sum())))

        # Smallest bound for which TOST rejects at alpha = 0.05.
        smallest = None
        for b in np.arange(0.05, 6.0, 0.05):
            p, _ = tost(d, b)
            if p < 0.05:
                smallest = b
                break
        p_at_mdd, _ = tost(d, a.mdd)

        lines.append("")
        lines.append("  Equivalence testing (TOST, bootstrap, alpha = 0.05):")
        if smallest is not None:
            lines.append("    smallest bound the data can establish: +/- %.2f HOTA" % smallest)
            lines.append("    i.e. a true difference larger than %.2f HOTA in either" % smallest)
            lines.append("    direction is excluded at the 5%% level.")
        else:
            lines.append("    no bound below 6 HOTA can be established at this n.")
        lines.append("    against the paper's stated MDD of %.2f: p = %.4f  (%s)"
                     % (a.mdd, p_at_mdd,
                        "equivalent within the MDD" if p_at_mdd < 0.05
                        else "cannot establish equivalence even at the MDD"))
        lines.append("")
        lines.append("  Reading: a non-significant difference test would have said only")
        lines.append("  'no difference detected'. TOST states what is actually excluded,")
        lines.append("  which is the claim the manuscript should make.")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    print("wrote %s/report.txt" % a.out)


if __name__ == "__main__":
    main()

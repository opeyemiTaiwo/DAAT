#!/usr/bin/env python3
"""
make_fig_dancequal.py -- rebuild Figure 2 with ground truth overlaid and the
sequence chosen by a stated rule.

Reviewer 4, comment 17: "Figure 2 shows horizontal trajectories that look
continuous, but it does not show correctness against GT. Given the thousands
of switches reported in Table 10 for this dataset, it risks being read as a
cherry-picked success. Please overlay GT or select the sequence by a stated
rule."

Both, then. The rule is the median sequence by HOTA among those the run can
serve, computed here with TrackEval rather than asserted, so the figure is
neither the best nor the worst case. Ground truth supplies the trajectories
and the tracker identity assigned to each is drawn on top of them, so an
identity switch appears as a change of color partway along a line instead
of being invisible. The previous version plotted the emitted track file
alone, in which a switch and a correct hand-off look identical.

The sequence in the submitted figure, dancetrack0007, is not among the
validation sequences for which we hold annotation, so it cannot carry an
overlay at all; that is why the figure moves rather than being amended.

Usage:
    python3 make_fig_dancequal.py --run <substring> [--out figs/dancequal.pdf]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flag_calibration import (SPATIAL_GATE, clear_match, find_switches,
                              load_flags, load_mot)

# Paths default to the reproduction bundle's own layout, which is the parent
# of the directory holding this script. Override on the command line, or with
# the DAAT_GT, DAAT_TRACKS and DAAT_LOGS environment variables, if your copy
# lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.dirname(_HERE)
_DATA = os.path.join(_BUNDLE, "data")
GT_ROOT = Path(os.environ.get("DAAT_GT", os.path.join(_DATA, "dancetrack_gt")))
TRACKS = Path(os.environ.get("DAAT_TRACKS", os.path.join(_DATA, "dancetrack_tracks")))
LOGS = [os.environ.get("DAAT_LOGS", os.path.join(_DATA, "logs", "dancetrack_val"))]


def seqs_for_run(run_sub):
    """Sequences the run can serve: tracker output, log and GT together."""
    tmap, lmap = {}, {}
    for dirpath, _, files in os.walk(TRACKS):
        if run_sub not in dirpath:
            continue
        for f in files:
            if f.startswith("dancetrack") and f.endswith(".txt"):
                tmap[f[:-4]] = os.path.join(dirpath, f)
    for root in LOGS:
        for dirpath, _, files in os.walk(root):
            if run_sub not in dirpath:
                continue
            for f in files:
                if f.startswith("dancetrack") and f.endswith("_ids_events.csv"):
                    lmap.setdefault(f.split("_")[0], os.path.join(dirpath, f))
    gmap = {d: str(GT_ROOT / d / "gt" / "gt.txt")
            for d in os.listdir(GT_ROOT)
            if (GT_ROOT / d / "gt" / "gt.txt").exists()}
    keys = sorted(set(tmap) & set(lmap) & set(gmap))
    return [(k, tmap[k], lmap[k], gmap[k]) for k in keys]


def hota_per_sequence(items):
    """TrackEval HOTA at IoU 0.5, one score per sequence."""
    import shutil
    import tempfile

    import trackeval

    root = tempfile.mkdtemp(prefix="fig2_")
    gt_root, tr_root = os.path.join(root, "gt"), os.path.join(root, "trk")
    seq_root = os.path.join(root, "seqmaps")
    os.makedirs(seq_root, exist_ok=True)
    names = [k for k, _, _, _ in items]
    for k, tp, _, gp in items:
        d = os.path.join(gt_root, k, "gt")
        os.makedirs(d, exist_ok=True)
        shutil.copy(gp, os.path.join(d, "gt.txt"))
        last = 0
        with open(gp) as fh:
            for line in fh:
                if line.strip():
                    last = max(last, int(float(line.split(",")[0])))
        with open(os.path.join(gt_root, k, "seqinfo.ini"), "w") as fh:
            fh.write("[Sequence]\nname=%s\nseqLength=%d\n" % (k, last))
        d2 = os.path.join(tr_root, "ours", "data")
        os.makedirs(d2, exist_ok=True)
        shutil.copy(tp, os.path.join(d2, k + ".txt"))
    with open(os.path.join(seq_root, "x.txt"), "w") as fh:
        fh.write("name\n" + "\n".join(names) + "\n")

    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False,
                       PRINT_CONFIG=False, TIME_PROGRESS=False,
                       OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                       PLOT_CURVES=False, DISPLAY_LESS_PROGRESS=True))
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(dict(GT_FOLDER=gt_root, TRACKERS_FOLDER=tr_root,
                       BENCHMARK="MOT17", SPLIT_TO_EVAL="x",
                       SEQMAP_FILE=os.path.join(seq_root, "x.txt"),
                       SKIP_SPLIT_FOL=True, DO_PREPROC=False,
                       TRACKERS_TO_EVAL=["ours"],
                       SEQ_INFO={s: None for s in names}))
    ev = trackeval.Evaluator(ev_cfg)
    res, _ = ev.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)],
                         [trackeval.metrics.HOTA({"THRESHOLD": 0.5})])
    block = res["MotChallenge2DBox"]["ours"]
    shutil.rmtree(root, ignore_errors=True)
    return {s: 100.0 * float(np.mean(block[s]["pedestrian"]["HOTA"]["HOTA"]))
            for s in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="Test-27-sam3mask/27_mask_w75")
    ap.add_argument("--out", default=os.path.join(_BUNDLE, "out", "dancequal.pdf"))
    ap.add_argument("--max-ids", type=int, default=6)
    a = ap.parse_args()

    items = seqs_for_run(a.run)
    if not items:
        sys.exit("no sequence in run %r has all three artifacts" % a.run)
    print("run %s serves %d sequences: %s"
          % (a.run, len(items), ", ".join(k for k, _, _, _ in items)))

    scores = hota_per_sequence(items)
    order = sorted(scores, key=lambda s: scores[s])
    print("\nHOTA per sequence (TrackEval, IoU 0.5):")
    for s in order:
        print("   %-16s %6.2f" % (s, scores[s]))
    pick = order[len(order) // 2] if len(order) % 2 else order[len(order) // 2 - 1]
    print("\nselection rule: median HOTA over the %d sequences -> %s (%.2f)"
          % (len(order), pick, scores[pick]))

    tp, lp, gp = next((t, l, g) for k, t, l, g in items if k == pick)
    trk = load_mot(tp, is_gt=False)
    gt = load_mot(gp, is_gt=True)
    m = clear_match(trk, gt, SPATIAL_GATE)
    sw = find_switches(m)
    fl = load_flags(lp)
    print("\n%s: %d GT rows, %d track rows, %d matched, %d switches, %d flags"
          % (pick, len(gt), len(trk), len(m), len(sw), len(fl)))

    # ---- plot -----------------------------------------------------------
    W = max(gt.x2.max(), trk.x2.max())
    gid_order = (gt.groupby("id").size().sort_values(ascending=False)
                 .index.tolist()[:a.max_ids])
    m = m[m.gt_id.isin(gid_order)]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    cmap = plt.get_cmap("tab10")
    tids = sorted(m.trk_id.unique())
    tcol = {t: cmap(i % 10) for i, t in enumerate(tids)}

    gt_proxy = None
    for gi, gid in enumerate(gid_order):
        g = m[m.gt_id == gid].sort_values("frame")
        if g.empty:
            continue
        xc = ((g.x + g.x2) / 2.0 / W).to_numpy()
        fr = g.frame.to_numpy()
        tt = g.trk_id.to_numpy()
        # Grey spine is the ground-truth trajectory. Where a tracker identity
        # was matched to it, that identity's color is drawn on top, so grey
        # showing through means the object was tracked by nothing at all and a
        # change of color is an identity switch.
        ln, = ax.plot(fr, xc, color="0.72", lw=3.0, solid_capstyle="butt",
                      zorder=1)
        gt_proxy = gt_proxy or ln
        cuts = np.flatnonzero(np.diff(tt)) + 1
        for seg in np.split(np.arange(len(fr)), cuts):
            if len(seg) < 2:
                continue
            ax.plot(fr[seg], xc[seg], color=tcol[tt[seg[0]]], lw=1.4, zorder=2)

    swv = sw[sw.gt_id.isin(gid_order)]
    if not swv.empty:
        ax.scatter(swv.frame, (swv.x + swv.x2) / 2.0 / W, s=34,
                   facecolors="none", edgecolors="crimson", lw=1.1, zorder=4,
                   label="identity switch (GT-matched)")
    if not fl.empty:
        ax.scatter(fl.frame, np.full(len(fl), 1.015), marker="|", s=46,
                   color="#1f4e79", lw=0.9, zorder=3, label="ambiguity flag")

    ax.set_xlabel("frame")
    ax.set_ylabel("normalized horizontal position")
    ax.set_ylim(-0.03, 1.09)
    ax.set_title("%s, the median sequence by HOTA of the %d scored"
                 % (pick, len(order)), fontsize=9.5)
    if gt_proxy is not None:
        gt_proxy.set_label("ground truth, no identity matched")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("\nwrote %s" % a.out)

    # ---- self-check: the caption's numbers must come from here ----------
    print("\ncaption figures, to be transcribed verbatim:")
    print("  sequence            : %s" % pick)
    print("  selection rule      : median HOTA of %d, HOTA %.2f"
          % (len(order), scores[pick]))
    print("  GT identities shown : %d of %d" % (len(gid_order), gt.id.nunique()))
    print("  frames              : %d" % int(gt.frame.max()))
    print("  GT switches shown   : %d (of %d in the sequence)"
          % (len(swv), len(sw)))
    print("  flags in sequence   : %d" % len(fl))
    print("  run                 : %s" % a.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

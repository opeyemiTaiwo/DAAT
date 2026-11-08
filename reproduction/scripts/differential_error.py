#!/usr/bin/env python3
"""
differential_error.py -- Table 4, differential identity-switch rate on MOT17.

Reviewer 4 asked for three things this script supplies:

  1. per-bin counts and confidence intervals, not bare ratios;
  2. a test of the confound, since occluded boxes and small boxes are not
     independent, so two marginal tables can both be driven by one effect;
  3. intervals computed at the real sample size, which is 7 scenes and not
     21 sequences, because the three detector variants of a scene are
     byte-identical runs under private detections.

Method
------
A new-identity mint (stage `6_new`) is the event that produces a genuine
identity switch; re-activations and revivals preserve the track id. Each mint
is matched to the ground-truth box it best overlaps in that frame, and takes
that box's visibility and height. The denominator is every ground-truth
pedestrian box in the bin, so the rate is switches per 1000 considered boxes,
as in the manuscript.

Confidence intervals are bootstrapped over SCENES (n=7), which is the unit
that varies. Bootstrapping over boxes would treat 40,000 correlated
observations as independent and produce intervals several times too narrow.

Usage
-----
    python differential_error.py /path/to/daat-work [--run RUN] [--out DIR]

    --run   substring selecting the run directory under daat-work, e.g.
            "Test-6-YOLOX/ALL-21" (default) or "Test-16b-fpfix"
    --gt    root holding MOT17/train/<seq>/gt/gt.txt
            (default: searched for under daat-work)

Outputs (default ./diff_error_out):
    visibility_bins.csv   rate, count, denominator, CI per visibility bin
    height_bins.csv       the same by box height
    joint_bins.csv        visibility x height, the confound check
    report.txt            the console report
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Manuscript claims, Table 4 and Sec. 4.5. Edit when the manuscript changes.
# --------------------------------------------------------------------------

CLAIM_VIS_RATIO    = 8.2   # low visibility (<0.3) vs high (>0.7)
CLAIM_HEIGHT_RATIO = 3.4   # small boxes vs large

TRAIN_SCENES = ["MOT17-02", "MOT17-04", "MOT17-05",
                "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

# A new-id mint is the switch source. Re-activation, revival and the
# unconfirmed pass all preserve the track id.
SWITCH_STAGE = "6_new"

VIS_BINS    = [0.0, 0.3, 0.5, 0.7, 1.01]
VIS_LABELS  = ["<0.3", "0.3-0.5", "0.5-0.7", ">0.7"]

N_BOOT = 10_000
RNG    = np.random.default_rng(20261106)


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def find_gt_root(root: Path, say: Report) -> Path | None:
    """Locate a directory holding MOT17/train/<seq>/gt/gt.txt."""
    hits = glob.glob(str(root / "**" / "MOT17-02*" / "gt" / "gt.txt"), recursive=True)
    if not hits:
        return None
    # .../MOT17/train/MOT17-02-DPM/gt/gt.txt  ->  .../MOT17/train
    return Path(hits[0]).parent.parent.parent


def load_gt(gt_root: Path, scene: str) -> pd.DataFrame | None:
    """MOT17 gt.txt: frame,id,x,y,w,h,conf,class,visibility. Identical across
    the three detector variants of a scene, so the first match is taken."""
    hits = sorted(glob.glob(str(gt_root / f"{scene}-*" / "gt" / "gt.txt")))
    if not hits:
        return None
    d = pd.read_csv(hits[0], header=None,
                    names=["frame", "id", "x", "y", "w", "h",
                           "conf", "cls", "vis"],
                    usecols=range(9))
    # Evaluated pedestrians only: class 1, flagged as considered.
    d = d[(d["cls"] == 1) & (d["conf"] == 1)].copy()
    d["x2"] = d.x + d.w
    d["y2"] = d.y + d.h
    d["scene"] = scene
    return d


def find_run_logs(root: Path, run: str, say: Report) -> dict[str, Path]:
    """One -DPM log per training scene, from the run whose path contains `run`."""
    out: dict[str, Path] = {}
    for f in root.glob("**/*-DPM_ids_events.csv"):
        scene = f.name.split("-DPM")[0]
        if scene in TRAIN_SCENES and run in str(f.parent):
            out[scene] = f
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def parse_box(s) -> tuple[float, float, float, float] | None:
    """det_box is serialized as '[x1, y1, x2, y2]'."""
    if not isinstance(s, str):
        return None
    try:
        b = ast.literal_eval(s)
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except Exception:                                     # noqa: BLE001
        return None


def iou_against(box, gx1, gy1, gx2, gy2) -> np.ndarray:
    x1, y1, x2, y2 = box
    ix1 = np.maximum(x1, gx1)
    iy1 = np.maximum(y1, gy1)
    ix2 = np.minimum(x2, gx2)
    iy2 = np.minimum(y2, gy2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    union = a + b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_switches(events: pd.DataFrame, gt: pd.DataFrame,
                   iou_min: float) -> pd.DataFrame:
    """Attach the best-overlapping GT box's visibility and height to each mint."""
    by_frame = {f: g for f, g in gt.groupby("frame")}
    rows = []
    for _, e in events.iterrows():
        box = parse_box(e.det_box)
        if box is None:
            rows.append((np.nan, np.nan, 0.0))
            continue
        g = by_frame.get(int(e.frame))
        if g is None or g.empty:
            rows.append((np.nan, np.nan, 0.0))
            continue
        ious = iou_against(box, g.x.to_numpy(), g.y.to_numpy(),
                           g.x2.to_numpy(), g.y2.to_numpy())
        k = int(np.argmax(ious))
        if ious[k] < iou_min:
            rows.append((np.nan, np.nan, float(ious[k])))
        else:
            rows.append((float(g.vis.iloc[k]), float(g.h.iloc[k]), float(ious[k])))
    out = events.copy()
    out[["gt_vis", "gt_h", "match_iou"]] = pd.DataFrame(rows, index=events.index)
    return out


# --------------------------------------------------------------------------
# Rates and intervals
# --------------------------------------------------------------------------

def boot_ci(num_by_scene: dict[str, int], den_by_scene: dict[str, int],
            n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap over scenes. The scene is the unit that varies;
    resampling boxes would understate the interval several-fold."""
    scenes = sorted(set(num_by_scene) | set(den_by_scene))
    if len(scenes) < 2:
        return (np.nan, np.nan)
    num = np.array([num_by_scene.get(s, 0) for s in scenes], float)
    den = np.array([den_by_scene.get(s, 0) for s in scenes], float)
    idx = RNG.integers(0, len(scenes), size=(n_boot, len(scenes)))
    n = num[idx].sum(axis=1)
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(d > 0, n / d * 1000.0, np.nan)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def rate_table(matched: pd.DataFrame, gt: pd.DataFrame,
               col_ev: str, col_gt: str, bins, labels) -> pd.DataFrame:
    """switches per 1000 considered boxes, per bin, with a bootstrap CI."""
    ev = matched.dropna(subset=[col_ev]).copy()
    ev["bin"] = pd.cut(ev[col_ev], bins=bins, labels=labels,
                       right=False, include_lowest=True)
    g = gt.copy()
    g["bin"] = pd.cut(g[col_gt], bins=bins, labels=labels,
                      right=False, include_lowest=True)

    rows = []
    for lab in labels:
        n_by = ev[ev.bin == lab].groupby("scene").size().to_dict()
        d_by = g[g.bin == lab].groupby("scene").size().to_dict()
        n, d = sum(n_by.values()), sum(d_by.values())
        rate = n / d * 1000.0 if d else np.nan
        lo, hi = boot_ci(n_by, d_by)
        rows.append({"bin": lab, "switches": n, "gt_boxes": d,
                     "rate_per_1000": round(rate, 2) if d else np.nan,
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                     "scenes_with_data": len(d_by)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="daat-work tree (contains MOT-ALL/ and the GT)")
    ap.add_argument("--run", default="Test-6-YOLOX/ALL-21",
                    help="substring selecting the run directory")
    ap.add_argument("--gt", default=None, help="root holding MOT17/train/<seq>/gt/gt.txt")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="minimum IoU to accept an event-to-GT match (default 0.5)")
    ap.add_argument("--out", default="diff_error_out")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    say = Report()
    say(f"differential_error.py   root={root}")
    say(f"                        run ~ {args.run!r}   match IoU >= {args.iou}")

    # ---- inputs ----------------------------------------------------------
    gt_root = Path(args.gt).expanduser().resolve() if args.gt else find_gt_root(root, say)
    if gt_root is None or not gt_root.is_dir():
        say("\nerror: could not find MOT17 train ground truth.")
        say("       Expected .../MOT17/train/MOT17-02-*/gt/gt.txt under the tree,")
        say("       or pass --gt explicitly.")
        return 2
    say(f"                        gt  = {gt_root}")

    logs = find_run_logs(root, args.run, say)
    say("")
    say(f"training scenes with logs: {len(logs)} of {len(TRAIN_SCENES)}")
    for s in TRAIN_SCENES:
        say(f"   {'OK ' if s in logs else '!! '}{s}"
            + (f"  {logs[s].parent}" if s in logs else ""))
    if len(logs) < 2:
        say("\nerror: need at least two scenes. Check --run.")
        return 2

    # ---- load and match --------------------------------------------------
    say.rule("1. MATCHING SWITCH EVENTS TO GROUND TRUTH")
    gts, matches = [], []
    for scene in TRAIN_SCENES:
        if scene not in logs:
            continue
        gt = load_gt(gt_root, scene)
        if gt is None:
            say(f"  !! {scene}: no gt.txt")
            continue
        ev = pd.read_csv(logs[scene])
        mints = ev[ev.stage.astype(str) == SWITCH_STAGE].copy()
        mints["scene"] = scene
        m = match_switches(mints, gt, args.iou)
        ok = m.gt_vis.notna().sum()
        say(f"  {scene}: {len(gt):>6,} GT boxes   {len(mints):>5} mints   "
            f"{ok:>5} matched ({ok/max(len(mints),1)*100:4.1f}%)")
        gts.append(gt)
        matches.append(m)

    gt_all = pd.concat(gts, ignore_index=True)
    mt_all = pd.concat(matches, ignore_index=True)
    unmatched = mt_all.gt_vis.isna().sum()
    say("")
    say(f"  total GT boxes : {len(gt_all):,}")
    say(f"  total mints    : {len(mt_all):,}")
    say(f"  unmatched      : {unmatched:,} ({unmatched/max(len(mt_all),1)*100:.1f}%)")
    say("  Unmatched mints are detections with no GT box above the IoU floor,")
    say("  i.e. false positives. They are excluded from the rate numerator and")
    say("  reported here so the exclusion is visible rather than silent.")

    # ---- visibility ------------------------------------------------------
    say.rule("2. BY VISIBILITY (occlusion severity)")
    vis = rate_table(mt_all, gt_all, "gt_vis", "vis", VIS_BINS, VIS_LABELS)
    say(vis.to_string(index=False))
    vis.to_csv(out / "visibility_bins.csv", index=False)

    lo_r = vis.loc[vis.bin == "<0.3", "rate_per_1000"].iloc[0]
    hi_r = vis.loc[vis.bin == ">0.7", "rate_per_1000"].iloc[0]
    ratio_v = lo_r / hi_r if hi_r else np.nan
    say("")
    say(f"  low (<0.3) vs high (>0.7) : {ratio_v:.1f}x")
    say(f"  manuscript claims         : {CLAIM_VIS_RATIO}x")
    say(f"  -> {'consistent' if abs(ratio_v - CLAIM_VIS_RATIO) < 0.5 else 'DOES NOT MATCH, correct the manuscript'}")

    # ---- height ----------------------------------------------------------
    say.rule("3. BY BOX HEIGHT (scale)")
    qs = gt_all.h.quantile([0, .25, .5, .75, 1.0]).tolist()
    qs[-1] += 1
    h_labels = [f"{qs[i]:.0f}-{qs[i+1]:.0f}px" for i in range(4)]
    say(f"  quartile edges from the GT height distribution: "
        + ", ".join(f"{q:.0f}" for q in qs))
    hgt = rate_table(mt_all, gt_all, "gt_h", "h", qs, h_labels)
    say("")
    say(hgt.to_string(index=False))
    hgt.to_csv(out / "height_bins.csv", index=False)

    ratio_h = hgt.rate_per_1000.iloc[0] / hgt.rate_per_1000.iloc[-1] \
        if hgt.rate_per_1000.iloc[-1] else np.nan
    say("")
    say(f"  smallest vs largest quartile : {ratio_h:.1f}x")
    say(f"  manuscript claims            : {CLAIM_HEIGHT_RATIO}x")
    say(f"  -> {'consistent' if abs(ratio_h - CLAIM_HEIGHT_RATIO) < 0.5 else 'DOES NOT MATCH, correct the manuscript'}")
    say("")
    say("  Monotonicity check (Reviewer 4 noted the published column is not")
    say("  monotonic in height):")
    rates = hgt.rate_per_1000.tolist()
    mono = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
    say(f"    rates by increasing height: {rates}")
    say(f"    monotonically decreasing  : {mono}")

    # ---- the confound ----------------------------------------------------
    say.rule("4. THE CONFOUND: VISIBILITY x HEIGHT")
    say("  Reviewer 4: visibility and height are probably not independent, so")
    say("  two marginal tables can both be driven by a single effect. The joint")
    say("  table shows whether each one survives conditioning on the other.")
    say("")

    gt_all = gt_all.copy()
    mt_ok = mt_all.dropna(subset=["gt_vis", "gt_h"]).copy()
    gt_all["vbin"] = pd.cut(gt_all.vis, VIS_BINS, labels=VIS_LABELS,
                            right=False, include_lowest=True)
    gt_all["hbin"] = pd.cut(gt_all.h, qs, labels=h_labels,
                            right=False, include_lowest=True)
    mt_ok["vbin"] = pd.cut(mt_ok.gt_vis, VIS_BINS, labels=VIS_LABELS,
                           right=False, include_lowest=True)
    mt_ok["hbin"] = pd.cut(mt_ok.gt_h, qs, labels=h_labels,
                           right=False, include_lowest=True)

    rows = []
    for v in VIS_LABELS:
        for h in h_labels:
            n = int(((mt_ok.vbin == v) & (mt_ok.hbin == h)).sum())
            d = int(((gt_all.vbin == v) & (gt_all.hbin == h)).sum())
            rows.append({"visibility": v, "height": h, "switches": n,
                         "gt_boxes": d,
                         "rate_per_1000": round(n / d * 1000, 2) if d else np.nan})
    joint = pd.DataFrame(rows)
    joint.to_csv(out / "joint_bins.csv", index=False)
    say(joint.pivot(index="visibility", columns="height",
                    values="rate_per_1000").to_string())
    say("")
    say("  supporting counts (switches / GT boxes):")
    say(joint.assign(cell=joint.switches.astype(str) + " / " + joint.gt_boxes.astype(str))
             .pivot(index="visibility", columns="height", values="cell").to_string())

    say("")
    say("  correlation between visibility and height over GT boxes: "
        f"{gt_all[['vis', 'h']].corr().iloc[0, 1]:+.3f}")
    say("  A correlation near zero means the two marginals are close to")
    say("  independent and both effects stand. A strong one means they are not")
    say("  separable at this sample size, and the manuscript should say so.")

    # ---- summary ---------------------------------------------------------
    say.rule("SUMMARY")
    say(f"  scenes                    : {len(logs)}  (bootstrap unit, n={len(logs)})")
    say(f"  GT pedestrian boxes       : {len(gt_all):,}")
    say(f"  identity-switch events    : {len(mt_all):,}, matched {len(mt_ok):,}")
    say(f"  visibility ratio          : {ratio_v:.1f}x   (manuscript {CLAIM_VIS_RATIO}x)")
    say(f"  height ratio              : {ratio_h:.1f}x   (manuscript {CLAIM_HEIGHT_RATIO}x)")
    say("")
    say(f"  CSVs written to {out}")
    say.save(out / "report.txt")
    print(f"\nreport.txt written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

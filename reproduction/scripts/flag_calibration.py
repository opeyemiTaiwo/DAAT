#!/usr/bin/env python3
"""
flag_calibration.py -- Table 10, the ambiguity flag against real identity switches.

The flag is the set of refusals the audit layer records: an appearance margin
below delta, or a similarity below the gate. Those are the events the layer
offers an examiner as "this link is contested". Whether that is useful depends
on two things the manuscript must report together:

  precision -- of the links the flag warns about, how many sit on a real
               identity error;
  recall    -- of the real identity errors, how many the flag warns about.
               Its complement is the silent-error rate, which is the number
               that matters evidentially: an unflagged link presents to an
               examiner as uncontested.

Reviewer 4 asked for per-window figures and a chance baseline, because a flag
that fires often enough will intersect switches by coincidence. This script
supplies both.

Method
------
Real switches are located by CLEAR matching: tracker output is matched to
ground truth frame by frame at an IoU gate, each GT identity is followed, and
a switch is recorded wherever the tracker identity assigned to a GT identity
changes between consecutive matched appearances. No audit log is consulted in
finding them, so the flag is scored against an independent reference.

A switch counts as flagged if a refusal occurs within W frames of it and
overlaps the GT box at the same spatial gate. The chance baseline redraws the
same number of refusals uniformly over the frames that carried any detection,
preserving the firing rate while destroying any association with error, and
reports the precision that produces.

Usage
-----
    python flag_calibration.py /path/to/daat-work [--run RUN] [--out DIR]

Outputs (default ./flag_cal_out):
    calibration.csv   precision, recall, silent rate and chance, per window
    switches.csv      every located switch, with whether a flag covered it
    report.txt        the console report
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Manuscript claims, Table 10 and Sec. 4.8.
# --------------------------------------------------------------------------

CLAIM_MOT17_SWITCHES  = 1204
CLAIM_PRECISION_RANGE = (0.65, 0.81)
CLAIM_SILENT_W30      = 0.006     # 0.6% silent at a 30-frame window
CLAIM_SILENT_W5       = 0.092     # 9.2% silent at a 5-frame window

TRAIN_SCENES = ["MOT17-02", "MOT17-04", "MOT17-05",
                "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

FLAG_REASONS = ("margin_failed", "gate_failed")

WINDOWS      = [5, 10, 30]
SPATIAL_GATE = 0.3                 # IoU, as in the manuscript
GATE_SWEEP   = [0.0, 0.1, 0.3, 0.5]  # 0.0 = temporal proximity only

N_CHANCE = 100   # override with --chance
RNG      = np.random.default_rng(20261106)


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

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_mot(path: str, is_gt: bool) -> pd.DataFrame:
    """MOT format: frame,id,x,y,w,h,conf,(class,vis)."""
    names = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
    d = pd.read_csv(path, header=None, names=names,
                    usecols=range(min(9, len(pd.read_csv(path, header=None, nrows=1).columns))))
    if is_gt:
        if "cls" in d:
            d = d[(d["cls"] == 1) & (d["conf"] == 1)]
    d = d.copy()
    d["x2"] = d.x + d.w
    d["y2"] = d.y + d.h
    return d[["frame", "id", "x", "y", "x2", "y2"]]


def clear_match(tracks: pd.DataFrame, gt: pd.DataFrame,
                gate: float) -> pd.DataFrame:
    """Greedy highest-IoU matching per frame. Returns frame, gt_id, trk_id."""
    rows = []
    tb = dict(tuple(tracks.groupby("frame")))
    for f, g in gt.groupby("frame"):
        t = tb.get(f)
        if t is None or t.empty:
            continue
        gb = g[["x", "y", "x2", "y2"]].to_numpy(float)
        tbx = t[["x", "y", "x2", "y2"]].to_numpy(float)
        M = iou_matrix(gb, tbx)
        used_g, used_t = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
        for i, j in order:
            if M[i, j] < gate:
                break
            if i in used_g or j in used_t:
                continue
            used_g.add(i); used_t.add(j)
            rows.append((int(f), int(g.id.iloc[i]), int(t.id.iloc[j]),
                         float(gb[i, 0]), float(gb[i, 1]),
                         float(gb[i, 2]), float(gb[i, 3])))
    return pd.DataFrame(rows, columns=["frame", "gt_id", "trk_id",
                                       "x", "y", "x2", "y2"])


def find_switches(matched: pd.DataFrame) -> pd.DataFrame:
    """A switch: the tracker id assigned to a GT identity changes between
    consecutive matched appearances of that identity."""
    out = []
    for gid, g in matched.sort_values("frame").groupby("gt_id"):
        prev_t, prev_f = None, None
        for _, r in g.iterrows():
            if prev_t is not None and r.trk_id != prev_t:
                out.append({"frame": int(r.frame), "gt_id": int(gid),
                            "from_trk": int(prev_t), "to_trk": int(r.trk_id),
                            "gap": int(r.frame - prev_f),
                            "x": r.x, "y": r.y, "x2": r.x2, "y2": r.y2})
            prev_t, prev_f = r.trk_id, r.frame
    return pd.DataFrame(out)


def parse_box(s):
    if not isinstance(s, str):
        return None
    try:
        b = ast.literal_eval(s)
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception:                                     # noqa: BLE001
        return None


def load_flags(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.reason.astype(str).str.startswith(FLAG_REASONS)].copy()
    boxes = d.det_box.map(parse_box)
    keep = boxes.notna()
    d = d[keep].copy()
    bb = np.array(boxes[keep].tolist(), dtype=float)
    d[["x", "y", "x2", "y2"]] = bb
    return d[["frame", "reason", "x", "y", "x2", "y2"]]


def score(switches: pd.DataFrame, flags: pd.DataFrame,
          window: int, gate: float) -> tuple[int, int, int]:
    """Returns (switches covered, flags that hit a switch, total)."""
    if switches.empty or flags.empty:
        return 0, 0, 0
    sb = switches[["x", "y", "x2", "y2"]].to_numpy(float)
    fb = flags[["x", "y", "x2", "y2"]].to_numpy(float)
    sf = switches.frame.to_numpy()
    ff = flags.frame.to_numpy()

    near = np.abs(sf[:, None] - ff[None, :]) <= window
    if not near.any():
        return 0, 0, len(switches)
    # gate 0 means temporal proximity alone, with no requirement that the
    # flag concern the same object. It is included because a published
    # figure may have been computed that way, and the comparison shows what
    # dropping the spatial condition buys.
    hit = near if gate <= 0 else (near & (iou_matrix(sb, fb) >= gate))
    return int(hit.any(axis=1).sum()), int(hit.any(axis=0).sum()), len(switches)


def chance_precision(switches: pd.DataFrame, flags: pd.DataFrame,
                     frames: np.ndarray, boxes: np.ndarray,
                     window: int, gate: float, n: int = N_CHANCE) -> float:
    """Redraw the same number of flags from the real detection pool, keeping
    the firing rate but removing any association with error."""
    if switches.empty or len(flags) == 0 or len(frames) == 0:
        return float("nan")
    k = len(flags)
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(frames), size=k)
        fake = pd.DataFrame({"frame": frames[idx]})
        fake[["x", "y", "x2", "y2"]] = boxes[idx]
        _, fh, _ = score(switches, fake, window, gate)
        vals.append(fh / k)
    return float(np.mean(vals))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--run", default="Test-6-YOLOX/ALL-21")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--chance", type=int, default=N_CHANCE,
                    help="redraws for the chance baseline (default 100; "
                         "lower it if the run is slow, 0 skips it)")
    ap.add_argument("--out", default="flag_cal_out")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    say = Report()
    say(f"flag_calibration.py   root={root}   run ~ {args.run!r}")

    if args.gt:
        gt_root = Path(args.gt).expanduser().resolve()
    else:
        hits = glob.glob(str(root / "**" / "MOT17-02*" / "gt" / "gt.txt"), recursive=True)
        if not hits:
            say("error: MOT17 train ground truth not found; pass --gt")
            return 2
        gt_root = Path(hits[0]).parent.parent.parent
    say(f"                      gt  = {gt_root}")

    # ---- gather per-scene inputs -----------------------------------------
    say.rule("1. INPUTS")
    scenes = {}
    for scene in TRAIN_SCENES:
        ev = [f for f in root.glob(f"**/{scene}-DPM_ids_events.csv")
              if args.run in str(f.parent)]
        txt = [f for f in root.glob(f"**/{scene}-DPM.txt")
               if args.run in str(f.parent)]
        gtf = sorted(glob.glob(str(gt_root / f"{scene}-*" / "gt" / "gt.txt")))
        ok = bool(ev) and bool(txt) and bool(gtf)
        say(f"  {'OK ' if ok else '!! '}{scene}   "
            f"log={'y' if ev else 'n'} txt={'y' if txt else 'n'} gt={'y' if gtf else 'n'}")
        if ok:
            scenes[scene] = (ev[0], txt[0], gtf[0])

    if not scenes:
        say("")
        say("error: no scene has all three inputs.")
        say("The tracker output .txt files are the usual gap. Extract them with:")
        say('  for p in 001 002 003 004 005 006; do')
        say('    unzip -o -q MOT-ALL-*-1-$p.zip "*/MOT17-*.txt" -d ~/daat-work/')
        say('  done')
        return 2

    # ---- locate real switches -------------------------------------------
    say.rule("2. REAL IDENTITY SWITCHES (CLEAR matching, independent of the log)")
    all_sw, all_fl, pool_f, pool_b = [], [], [], []
    for scene, (evp, txtp, gtp) in scenes.items():
        trk = load_mot(str(txtp), is_gt=False)
        gt = load_mot(gtp, is_gt=True)
        m = clear_match(trk, gt, SPATIAL_GATE)
        sw = find_switches(m)
        fl = load_flags(str(evp))
        sw["scene"] = scene
        fl["scene"] = scene
        say(f"  {scene}: {len(gt):>6,} gt  {len(trk):>6,} trk  "
            f"{len(m):>6,} matched  {len(sw):>4} switches  {len(fl):>4} flags")
        all_sw.append(sw); all_fl.append(fl)
        pool_f.append(trk.frame.to_numpy())
        pool_b.append(trk[["x", "y", "x2", "y2"]].to_numpy(float))

    SW = pd.concat(all_sw, ignore_index=True)
    FL = pd.concat(all_fl, ignore_index=True)
    say("")
    say(f"  total switches : {len(SW):,}   manuscript states {CLAIM_MOT17_SWITCHES:,}")
    say(f"  total flags    : {len(FL):,}")
    diff = abs(len(SW) - CLAIM_MOT17_SWITCHES)
    say(f"  -> {'consistent' if diff <= 0.05*CLAIM_MOT17_SWITCHES else 'DOES NOT MATCH; reconcile before publishing'}")

    # ---- calibrate -------------------------------------------------------
    say.rule("3. CALIBRATION BY ASSOCIATION WINDOW")
    rows = []
    for gate in GATE_SWEEP:
        for w in WINDOWS:
            cov = hits = tot = 0
            fl_tot = 0
            for scene in scenes:
                s = SW[SW.scene == scene]
                f = FL[FL.scene == scene]
                c, h, t = score(s, f, w, gate)
                cov += c; hits += h; tot += t; fl_tot += len(f)
            recall = cov / tot if tot else np.nan
            prec = hits / fl_tot if fl_tot else np.nan
            ch = (chance_precision(SW, FL,
                                   np.concatenate(pool_f),
                                   np.concatenate(pool_b), w, gate,
                                   n=args.chance)
                  if args.chance > 0 else float("nan"))
            rows.append({"iou_gate": gate, "window": w,
                         "switches": tot, "flags": fl_tot,
                         "switches_flagged": cov,
                         "recall": round(recall, 3),
                         "silent_rate": round(1 - recall, 3),
                         "precision": round(prec, 3),
                         "chance_precision": round(ch, 3),
                         "lift": round(prec / ch, 1) if ch else np.nan})
    cal = pd.DataFrame(rows)
    say(cal.to_string(index=False))
    cal.to_csv(out / "calibration.csv", index=False)
    SW.to_csv(out / "switches.csv", index=False)

    # ---- compare against the manuscript ----------------------------------
    say.rule("4. AGAINST THE MANUSCRIPT")
    main = cal[cal.iou_gate == SPATIAL_GATE]

    def at(w, col):
        r = main[main.window == w]
        return float(r[col].iloc[0]) if len(r) else np.nan

    for w, claim in ((30, CLAIM_SILENT_W30), (5, CLAIM_SILENT_W5)):
        got = at(w, "silent_rate")
        say(f"  silent rate at {w:>2}-frame window : {got:.3f}   "
            f"manuscript {claim:.3f}   "
            f"{'consistent' if abs(got-claim) < 0.02 else 'DOES NOT MATCH'}")

    lo, hi = main.precision.min(), main.precision.max()
    say(f"  precision range                : {lo:.2f} to {hi:.2f}   "
        f"manuscript {CLAIM_PRECISION_RANGE[0]} to {CLAIM_PRECISION_RANGE[1]}")
    say("")
    say("  Chance precision is the column that answers Reviewer 4. A flag that")
    say("  fires often enough will intersect switches by coincidence; lift is")
    say("  how much better than that the flag actually does. A lift near 1")
    say("  means the flag carries no information about real error, whatever")
    say("  its raw precision looks like.")
    say("")
    say("  Spatial gate sensitivity (the manuscript says it barely matters):")
    for g in GATE_SWEEP:
        sub = cal[(cal.iou_gate == g) & (cal.window == 30)]
        if len(sub):
            say(f"    IoU {g}: precision {sub.precision.iloc[0]:.3f}  "
                f"recall {sub.recall.iloc[0]:.3f}")

    say.rule("SUMMARY")
    say(f"  scenes                : {len(scenes)}")
    say(f"  real switches (CLEAR) : {len(SW):,}")
    say(f"  flags (margin + gate) : {len(FL):,}")
    say(f"  CSVs written to {out}")
    say.save(out / "report.txt")
    print(f"\nreport.txt written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

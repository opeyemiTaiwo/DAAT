#!/usr/bin/env python3
"""
flag_calibration_dance.py -- the DanceTrack half of Table 13, computed by the
same code path as the MOT17 half.

Table 13 currently reports its two halves under two different matching
criteria: MOT17 under an IoU 0.3 spatial gate, DanceTrack under temporal
co-occurrence alone. The caption concedes the halves "are not comparable".
Reviewer 4 objected to exactly that. This script recomputes the DanceTrack
half under the MOT17 criterion by importing the MOT17 script's own functions,
so the two halves differ in data and in nothing else.

It needs three artifacts per sequence and will not substitute for a missing
one:

  tracker output .txt   to locate real identity switches by CLEAR matching
  audit event log .csv  to locate the refusals that constitute the flag
  ground-truth gt.txt   to match against

The audit log is event-driven, so a frame with nothing to report writes
nothing and per-frame assignments cannot be recovered from it. The tracker
output is therefore not optional.

Because no recovered run holds all three for every validation sequence, the
script sweeps every run it finds and reports the per-run sequence count
alongside the figures, so a thin run is visible as thin rather than being
pooled with others. Pooling across runs is refused: fusion on and fusion off
are different systems and their refusal rates are not interchangeable.

Usage
-----
    python3 flag_calibration_dance.py --tracks <extracted dance_tracks dir> \\
        [--gt <val root>] [--logs <dir> ...] [--run SUBSTRING] [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flag_calibration import (  # noqa: E402  -- same code path, by design
    FLAG_REASONS,
    GATE_SWEEP,
    SPATIAL_GATE,
    WINDOWS,
    chance_precision,
    clear_match,
    find_switches,
    load_flags,
    load_mot,
    score,
)

# Paths default to the reproduction bundle's own layout, which is the parent
# of the directory holding this script. Override on the command line, or with
# the DAAT_GT, DAAT_TRACKS and DAAT_LOGS environment variables, if your copy
# lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.dirname(_HERE)
_DATA = os.path.join(_BUNDLE, "data")
GT_DEFAULT = os.environ.get("DAAT_GT", os.path.join(_DATA, "dancetrack_gt"))
LOG_DEFAULT = [os.environ.get("DAAT_LOGS", os.path.join(_DATA, "logs", "dancetrack_val"))]

SEQ_RE = re.compile(r"(dancetrack\d{4})", re.I)


def seq_of(name: str):
    m = SEQ_RE.search(os.path.basename(name))
    return m.group(1).lower() if m else None


def normalize_run(key: str) -> str:
    parts = [p for p in Path(key).parts if p]
    for i, p in enumerate(parts):
        u = p.upper()
        if u.startswith("TEST-") or u.startswith("DANCETRACK-") or u == "TEST-SUBMIT":
            return str(Path(*parts[i:]))
    return str(Path(*parts)) if parts else key


def index_tracks(root: Path):
    out = defaultdict(dict)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            s = seq_of(f)
            if s:
                run = normalize_run(os.path.relpath(dirpath, root))
                out[run][s] = os.path.join(dirpath, f)
    return out


def index_logs(roots):
    out = defaultdict(dict)
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dirpath, _, files in os.walk(r):
            if "DANCE" not in dirpath.upper():
                continue
            for f in files:
                if not f.lower().endswith(".csv"):
                    continue
                s = seq_of(f)
                if s:
                    run = normalize_run(os.path.relpath(dirpath, r))
                    out[run].setdefault(s, os.path.join(dirpath, f))
    return out


def index_gt(root: Path):
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(os.listdir(root)):
        p = root / d / "gt" / "gt.txt"
        if p.exists():
            s = seq_of(d + ".txt")
            if s:
                out[s] = str(p)
    return out


def run_one(seqs, tracks, logs, gt, n_chance, log):
    """Compute the calibration for one run over the sequences it can serve."""
    all_sw, all_fl, pool_f, pool_b = [], [], [], []
    per_seq = []
    for s in sorted(seqs):
        trk = load_mot(tracks[s], is_gt=False)
        g = load_mot(gt[s], is_gt=True)
        m = clear_match(trk, g, SPATIAL_GATE)
        sw = find_switches(m)
        fl = load_flags(logs[s])
        if not sw.empty:
            sw = sw.copy()
            sw["seq"] = s
            all_sw.append(sw)
        if not fl.empty:
            fl = fl.copy()
            fl["seq"] = s
            all_fl.append(fl)
        pool_f.append(trk.frame.to_numpy())
        pool_b.append(trk[["x", "y", "x2", "y2"]].to_numpy(float))
        per_seq.append((s, len(g), len(trk), len(m), len(sw), len(fl)))
        log(f"    {s}: {len(g):>6,} gt  {len(trk):>6,} trk  {len(m):>6,} matched"
            f"  {len(sw):>4} switches  {len(fl):>4} flags")

    sw = pd.concat(all_sw, ignore_index=True) if all_sw else pd.DataFrame()
    fl = pd.concat(all_fl, ignore_index=True) if all_fl else pd.DataFrame()
    frames = np.concatenate(pool_f) if pool_f else np.array([])
    boxes = np.vstack(pool_b) if pool_b else np.zeros((0, 4))

    if sw.empty or fl.empty:
        return per_seq, sw, fl, []

    rows = []
    for gate in GATE_SWEEP:
        for w in WINDOWS:
            cov, hit, tot = score(sw, fl, w, gate)
            rec = cov / tot if tot else float("nan")
            prec = hit / len(fl) if len(fl) else float("nan")
            ch = chance_precision(sw, fl, frames, boxes, w, gate, n=n_chance) \
                if n_chance else float("nan")
            rows.append(dict(gate=gate, window=w, switches=tot, flags=len(fl),
                             covered=cov, recall=rec, silent=1 - rec,
                             precision=prec, chance=ch,
                             lift=(prec / ch) if ch and ch == ch and ch > 0
                             else float("nan")))
    return per_seq, sw, fl, rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--gt", default=GT_DEFAULT)
    ap.add_argument("--logs", nargs="*", default=LOG_DEFAULT)
    ap.add_argument("--run", default=None,
                    help="restrict to runs whose path contains this substring")
    ap.add_argument("--chance", type=int, default=100)
    ap.add_argument("--min-seqs", type=int, default=1)
    ap.add_argument("--out", default="flag_dance_out")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    tracks = index_tracks(Path(a.tracks).resolve())
    logs = index_logs(a.logs)
    gt = index_gt(Path(a.gt).resolve())

    log(f"flag_calibration_dance.py")
    log(f"  tracks : {a.tracks}  ({len(tracks)} run folders)")
    log(f"  logs   : {', '.join(a.logs)}  ({len(logs)} run folders)")
    log(f"  gt     : {a.gt}  ({len(gt)} sequences)")
    log(f"  spatial gate {SPATIAL_GATE}, windows {WINDOWS}, "
        f"gate sweep {GATE_SWEEP}")
    log()

    cands = []
    for run, tmap in tracks.items():
        if a.run and a.run not in run:
            continue
        lmap = logs.get(run, {})
        usable = sorted(set(tmap) & set(lmap) & set(gt))
        if len(usable) >= a.min_seqs:
            cands.append((len(usable), run, usable))
    cands.sort(key=lambda r: -r[0])

    if not cands:
        log("No run holds tracker output, an event log and ground truth for any")
        log("sequence. The DanceTrack half cannot be recomputed.")
        (out / "report.txt").write_text("\n".join(lines) + "\n")
        return 2

    log("=" * 74)
    log("Runs with all three artifacts, by usable sequence count")
    log("=" * 74)
    for n, run, usable in cands:
        log(f"  {n:>2}  {run}")
    log()

    summary = []
    for n, run, usable in cands:
        log("=" * 74)
        log(f"RUN  {run}   ({n} usable sequence{'s' if n != 1 else ''})")
        log("=" * 74)
        per_seq, sw, fl, rows = run_one(usable, tracks[run], logs[run], gt,
                                        a.chance, log)
        if not rows:
            log("    no switches or no flags; nothing to calibrate")
            log()
            continue
        df = pd.DataFrame(rows)
        df.insert(0, "run", run)
        df.insert(1, "n_seqs", n)
        summary.append(df)

        log()
        log(f"    total: {int(df['switches'].iloc[0]):,} switches, "
            f"{int(df['flags'].iloc[0]):,} flags")
        log()
        log("    gate  W   recall  silent  precision  chance   lift")
        log("    " + "-" * 50)
        for _, r in df.iterrows():
            mark = "  <-- manuscript gate" if r.gate == SPATIAL_GATE and r.window == 30 else ""
            log(f"    {r.gate:>4.1f} {int(r.window):>2}   "
                f"{r.recall:6.3f}  {r.silent*100:5.1f}%  "
                f"{r.precision:9.3f}  {r.chance:6.3f}  "
                f"{r.lift:5.2f}x{mark}")
        log()

    if summary:
        allrows = pd.concat(summary, ignore_index=True)
        allrows.to_csv(out / "calibration_by_run.csv", index=False)
        log(f"wrote {out}/calibration_by_run.csv")

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    log(f"wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

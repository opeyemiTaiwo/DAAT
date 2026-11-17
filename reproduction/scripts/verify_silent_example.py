#!/usr/bin/env python3
"""
verify_silent_example.py -- check the worked silent-error example in
Sec. 4.8 against every surviving run that holds dancetrack0004.

The manuscript claims: in dancetrack0004 at frame 261, GT object 0 is
silently taken over by track 2 (previously track 5), the nearest flag is 35
frames away, and the log records the causing re-activation with IoU 0.30,
fused cost 0.31 and no appearance corroboration.

Every one of those is checkable. This script locates the switches on GT
object 0 by the same CLEAR matching used for the table, reports the nearest
flag to each, and prints the log record at the switch frame, for each run
that has both the tracker output and the event log.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flag_calibration import (SPATIAL_GATE, clear_match, find_switches,
                              load_flags, load_mot)

SEQ = "dancetrack0004"
CLAIM_FRAME, CLAIM_GT_ID, CLAIM_TO, CLAIM_FROM, CLAIM_GAP = 261, 0, 2, 5, 35

# Paths default to the layout of the reproduction bundle, which is the parent
# of the directory this script lives in. Override them if your copy sits
# elsewhere. The script locates the example from the artifacts; the frame and
# the track identifiers above are the claim being tested, not an input to the
# search.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.dirname(_HERE)
GT_ROOT = os.environ.get("DAAT_GT", os.path.join(_BUNDLE, "data", "dancetrack_gt"))
TRACK_ROOT = os.environ.get("DAAT_TRACKS",
                            os.path.join(_BUNDLE, "data", "dancetrack_tracks"))
LOG_ROOTS = [r for r in (
    os.environ.get("DAAT_LOGS",
                   os.path.join(_BUNDLE, "data", "logs", "dancetrack_val")),
) if r]
GT = os.path.join(GT_ROOT, SEQ, "gt", "gt.txt")


def main():
    for p, what in ((GT, "ground truth"), (TRACK_ROOT, "tracker output"),
                    (LOG_ROOTS[0], "event logs")):
        if not os.path.exists(p):
            sys.exit("%s not found at %s; set DAAT_GT, DAAT_TRACKS or DAAT_LOGS"
                     % (what, p))
    tracks = []
    for dirpath, _, files in os.walk(TRACK_ROOT):
        for f in files:
            if f.lower() == SEQ + ".txt":
                tracks.append(os.path.join(dirpath, f))
    logs = []
    for root in LOG_ROOTS:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().startswith(SEQ) and f.endswith("_ids_events.csv"):
                    logs.append(os.path.join(dirpath, f))

    def runkey(p):
        parts = Path(p).parts
        for i, x in enumerate(parts):
            if x.upper().startswith("TEST-"):
                return "/".join(parts[i:i + 2])
        return str(Path(p).parent)

    logmap = {}
    for p in logs:
        logmap.setdefault(runkey(p), p)

    gt = load_mot(GT, is_gt=True)
    print(f"{SEQ}: {len(gt):,} GT rows, GT ids present: "
          f"{sorted(gt.id.unique())[:12]}")
    print(f"manuscript claim: frame {CLAIM_FRAME}, GT id {CLAIM_GT_ID}, "
          f"track {CLAIM_FROM} -> {CLAIM_TO}, nearest flag {CLAIM_GAP} frames\n")

    any_match = False
    for tp in sorted(tracks):
        rk = runkey(tp)
        lp = logmap.get(rk)
        print("=" * 72)
        print(f"RUN {rk}")
        print(f"  trk {tp}")
        print(f"  log {lp or 'MISSING'}")
        trk = load_mot(tp, is_gt=False)
        m = clear_match(trk, gt, SPATIAL_GATE)
        sw = find_switches(m)
        if sw.empty:
            print("  no switches located")
            continue
        s0 = sw[sw.gt_id == CLAIM_GT_ID]
        print(f"  {len(sw)} switches total, {len(s0)} on GT id {CLAIM_GT_ID}")
        if s0.empty:
            continue
        fl = load_flags(lp) if lp else pd.DataFrame()
        for _, r in s0.iterrows():
            if fl.empty:
                near = float("nan")
            else:
                near = int(np.min(np.abs(fl.frame.to_numpy() - r.frame)))
            hit = (int(r.frame) == CLAIM_FRAME and int(r.from_trk) == CLAIM_FROM
                   and int(r.to_trk) == CLAIM_TO)
            if hit:
                any_match = True
            print(f"    frame {int(r.frame):>5}  track {int(r.from_trk)} -> "
                  f"{int(r.to_trk)}  gap {int(r.gap)}f  nearest flag "
                  f"{near}f{'   <== MATCHES CLAIM' if hit else ''}")
        # show the log around the claimed frame
        if lp:
            ev = pd.read_csv(lp)
            win = ev[(ev.frame >= CLAIM_FRAME - 3) & (ev.frame <= CLAIM_FRAME + 3)]
            if not win.empty:
                print(f"    log records, frames {CLAIM_FRAME-3}-{CLAIM_FRAME+3}:")
                for _, e in win.iterrows():
                    print(f"      f{int(e.frame)} {e.stage} {e.old_id}->{e.new_id} "
                          f"{str(e.reason)[:38]} iou={e.iou_cost} "
                          f"fused={e.fused_cost} cos={e.cosine_sim}")
    print()
    print("CLAIM REPRODUCES" if any_match else
          "CLAIM DOES NOT REPRODUCE in any surviving run")
    return 0 if any_match else 1


if __name__ == "__main__":
    raise SystemExit(main())

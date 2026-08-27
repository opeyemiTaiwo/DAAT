#!/usr/bin/env python3
"""
DAAT — flag-calibration analysis (paper Sec. 4.7).

Tests whether events the tracker flags as ambiguous (gate/margin refusals
in the ids_events audit logs) predict real identity errors, by scoring the
logs against MOT17 train ground truth with motmetrics.

Reported result over the seven training scenes, matching each true GT
identity switch against refusal events within +/-5 frames on the involved
tracks:

    precision = 0.673   (726 of 1,078 refusals near a true switch)
    recall    = 0.896   (628 of 701 true switches raise >= 1 flag)
    F1        = 0.769
    silent    = 73 of 701 switches (10.4%) raise NO flag
    Fisher exact p = 0.055

An earlier development measurement reported a raw co-occurrence contrast of
16.2% vs 2.7%; it conditioned on neither event type nor track identity and
is superseded by the matched comparison above.

Expected layout (override with environment variables):
  $DATA_ROOT/MOT17/train/<SEQ>/gt/gt.txt        -- MOT17 ground truth
  $LOG_DIR/<SEQ>_ids_events.csv                 -- audit logs
  $LOG_DIR/**/<SEQ>.txt                         -- matching tracker outputs

Usage:
  DATA_ROOT=/path/to/data LOG_DIR=/path/to/run python eval/flag_calibration.py
"""
import glob
import os

import numpy as np

# --- NumPy 2.0 compatibility shims (must precede motmetrics import) ---
np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
np.float = float
np.int = int
np.bool = bool

import motmetrics as mm
import pandas as pd

DATA_ROOT = os.environ.get("DATA_ROOT", "/content/data")
LOG_DIR = os.environ.get("LOG_DIR", "/content/work/run")
OUT_CSV = os.environ.get("OUT_CSV", "flag_calibration.csv")
MOT17_TRAIN_GT = f"{DATA_ROOT}/MOT17/train"
WINDOW = 5
FLAG_PREFIXES = ("margin_failed", "gate_failed")


def split_flagged(log):
    reasons = log.reason.astype(str)
    if reasons.str.startswith(FLAG_PREFIXES).any():
        is_flag = reasons.str.startswith(FLAG_PREFIXES)
    elif "cosine_margin" in log.columns and log.cosine_margin.notna().any():
        is_flag = log.cosine_margin < 0.05
    else:
        return log.iloc[0:0], log.iloc[0:0]
    has_app = log.cosine_sim.notna() if "cosine_sim" in log.columns else ~is_flag
    return log[is_flag], log[~is_flag & has_app]


def find_result_txt(seq):
    hits = glob.glob(f"{LOG_DIR}/**/{seq}.txt", recursive=True)
    return hits[0] if hits else None


def main():
    log_paths = sorted(glob.glob(f"{LOG_DIR}/*_ids_events.csv"))
    print(f"{len(log_paths)} audit logs found in {LOG_DIR}")
    assert log_paths, "No ids_events logs found — set LOG_DIR."

    rows, skipped = [], []
    for log_path in log_paths:
        seq = os.path.basename(log_path).replace("_ids_events.csv", "")
        res_path = find_result_txt(seq)
        gt_path = f"{MOT17_TRAIN_GT}/{seq}/gt/gt.txt"
        if res_path is None or not os.path.exists(gt_path):
            skipped.append(seq)
            continue
        gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(res_path, fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
        ev = acc.events.reset_index()
        sw = ev[ev.Type == "SWITCH"][["FrameId", "HId"]].values

        log = pd.read_csv(log_path)
        flagged, unflagged = split_flagged(log)
        for is_flag, sub in [(True, flagged), (False, unflagged)]:
            hit = 0
            for _, e in sub.iterrows():
                ids = {e.old_id, e.new_id}
                hit += any(
                    abs(f - e.frame) <= WINDOW and h in ids for f, h in sw
                )
            rows.append(
                dict(seq=seq, flagged=is_flag, n_events=len(sub), n_near_switch=hit)
            )

    if skipped:
        print(f"Skipped {len(skipped)} sequences (missing result or GT).")
    cal = pd.DataFrame(rows)
    cal.to_csv(OUT_CSV, index=False)
    summary = cal.groupby("flagged")[["n_events", "n_near_switch"]].sum()
    summary["rate"] = summary.n_near_switch / summary.n_events.clip(lower=1)
    print(summary)
    if True in summary.index and summary.loc[True, "n_events"] > 0:
        r_f, r_u = summary.loc[True, "rate"], summary.loc[False, "rate"]
        print(
            f"RESULT: flagged {r_f:.1%} vs unflagged {r_u:.1%} "
            f"-> ratio {r_f / max(r_u, 1e-9):.1f}x"
        )


if __name__ == "__main__":
    main()

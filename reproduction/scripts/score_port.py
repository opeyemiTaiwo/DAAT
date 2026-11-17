#!/usr/bin/env python3
"""
score_port.py -- score the instrumented ByteTrack and OC-SORT against ground
truth on the same public detections our own tracker consumed.

Reviewer 4, comment 1: "Instrument ByteTrack and OC-SORT with the audit layer
and report (i) logging on/off equivalence of outputs and (ii) their accuracy
under your detector and output policy. Without this, the paper's central
framing should be softened."

Part (i) was reported in the first revision. Part (ii) we declined, on the
grounds that porting our detector and output policy into their codebases
produces systems that are no longer ByteTrack and OC-SORT. That objection
stands for *our detector*, but it does not stand for the comparison the
comment is actually after, which is whether the accuracy gap is a property of
detection or of association. Holding the detections fixed and varying only the
tracker isolates exactly that, and MOT17 public detections give a common input
all three systems already accept.

So: all three run on MOT17 public detections; this scores them against the
same ground truth with the same evaluator.

Usage:
    python3 score_port.py --gt <dir of MOT17-XX-YYY.txt> \\
                          --port <bytetrack_ocsort_port dir> \\
                          [--out port_scores]
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np


def run_trackeval(gt_dir, seqs, tracker_dirs, out_dir):
    """Score each tracker with TrackEval's MOT17 configuration."""
    import trackeval

    root = tempfile.mkdtemp(prefix="te_")
    gt_root = os.path.join(root, "gt", "mot_challenge", "MOT17-port")
    seq_root = os.path.join(root, "gt", "mot_challenge", "seqmaps")
    tr_root = os.path.join(root, "trackers", "mot_challenge", "MOT17-port")
    os.makedirs(seq_root, exist_ok=True)

    for seq in seqs:
        d = os.path.join(gt_root, seq, "gt")
        os.makedirs(d, exist_ok=True)
        shutil.copy(gt_dir[seq], os.path.join(d, "gt.txt"))
        # TrackEval needs seqinfo for frame count; derive it from the GT.
        last = 0
        with open(gt_dir[seq]) as fh:
            for line in fh:
                if line.strip():
                    last = max(last, int(float(line.split(",")[0])))
        with open(os.path.join(gt_root, seq, "seqinfo.ini"), "w") as fh:
            fh.write("[Sequence]\nname=%s\nseqLength=%d\n" % (seq, last))

    with open(os.path.join(seq_root, "MOT17-port.txt"), "w") as fh:
        fh.write("name\n" + "\n".join(seqs) + "\n")

    for name, src in tracker_dirs.items():
        d = os.path.join(tr_root, name, "data")
        os.makedirs(d, exist_ok=True)
        for seq in seqs:
            s = os.path.join(src, seq + ".txt")
            if os.path.exists(s):
                shutil.copy(s, os.path.join(d, seq + ".txt"))

    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False,
                       PRINT_CONFIG=False, TIME_PROGRESS=False,
                       OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                       PLOT_CURVES=False, DISPLAY_LESS_PROGRESS=True))
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(dict(GT_FOLDER=gt_root, TRACKERS_FOLDER=tr_root,
                       BENCHMARK="MOT17", SPLIT_TO_EVAL="port",
                       SEQMAP_FILE=os.path.join(seq_root, "MOT17-port.txt"),
                       SKIP_SPLIT_FOL=True, DO_PREPROC=False,
                       TRACKERS_TO_EVAL=list(tracker_dirs),
                       SEQ_INFO={s: None for s in seqs}))

    ev = trackeval.Evaluator(ev_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    metrics = [trackeval.metrics.HOTA({"THRESHOLD": 0.5}),
               trackeval.metrics.CLEAR({"THRESHOLD": 0.5}),
               trackeval.metrics.Identity({"THRESHOLD": 0.5})]
    res, _ = ev.evaluate([dataset], metrics)
    shutil.rmtree(root, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--out", default="port_scores")
    a = ap.parse_args()

    # Only sequences the port actually produced AND we have GT for.
    bt = os.path.join(a.port, "bytetrack", "audit_on")
    oc = os.path.join(a.port, "ocsort", "audit_on")
    seqs = sorted({f[:-4] for f in os.listdir(bt) if f.endswith(".txt")
                   and not f.endswith("_assoc_events.csv")}
                  & {f[:-4] for f in os.listdir(oc) if f.endswith(".txt")})
    gt = {}
    for s in seqs:
        p = os.path.join(a.gt, s + ".txt")
        if os.path.exists(p):
            gt[s] = p
    seqs = [s for s in seqs if s in gt]
    if not seqs:
        sys.exit("no sequence has both a port output and ground truth")

    print("scoring %d sequence(s): %s" % (len(seqs), ", ".join(seqs)))
    print("all three systems consume the same MOT17 public detections\n")

    res = run_trackeval(gt, seqs, {"ByteTrack": bt, "OC-SORT": oc}, a.out)
    key = list(res["MotChallenge2DBox"].keys())[0] if "MotChallenge2DBox" in res else None
    block = res["MotChallenge2DBox"] if key is None else res["MotChallenge2DBox"]

    os.makedirs(a.out, exist_ok=True)
    lines = []
    hdr = ("  %-12s %-18s %7s %7s %7s %7s %7s %6s"
           % ("Tracker", "Sequence", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"))
    lines.append(hdr)
    lines.append("  " + "-" * 76)

    for tracker in ("ByteTrack", "OC-SORT"):
        if tracker not in block:
            continue
        for seq in seqs + ["COMBINED_SEQ"]:
            if seq not in block[tracker]:
                continue
            m = block[tracker][seq]["pedestrian"]
            h, c, i = m["HOTA"], m["CLEAR"], m["Identity"]
            lines.append("  %-12s %-18s %7.2f %7.2f %7.2f %7.2f %7.2f %6d"
                         % (tracker, seq if seq != "COMBINED_SEQ" else "(combined)",
                            100 * np.mean(h["HOTA"]), 100 * np.mean(h["DetA"]),
                            100 * np.mean(h["AssA"]), c["MOTA"] * 100,
                            i["IDF1"] * 100, int(c["IDSW"])))

    out = "\n".join(lines)
    print(out)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write("Scored on MOT17 public detections, identical input to all systems.\n\n")
        fh.write(out + "\n")
    print("\nwrote %s/report.txt" % a.out)


if __name__ == "__main__":
    main()

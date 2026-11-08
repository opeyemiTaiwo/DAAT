#!/usr/bin/env python3
"""
match_dance_runs.py -- find which DanceTrack run folder gives the largest
three-way overlap between:

  (a) tracker output .txt files recovered from the author's machine
  (b) per-frame audit event logs held in logs/ and sim/
  (c) the 15 DanceTrack validation ground-truth files from dtval.zip

Table 13's DanceTrack half currently rests on a temporal-only matching
criterion that the MOT17 half no longer uses. To recompute it under the same
IoU 0.3 spatial gate I need, for one single run, the tracker output and the
event log for the same sequences, and ground truth to score against. No run
is complete, so the job is to pick the best available subset and then state
its exact size rather than implying a full split.

Usage:
    python3 match_dance_runs.py --tracks <dir of extracted dance_tracks.zip>
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# Paths default to the reproduction bundle's own layout, which is the parent
# of the directory holding this script. Override on the command line, or with
# the DAAT_GT, DAAT_TRACKS and DAAT_LOGS environment variables, if your copy
# lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = os.path.dirname(_HERE)
_DATA = os.path.join(_BUNDLE, "data")
GT_ROOT = os.environ.get("DAAT_GT", os.path.join(_DATA, "dancetrack_gt"))
LOG_ROOTS = [os.environ.get("DAAT_LOGS", os.path.join(_DATA, "logs", "dancetrack_val"))]

SEQ_RE = re.compile(r"(dancetrack\d{4})", re.I)


def seq_of(path):
    m = SEQ_RE.search(os.path.basename(path))
    return m.group(1).lower() if m else None


def run_key(path, root):
    """Everything between the archive root and the filename identifies the run."""
    rel = os.path.relpath(path, root)
    return os.path.dirname(rel)


def gt_sequences():
    if not os.path.isdir(GT_ROOT):
        sys.exit("ground truth not found at %s" % GT_ROOT)
    out = set()
    for d in sorted(os.listdir(GT_ROOT)):
        if os.path.exists(os.path.join(GT_ROOT, d, "gt", "gt.txt")):
            s = seq_of(d + ".txt")
            if s:
                out.add(s)
    return out


def log_runs():
    """run suffix (…/Test-XX-name/variant/results) -> set of sequences."""
    runs = defaultdict(set)
    for root in LOG_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if "DANCE" not in dirpath.upper():
                continue
            seqs = {seq_of(f) for f in files if f.lower().endswith(".csv")}
            seqs.discard(None)
            if not seqs:
                continue
            # normalize: drop the logs/ vs sim/ prefix so both collapse together
            rel = os.path.relpath(dirpath, root)
            runs[rel] |= seqs
    return runs


def track_runs(track_root):
    runs = defaultdict(set)
    for dirpath, _, files in os.walk(track_root):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            s = seq_of(f)
            if not s:
                continue
            runs[run_key(os.path.join(dirpath, f), track_root)].add(s)
    return runs


def normalize_run(key):
    """Strip any leading MOT-ALL/DANCE-TRACKS so both sides compare equal."""
    parts = [p for p in key.split(os.sep) if p]
    for i, p in enumerate(parts):
        if p.upper().startswith("TEST-") or p.upper().startswith("DANCETRACK-") \
           or p.upper() == "TEST-SUBMIT":
            return os.sep.join(parts[i:])
    return os.sep.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True,
                    help="directory holding the extracted dance_tracks.zip")
    a = ap.parse_args()

    gt = gt_sequences()
    logs = {normalize_run(k): v for k, v in log_runs().items()}
    tracks = {normalize_run(k): v for k, v in track_runs(a.tracks).items()}

    print("ground truth available : %d sequences" % len(gt))
    print("  %s" % ", ".join(sorted(gt)))
    print("track-file run folders : %d" % len(tracks))
    print("event-log run folders  : %d" % len(logs))
    print()

    rows = []
    for run, tseqs in tracks.items():
        lseqs = logs.get(run, set())
        three = tseqs & lseqs & gt
        rows.append((len(three), len(tseqs & gt), len(tseqs), run,
                     sorted(three), run in logs))

    rows.sort(key=lambda r: (-r[0], -r[1], -r[2]))

    print("=" * 78)
    print("%-4s %-4s %-4s %-6s %s" % ("3way", "t&gt", "trk", "haslog", "run"))
    print("=" * 78)
    for three, tg, nt, run, seqs, haslog in rows:
        print("%-4d %-4d %-4d %-6s %s" % (three, tg, nt, "yes" if haslog else "NO", run))

    print()
    best = rows[0] if rows else None
    if best and best[0] > 0:
        print("Best three-way run: %s" % best[3])
        print("  usable sequences (%d): %s" % (best[0], ", ".join(best[4])))
    else:
        print("No run has tracker output, an event log, and ground truth for the")
        print("same sequence. Fall back to the best track-and-ground-truth overlap:")
        fb = sorted(rows, key=lambda r: -r[1])
        for three, tg, nt, run, seqs, haslog in fb[:5]:
            print("  %-2d  %s  (event log: %s)" % (tg, run, "yes" if haslog else "no"))

    # Union view: how many distinct validation sequences are covered at all.
    covered = set()
    for _, _, _, _, _, _ in rows:
        pass
    for tseqs in tracks.values():
        covered |= (tseqs & gt)
    print()
    print("Across all run folders, %d of the %d ground-truth sequences have"
          % (len(covered), len(gt)))
    print("tracker output somewhere: %s" % ", ".join(sorted(covered)))
    missing = sorted(gt - covered)
    if missing:
        print("No tracker output anywhere for: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()

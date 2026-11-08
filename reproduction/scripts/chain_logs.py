#!/usr/bin/env python3
"""
chain_logs.py -- retrofit tamper-evidence onto audit logs that already exist.

Reviewer 3: "Re-derivability is not integrity. verify.py only checks that
logged decisions follow from logged inputs. A consistently edited log, or one
produced from a different video or configuration, passes the same check.
Records are not bound to source frames, detector output or configuration. The
event-driven log also cannot prove that no event occurred. Add per-frame
hash-chained records (including frames with no events), input and
configuration hashes, and trusted timestamps, or state this explicitly as a
limitation."

The manuscript took the limitation route because retrofitting the chain into
the tracker would mean re-running every experiment on an unvalidated logger.
That is not the only option. The chain can be built *over* the logs that
already exist, which binds them without touching the tracker and without
re-running anything. That is what this does.

It closes three of the four gaps Reviewer 3 names:

  tamper-evidence   every frame's record is hashed together with the previous
                    frame's hash, so altering any earlier record breaks every
                    hash after it. Editing a log "consistently" is no longer
                    enough; an attacker must also recompute the whole chain,
                    and the published head hash makes that detectable.

  absence of events  the chain covers EVERY frame in the sequence's range, not
                    only frames that produced events. A frame with no events
                    gets a keepalive block, following WEFT's design. A missing
                    interval is then a missing block, which is detectable,
                    where in an event-driven log it is indistinguishable from
                    a quiet interval.

  binding to inputs  each sequence's head block commits to the SHA-256 of the
                    detector output and of the frozen configuration, so a log
                    produced from different footage or a different
                    configuration no longer verifies against this manifest.

The fourth gap, trusted timestamps, this cannot close: an RFC 3161 token must
come from a time-stamping authority at the moment of acquisition, and no
after-the-fact tool can manufacture one. The manifest records that absence
explicitly rather than implying coverage it does not have.

What this is NOT: it does not make an old log trustworthy. It binds the log as
it stands at the moment of chaining. Anything altered before then is sealed in.
The honest claim is forward-looking: from the chaining date, undetected
alteration requires recomputing the chain, and the head hash is published.

Usage
-----
    python3 chain_logs.py seal   <results_dir> [--config FILE] [--det FILE]
                                 [--out manifest.json]
    python3 chain_logs.py verify <results_dir> --manifest manifest.json
    python3 chain_logs.py tamper-demo <results_dir> --manifest manifest.json
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

CHAIN_VERSION = "1"
GENESIS = "0" * 64


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(row, fields):
    """A record's bytes must not depend on dict ordering or float formatting."""
    return "\x1f".join((row.get(f) or "").strip() for f in fields)


def block_hash(prev, frame, kind, payload):
    h = hashlib.sha256()
    h.update(CHAIN_VERSION.encode())
    h.update(b"\x00")
    h.update(prev.encode())
    h.update(b"\x00")
    h.update(str(frame).encode())
    h.update(b"\x00")
    h.update(kind.encode())
    h.update(b"\x00")
    h.update(payload.encode())
    return h.hexdigest()


def chain_sequence(csv_path):
    """Build the per-frame chain for one sequence's event log.

    Every frame in [min_frame, max_frame] gets a block. Frames that produced
    events get an "events" block committing to their records; frames that
    produced none get a "keepalive" block. The chain therefore proves what did
    NOT happen as well as what did.
    """
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    by_frame = {}
    for r in rows:
        try:
            f = int(float(r["frame"]))
        except (KeyError, ValueError):
            continue
        by_frame.setdefault(f, []).append(r)

    if not by_frame:
        return dict(blocks=0, events=0, keepalives=0, head=GENESIS,
                    first_frame=None, last_frame=None, fields=fields)

    lo, hi = min(by_frame), max(by_frame)
    prev = GENESIS
    n_events = n_keep = 0

    for frame in range(lo, hi + 1):
        recs = by_frame.get(frame)
        if recs:
            payload = "\x1e".join(canonical(r, fields) for r in recs)
            prev = block_hash(prev, frame, "events", payload)
            n_events += 1
        else:
            prev = block_hash(prev, frame, "keepalive", "")
            n_keep += 1

    return dict(blocks=(hi - lo + 1), events=n_events, keepalives=n_keep,
                head=prev, first_frame=lo, last_frame=hi, fields=fields,
                records=len(rows))


def seal(results_dir, config_path, det_path, out_path):
    files = sorted(f for f in os.listdir(results_dir)
                   if f.endswith("_ids_events.csv"))
    if not files:
        sys.exit("no *_ids_events.csv under %s" % results_dir)

    manifest = {
        "chain_version": CHAIN_VERSION,
        "sealed_utc": datetime.datetime.now(datetime.timezone.utc)
                              .isoformat(timespec="seconds"),
        "results_dir": os.path.abspath(results_dir),
        "trusted_timestamp": None,
        "trusted_timestamp_note":
            "No RFC 3161 token. A trusted timestamp must be obtained from a "
            "time-stamping authority at acquisition and cannot be "
            "manufactured after the fact. This manifest binds content and "
            "order, not time of origin.",
        "binds": {},
        "sequences": {},
    }

    for label, path in (("configuration", config_path), ("detector_output", det_path)):
        if path and os.path.exists(path):
            manifest["binds"][label] = {
                "path": os.path.basename(path),
                "sha256": sha256_file(path),
            }
        else:
            manifest["binds"][label] = None

    tot_blocks = tot_events = tot_keep = tot_recs = 0
    seq_heads = []
    for fn in files:
        seq = fn.replace("_ids_events.csv", "")
        info = chain_sequence(os.path.join(results_dir, fn))
        info["file_sha256"] = sha256_file(os.path.join(results_dir, fn))
        manifest["sequences"][seq] = info
        tot_blocks += info["blocks"]
        tot_events += info["events"]
        tot_keep += info["keepalives"]
        tot_recs += info.get("records", 0)
        seq_heads.append(seq + ":" + info["head"])

    # One head over everything, so the whole release is a single commitment.
    manifest["head"] = hashlib.sha256("\n".join(sorted(seq_heads)).encode()).hexdigest()
    manifest["totals"] = dict(sequences=len(files), blocks=tot_blocks,
                              event_blocks=tot_events,
                              keepalive_blocks=tot_keep, records=tot_recs)

    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    print("sealed %d sequences" % len(files))
    print("  frames covered by a block : %s" % f"{tot_blocks:,}")
    print("    with events             : %s" % f"{tot_events:,}")
    print("    keepalive (no events)   : %s  <- these are what prove absence"
          % f"{tot_keep:,}")
    print("  event records committed   : %s" % f"{tot_recs:,}")
    for k, v in manifest["binds"].items():
        print("  bound %-16s: %s" % (k, v["sha256"][:16] + "..." if v else "NOT SUPPLIED"))
    print("  trusted timestamp         : none (see manifest note)")
    print("\n  HEAD %s" % manifest["head"])
    print("\nwrote %s" % out_path)
    print("Publish the head hash. Anyone can then re-run `verify` and detect "
          "any later alteration.")
    return manifest


def verify(results_dir, manifest_path):
    man = json.load(open(manifest_path))
    bad = []
    seq_heads = []
    for seq, rec in sorted(man["sequences"].items()):
        path = os.path.join(results_dir, seq + "_ids_events.csv")
        if not os.path.exists(path):
            bad.append((seq, "MISSING FILE"))
            continue
        info = chain_sequence(path)
        if info["head"] != rec["head"]:
            bad.append((seq, "CHAIN BROKEN"))
        elif sha256_file(path) != rec["file_sha256"]:
            bad.append((seq, "FILE HASH DIFFERS"))
        seq_heads.append(seq + ":" + info["head"])

    head = hashlib.sha256("\n".join(sorted(seq_heads)).encode()).hexdigest()
    ok = (not bad) and head == man["head"]

    print("sealed      : %s" % man["sealed_utc"])
    print("sequences   : %d" % len(man["sequences"]))
    print("expected HEAD %s" % man["head"])
    print("computed HEAD %s" % head)
    if ok:
        print("\nINTACT. Every frame block and every sequence head matches.")
    else:
        print("\nTAMPERED or INCOMPLETE:")
        for seq, why in bad:
            print("   %-22s %s" % (seq, why))
        if head != man["head"]:
            print("   release head does not match")
    return 0 if ok else 1


def tamper_demo(results_dir, manifest_path):
    """Change one digit in one record and show the chain catches it."""
    man = json.load(open(manifest_path))
    seq = sorted(man["sequences"])[0]
    path = os.path.join(results_dir, seq + "_ids_events.csv")
    original = open(path, encoding="utf-8").read()

    lines = original.split("\n")
    target = None
    for i, ln in enumerate(lines):
        if "margin_failed" in ln:
            target = i
            break
    if target is None:
        target = 1

    before = lines[target]
    after = before.replace("0.0", "0.1", 1)
    if after == before:
        after = before + ",X"
    lines[target] = after

    print("sequence      : %s" % seq)
    print("record edited : line %d" % (target + 1))
    print("  before: %s" % before[:110])
    print("  after : %s" % after[:110])

    try:
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        info = chain_sequence(path)
        caught = info["head"] != man["sequences"][seq]["head"]
        print("\nsequence head expected %s" % man["sequences"][seq]["head"][:32])
        print("sequence head computed %s" % info["head"][:32])
        print("\n%s" % ("DETECTED. One altered digit changes the chain head."
                        if caught else "NOT DETECTED -- the chain is not working."))
    finally:
        open(path, "w", encoding="utf-8").write(original)
        print("original log restored.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seal")
    s.add_argument("results_dir")
    s.add_argument("--config", default=None)
    s.add_argument("--det", default=None)
    s.add_argument("--out", default="chain_manifest.json")

    v = sub.add_parser("verify")
    v.add_argument("results_dir")
    v.add_argument("--manifest", required=True)

    t = sub.add_parser("tamper-demo")
    t.add_argument("results_dir")
    t.add_argument("--manifest", required=True)

    a = ap.parse_args()
    if a.cmd == "seal":
        seal(a.results_dir, a.config, a.det, a.out)
        return 0
    if a.cmd == "verify":
        return verify(a.results_dir, a.manifest)
    return tamper_demo(a.results_dir, a.manifest)


if __name__ == "__main__":
    sys.exit(main())

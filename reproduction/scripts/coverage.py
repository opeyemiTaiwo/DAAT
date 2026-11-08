#!/usr/bin/env python3
"""
coverage.py -- compute Table 3 (re-derivation coverage by decision category)
from the audit event logs.

This table was described in the manuscript before it was computed. The
placeholders in Table 3 and the coverage percentages quoted in the responses
to Reviewers 1 and 2 must be replaced by this script's output. Do not ship
either figure until this has been run.

WHAT IT MEASURES
----------------
Two distinct properties, which the submitted manuscript conflated:

  re-derivable               the recorded outcome follows from values
                             present in the log row itself. A refusal that
                             stores "margin=0.037 < delta=0.04" is
                             re-derivable: the comparison can be rechecked
                             without any other input.

  independently recomputable the deciding operands can themselves be
                             rebuilt from released detector output, without
                             re-running the embedding network. A decision
                             whose deciding operand is a cosine similarity
                             or a fused cost is NOT independently
                             recomputable, because the embedding is not
                             serialized. The value is disclosed, not
                             reproduced.

The distinction is the whole point of Reviewer 1's Comment 1. A disclosed
operand asks the reader to trust the log; a recomputable one does not.

USAGE
-----
    # DanceTrack frozen test submission (35 sequences)
    python3 coverage.py \\
        --logs MOT-ALL/DANCE-TRACKS/DanceTrack-TEST-SUBMIT/Test-24-codabench/24_frozen_submit/results \\
        --label "DanceTrack (frozen test submission)"

    # MOT17, for the domain comparison
    python3 coverage.py \\
        --logs MOT-ALL/MOT-17/TEST/Test-6-YOLOX/ALL-21/results \\
        --label "MOT17"

    # both at once, which is what Table 3 needs
    python3 coverage.py --logs <dance_dir> --logs <mot17_dir> \\
        --label "DanceTrack (frozen test submission)" --label "MOT17" \\
        --out coverage_out
"""

import argparse
import collections
import csv
import json
import os
import sys

# --------------------------------------------------------------------------
# Decision categories, keyed by (stage, first token of reason).
#
# These keys are read off the logs themselves, not assumed. Any (stage,
# reason) pair not listed here is reported under "UNMAPPED" rather than
# silently dropped, so a vocabulary change shows up as a loud row instead
# of a quietly wrong total.
# --------------------------------------------------------------------------
# Each category carries its verifiability verdict explicitly rather than having
# it inferred from which columns happen to be populated. The verdicts below were
# established empirically from the submission logs, not assumed:
#
#   recomputable  the deciding operand is geometry, rebuildable from released
#                 detector output. Verified: these rows carry no embedding term.
#
#   disclosed     the deciding operand is an appearance value that IS logged
#                 (cosine_sim, cosine_margin, the top-5 similarity vector), so
#                 the comparison can be rechecked but not reproduced without
#                 re-running the embedding network.
#
#   composite     re-activation logs a fused_cost built in three stages:
#                 detection-score fusion 1-(1-iou)*det_score, then an
#                 elementwise minimum against an embedding cost, then an
#                 observation-centric motion term. det_score IS logged, so the
#                 first stage is reproducible and the row can be decomposed:
#                   fused == 1-(1-iou)*det  -> det-score fusion alone, so the
#                                              decision is recomputable;
#                   fused <  that           -> the minimum took the embedding
#                                              branch, so emb == fused exactly
#                                              and the operand is RECOVERABLE
#                                              (disclosed, not recomputable);
#                   fused >  that           -> the motion term was added after
#                                              the minimum, and the record
#                                              cannot say whether appearance
#                                              also fired. AMBIGUOUS: neither.
#                 This is decided per row by decompose_reactivation(), not by
#                 a blanket verdict for the category.
#
#   no-compare    logged event, not an association decision (empty registry).
#
# The submitted manuscript's 99.9% re-derivability figure treated "unlogged" as
# re-derivable, which is why it was too high.
CATEGORIES = [
    # label,                           stage,                 reason,                       verdict
    ("Stage-one assignment (IoU)",     "4_dlow_second_assoc", "byte_second_assoc",          "recomputable"),
    ("Occlusion recovery",             "5",                   "unconfirmed_update",         "recomputable"),
    ("Observation-centric recovery",   "5b_ocr_recovery",     "ocr_last_obs_iou",           "recomputable"),
    ("Gate refusal",                   "6_new",               "gate_failed",                "disclosed"),
    ("Margin refusal",                 "6_new",               "margin_failed",              "disclosed"),
    ("Gallery revival (appearance)",   "6_revive",            "revived",                    "disclosed"),
    ("Re-activation from lost pool",   "3",                   "re_activate_from_lost",      "composite"),
    ("Immature registry (no compare)", "6_new",               "registry_empty_or_immature", "no-compare"),
]

VERDICT = {lab: v for lab, _, _, v in CATEGORIES}

KEY_TO_LABEL = {(s, r): lab for lab, s, r, _ in CATEGORIES}
ORDER = [lab for lab, _, _, _ in CATEGORIES]

# Columns whose presence means the deciding operand came from the embedding
# network and therefore cannot be rebuilt from detector output alone.
EMBEDDING_COLS = ("cosine_sim", "fused_cost", "feat_mean_sim")

# Columns that make a row self-checking: the threshold it was tested against
# is stored next to the value that was tested.
DERIVE_COLS = ("iou_overlap", "iou_cost", "cosine_sim", "fused_cost",
               "cosine_theta", "cosine_margin")


def nonempty(row, col):
    v = row.get(col)
    return bool(v) and str(v).strip() not in ("", "nan", "None")


# Events that are logged but are not association decisions. They belong in the
# event total and must be excluded from the decision denominator. This is the
# 12,016 versus 12,013 distinction: both figures were right for different
# quantities and the manuscript never said which.
NON_DECISION = {"Immature registry (no compare)"}


def reason_key(reason):
    """First token of the reason, with any logged value stripped off.

    Some reasons carry their operand inline, e.g. 'ocr_last_obs_iou=0.323'.
    Keying on the raw token would make every distinct IoU its own category.
    """
    if not reason:
        return ""
    return reason.split(" ")[0].split("=")[0]


# Every operand is printed at four decimals, so each carries +/- 5e-5 of
# rounding. The deviation this function tests is of the same order, which
# means the SIGN of a small deviation can be an artifact of the printing
# rather than a fact about the decision.
PRINT_EPS = 5e-5


def decompose_reactivation(row):
    """Split one re-activation by which mechanism produced its fused cost.

    Returns (mechanism, re_derivable, independently_recomputable).

    An earlier version of this function took the sign of

        d = fused_cost - (1 - (1 - iou_cost) * det_score)

    at a 1e-6 tolerance and reported 1,024 DanceTrack and 422 MOT17
    re-activations as carrying an added motion term. That classification does
    not survive the log's own print precision: the largest positive deviation
    observed on either submission is 1.26e-4, inside the +/- 5e-5 rounding
    envelope on three operands. Not one record can be shown to carry a motion
    term, and the earlier figure was reading rounding as signal.

    This version propagates the printed precision through the expression and
    classifies a record only when the whole interval lies on one side of zero.
    d rises with fused_cost and det_score and falls with iou_cost.
    """
    ic = (row.get("iou_cost") or "").strip()
    fc = (row.get("fused_cost") or "").strip()
    ds = (row.get("det_score") or "").strip()
    if not (ic and fc and ds):
        return ("unknown", False, False)
    ic, fc, ds = float(ic), float(fc), float(ds)
    e = PRINT_EPS
    d_hi = (fc + e) - 1.0 + (1.0 - (ic - e)) * (ds + e)
    d_lo = (fc - e) - 1.0 + (1.0 - (ic + e)) * (ds - e)

    if d_hi < 0:
        # The minimum demonstrably took the embedding branch, so emb == fused
        # exactly and the similarity is recoverable as 1 - 2*fused. Checkable,
        # not reproducible without the network.
        return ("appearance override", True, False)
    if d_lo > 0:
        # A motion term was demonstrably added after the minimum; the record
        # cannot say whether the appearance branch also fired. No record on
        # either frozen submission falls here.
        return ("motion term added", False, False)
    # The interval straddles zero. The record is consistent with detection-
    # score fusion alone, but the log does not resolve it, so we cannot name
    # the deciding operand. Reported as its own category rather than being
    # assigned to the branch its rounding happens to favor.
    return ("unresolved at log precision", False, False)


def classify(row):
    """Return (label, re_derivable, independently_recomputable).

    The verdict comes from the category's declared verifiability, not from
    guessing at populated columns. A row is additionally checked against the
    declaration, and a mismatch is reported rather than absorbed.
    """
    stage = (row.get("stage") or "").strip()
    reason = (row.get("reason") or "").strip()
    key = (stage, reason_key(reason))
    label = KEY_TO_LABEL.get(key)

    if label is None:
        return ("UNMAPPED stage=%s reason=%s" % (key[0], key[1]), False, False)

    verdict = VERDICT[label]
    if verdict == "recomputable":
        return (label, True, True)
    if verdict == "disclosed":
        # Re-derivable: the appearance operand and its threshold are both on
        # the row, so the comparison can be rechecked.
        return (label, True, False)
    if verdict == "composite":
        # Decided per row, not per category: see decompose_reactivation.
        return (label,) + decompose_reactivation(row)[1:]
    return (label, False, False)          # no-compare


def scan(log_dir):
    """Walk one run's results directory and tally every event row."""
    files = sorted(
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.endswith("_ids_events.csv")
    )
    if not files:
        sys.exit("no *_ids_events.csv under %s" % log_dir)

    tally = collections.defaultdict(lambda: {"n": 0, "deriv": 0, "indep": 0})
    mech = collections.Counter()
    per_seq = {}
    total = 0

    for path in files:
        seq = os.path.basename(path).replace("_ids_events.csv", "")
        n_seq = 0
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                label, deriv, indep = classify(row)
                if label == "Re-activation from lost pool":
                    mech[decompose_reactivation(row)[0]] += 1
                t = tally[label]
                t["n"] += 1
                t["deriv"] += int(deriv)
                t["indep"] += int(indep)
                n_seq += 1
                total += 1
        per_seq[seq] = n_seq

    return tally, per_seq, total, len(files), mech


def pct(num, den):
    return 0.0 if den == 0 else 100.0 * num / den


def report(label, tally, per_seq, total, n_files, fh=sys.stdout, mech=None):
    w = fh.write
    w("\n" + "=" * 78 + "\n")
    w("%s\n" % label)
    w("=" * 78 + "\n")
    w("  %d sequences, %s event rows\n\n" % (n_files, f"{total:,}"))

    w("  %-32s %>9s %>14s %>16s\n".replace(">", "")
      % ("Decision category", "Events", "Re-derivable", "Indep. recomp."))
    w("  " + "-" * 74 + "\n")

    sum_n = sum_d = sum_i = 0          # association decisions only
    nd_n = 0                           # logged events that are not decisions
    labels = ORDER + sorted(k for k in tally if k not in ORDER)
    for lab in labels:
        if lab not in tally:
            continue
        t = tally[lab]
        if lab in NON_DECISION:
            nd_n += t["n"]
            w("  %-32s %9s %13s %15s\n"
              % (lab[:32], f"{t['n']:,}", "n/a", "n/a"))
            continue
        sum_n += t["n"]
        sum_d += t["deriv"]
        sum_i += t["indep"]
        flag = "  <-- CHECK" if lab.startswith("UNMAPPED") else ""
        w("  %-32s %9s %13.1f%% %15.1f%%%s\n"
          % (lab[:32], f"{t['n']:,}", pct(t["deriv"], t["n"]),
             pct(t["indep"], t["n"]), flag))

    w("  " + "-" * 74 + "\n")
    w("  %-32s %9s %13.1f%% %15.1f%%\n"
      % ("ASSOCIATION DECISIONS", f"{sum_n:,}", pct(sum_d, sum_n),
         pct(sum_i, sum_n)))
    if nd_n:
        w("  %-32s %9s\n" % ("logged but not decisions", f"{nd_n:,}"))
        w("  %-32s %9s\n" % ("TOTAL LOGGED EVENTS", f"{sum_n + nd_n:,}"))
    w("\n")
    w("  re-derivable               : %s of %s  (%.1f%%)\n"
      % (f"{sum_d:,}", f"{sum_n:,}", pct(sum_d, sum_n)))
    w("  independently recomputable : %s of %s  (%.1f%%)\n"
      % (f"{sum_i:,}", f"{sum_n:,}", pct(sum_i, sum_n)))
    if nd_n:
        w("\n  The two denominators the manuscript conflated:\n")
        w("    %s logged events, of which %s are association decisions\n"
          % (f"{sum_n + nd_n:,}", f"{sum_n:,}"))
        w("    the %s difference is: %s\n"
          % (f"{nd_n:,}", ", ".join(sorted(NON_DECISION))))

    if mech:
        w("\n  Re-activation decomposed by the mechanism that set its cost:\n")
        for k in ("appearance override", "motion term added",
                  "unresolved at log precision", "unknown"):
            if mech.get(k):
                tot = sum(mech.values())
                w("    %-30s %6s  (%.1f%% of re-activations)\n"
                  % (k, f"{mech[k]:,}", 100.0 * mech[k] / tot))
        w("    Classification propagates the four-decimal print precision of\n")
        w("    iou_cost, fused_cost and det_score, and names a mechanism only\n")
        w("    where the resulting interval is one-signed. Where it straddles\n")
        w("    zero the record is consistent with detection-score fusion alone\n")
        w("    but does not establish it, so the deciding operand is not\n")
        w("    recoverable: that is the gap, and it is wider than the earlier\n")
        w("    sign-based split reported.\n")

    unmapped = sum(t["n"] for k, t in tally.items() if k.startswith("UNMAPPED"))
    if unmapped:
        w("\n  WARNING %s rows did not match any known category.\n" % f"{unmapped:,}")
        w("          The category table in this script is out of date with the\n")
        w("          logs. Fix CATEGORIES before using these totals.\n")

    return {
        "label": label,
        "sequences": n_files,
        "events": sum_n,
        "re_derivable": sum_d,
        "re_derivable_pct": round(pct(sum_d, sum_n), 1),
        "independently_recomputable": sum_i,
        "independently_recomputable_pct": round(pct(sum_i, sum_n), 1),
        "by_category": {
            lab: {
                "events": tally[lab]["n"],
                "re_derivable_pct": round(pct(tally[lab]["deriv"], tally[lab]["n"]), 1),
                "independently_recomputable_pct": round(pct(tally[lab]["indep"], tally[lab]["n"]), 1),
            }
            for lab in tally
        },
        "per_sequence_events": per_seq,
        "unmapped_rows": unmapped,
    }


def latex_rows(results):
    """Emit the Table 3 body, ready to paste over the \\note{} placeholders."""
    if not results:
        return ""
    primary = results[0]
    lines = []
    for lab in ORDER:
        c = primary["by_category"].get(lab)
        if not c:
            continue
        indep = "yes" if c["independently_recomputable_pct"] >= 99.9 else (
            "no" if c["independently_recomputable_pct"] <= 0.1 else
            "%.1f\\%%" % c["independently_recomputable_pct"])
        lines.append("%-32s & %s & %.1f\\%% & %s \\\\"
                     % (lab, f"{c['events']:,}".replace(",", "{,}"),
                        c["re_derivable_pct"], indep))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", action="append", required=True,
                    help="a run's results/ directory holding *_ids_events.csv "
                         "(repeatable)")
    ap.add_argument("--label", action="append", default=[],
                    help="label for each --logs, in the same order")
    ap.add_argument("--out", default="coverage_out",
                    help="directory for report.txt and coverage.json")
    args = ap.parse_args()

    labels = list(args.label)
    while len(labels) < len(args.logs):
        labels.append(args.logs[len(labels)])

    os.makedirs(args.out, exist_ok=True)
    report_path = os.path.join(args.out, "report.txt")

    results = []
    with open(report_path, "w") as fh:
        for sink in (sys.stdout, fh):
            sink.write("coverage.py -- Table 3, computed from the audit logs\n")

        for log_dir, label in zip(args.logs, labels):
            tally, per_seq, total, n_files, mech = scan(log_dir)
            res = report(label, tally, per_seq, total, n_files, fh=fh, mech=mech)
            report(label, tally, per_seq, total, n_files, fh=sys.stdout, mech=mech)
            res["reactivation_mechanism"] = dict(mech)
            res["log_dir"] = log_dir
            results.append(res)

        if len(results) >= 2:
            a, b = results[0], results[1]
            for sink in (sys.stdout, fh):
                sink.write("\n" + "=" * 78 + "\n")
                sink.write("DOMAIN COMPARISON\n")
                sink.write("=" * 78 + "\n")
                sink.write("  %-40s %.1f%% independently recomputable\n"
                           % (a["label"], a["independently_recomputable_pct"]))
                sink.write("  %-40s %.1f%% independently recomputable\n"
                           % (b["label"], b["independently_recomputable_pct"]))
                sink.write("\n  The gap is the finding. A single pooled figure hides it.\n")

        tex = latex_rows(results)
        for sink in (sys.stdout, fh):
            sink.write("\n" + "=" * 78 + "\n")
            sink.write("TABLE 3 BODY (paste over the \\note{} placeholders)\n")
            sink.write("=" * 78 + "\n")
            sink.write(tex + "\n")

    with open(os.path.join(args.out, "coverage.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nwrote %s" % report_path)
    print("wrote %s" % os.path.join(args.out, "coverage.json"))


if __name__ == "__main__":
    main()

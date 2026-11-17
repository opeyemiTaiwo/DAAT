#!/usr/bin/env python3
"""
reactivation_precision.py -- is the re-activation cost decomposition of
Table 3 resolvable at the precision the log is printed to?

Table 3 splits `re_activate_from_lost` records three ways by the sign of

    d = fused_cost - (1 - (1 - iou_cost) * det_score)

where the subtrahend is detection-score fusion alone (Sec. 4.7, step c1).
Because the composite is built as c1, then an elementwise minimum against an
appearance cost, then a non-negative motion term:

    d < 0  =>  the appearance minimum selected c_emb
    d = 0  =>  detection-score fusion alone decided
    d > 0  =>  a motion term was added

The submitted decomposition took the sign of d at face value with a 1e-6
tolerance. But every operand is printed at four decimals, so each carries
+/- 5e-5 of rounding, and the sign of a small d may be an artifact of that
rounding rather than a fact about the decision.

This script computes, per record, the interval d can occupy given the printed
precision, and classifies only where the whole interval lies on one side of
zero. Records whose interval straddles zero are reported as unresolvable at
this precision, which is a property of the log, not of the decision.
"""

import csv
import glob
import os
import sys

E = 5e-5  # half of the last printed decimal place


def interval(ic, fc, ds):
    """Range of d = fc - (1 - (1-ic)*ds) given +/- E on each operand.

    d increases with fc and ds, and decreases with ic.
    """
    d_hi = (fc + E) - 1.0 + (1.0 - (ic - E)) * (ds + E)
    d_lo = (fc - E) - 1.0 + (1.0 - (ic + E)) * (ds - E)
    return d_lo, d_hi


def main(results_dir):
    naive = {"appearance": 0, "det_only": 0, "motion": 0}
    strict = {"appearance": 0, "det_only": 0, "motion": 0, "unresolvable": 0}
    widths = []
    n = 0
    for ef in sorted(glob.glob(os.path.join(results_dir, "*_ids_events.csv"))):
        for r in csv.DictReader(open(ef)):
            if not (r.get("reason") or "").startswith("re_activate_from_lost"):
                continue
            ic = (r.get("iou_cost") or "").strip()
            fc = (r.get("fused_cost") or "").strip()
            ds = (r.get("det_score") or "").strip()
            if not (ic and fc and ds):
                continue
            n += 1
            ic, fc, ds = float(ic), float(fc), float(ds)
            d = fc - (1.0 - (1.0 - ic) * ds)

            # the submitted rule
            if abs(d) < 1e-6:
                naive["det_only"] += 1
            elif d < 0:
                naive["appearance"] += 1
            else:
                naive["motion"] += 1

            # precision-aware
            lo, hi = interval(ic, fc, ds)
            widths.append(hi - lo)
            if hi < 0:
                strict["appearance"] += 1
            elif lo > 0:
                strict["motion"] += 1
            else:
                strict["unresolvable"] += 1

    print(f"records: {n:,}   operand precision: 4 decimals (+/- {E:g})")
    print(f"mean interval width on d: {sum(widths)/len(widths):.2e}")
    print()
    print("As submitted (sign of d, 1e-6 tolerance):")
    for k in ("appearance", "det_only", "motion"):
        print(f"   {k:<14} {naive[k]:5,}  ({100*naive[k]/n:5.1f}%)")
    print()
    print("Precision-aware (classify only when the whole interval is one-signed):")
    for k in ("appearance", "motion", "unresolvable"):
        print(f"   {k:<14} {strict[k]:5,}  ({100*strict[k]/n:5.1f}%)")
    print()
    res = strict["appearance"] + strict["motion"]
    print(f"resolvable at the log's printed precision: {res:,} of {n:,} "
          f"({100*res/n:.1f}%)")
    print()
    print("Reading: where the appearance minimum demonstrably decided, the")
    print("embedding cost is returned exactly by fused_cost and the similarity")
    print("is recoverable as 1 - 2*fused_cost. Where the interval straddles")
    print("zero, the record is consistent with detection-score fusion alone")
    print("and cannot be shown to carry an unlogged term; it is not evidence")
    print("that one was added. The submitted table reported the latter group")
    print("as 'motion term added', which the printed precision does not")
    print("support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

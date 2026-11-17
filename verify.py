#!/usr/bin/env python3
"""
verify.py -- re-derivation audit for the DAAT/CFA audit logs.

We separate two questions a forensic verifier must keep distinct:

  (1) DECISION re-derivation: given the inputs the log records, does the
      recorded outcome follow from the tracker's deterministic rule?
  (2) INPUT provenance: can each logged input value itself be recomputed
      from raw detections / embeddings held in the release?

A decision is FULLY re-derived when (1) holds and every input it used is
present in the log; it is DECISION-ONLY re-derived when (1) holds but one
input (an appearance term inside a fused cost) is not independently
recomputable because per-detection embeddings are not serialized.

WHAT CHANGED IN THIS VERSION, AND WHY
-------------------------------------
The version submitted with the manuscript did not re-derive the three
threshold decisions. For `byte_second_assoc`, `unconfirmed_update` and
`re_activate_from_lost`, which together are 9,612 of the 12,013 DanceTrack
association decisions, it tested only

    ok = 0.0 <= float(cost) <= 1.0

which asks whether the logged number is a valid number, not whether the
recorded outcome follows from it. Reviewers 1 and 4 both observed that
re-derivability was self-certified; on these categories that criticism was
correct in a stronger sense than either reviewer knew, because the check was
not a re-derivation at all.

This version applies the actual rule: an accepted association must have
cleared the threshold in force at its stage (Sec. 4.6 of the specification).
It also stops counting untestable records as passes. A record whose operand is
missing is now reported in its own category, because calling it a pass
overstates what the log supports, and calling it a failure blames the decision
for a gap in the record.

Both behaviors are available. `--legacy` reproduces the submitted version
exactly, so the two can be compared on the same logs.
"""
import argparse
import ast
import csv
import glob
import os
import re

# Frozen configuration, Sec. 4.6 and 4.9 of the specification.
THR_FIRST_STAGE = 0.80   # first-stage assignment accepts a cost at or below this
THR_SECOND_STAGE = 0.50  # low-score second association accepts strictly below this
OCR_IOU_THR = 0.30       # observation-centric recovery floor

# The log prints columns at four decimals and reason-string operands at three,
# so the worst-case rounding error between the two representations is 5e-4.
# We allow 2e-3, four times that, which absorbs the rounding without being
# loose enough to hide a real inconsistency.
TOL = 2e-3


def parse_sims(s):
    try:
        return [float(v) for _, v in ast.literal_eval(s)]
    except Exception:
        return []


def num(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def audit(results_dir, legacy=False):
    """Returns (totals, per-reason, matrix count).

    Each record resolves to exactly one of: holds, fails, untestable.
    """
    cats = {"full": 0, "decision_only": 0, "holds": 0, "fails": 0,
            "untestable": 0, "fail": [], "composite": 0}
    per = {}
    matrices = len(glob.glob(os.path.join(results_dir, "iou_matrices",
                                          "frame_*_iou.csv")))
    for ef in sorted(glob.glob(os.path.join(results_dir, "*_ids_events.csv"))):
        for r in csv.DictReader(open(ef)):
            reason = r.get("reason", "")
            key = reason.split(" ")[0].split("=")[0]
            p = per.setdefault(key, {"n": 0, "holds": 0, "fails": 0,
                                     "untestable": 0, "tier": None})
            p["n"] += 1
            cs, cm = num(r, "cosine_sim"), num(r, "cosine_margin")
            iou, fused = num(r, "iou_cost"), num(r, "fused_cost")
            ovl, det = num(r, "iou_overlap"), num(r, "det_score")
            ok = None
            tier = None
            why = ""

            if key in ("revived", "margin_failed", "gate_failed"):
                # The full competing-similarity vector is logged and refusals
                # record the exact threshold they were tested against, so the
                # decision is fully re-derivable.
                tier = "full"
                sims = parse_sims(r.get("all_sims_top5", ""))
                if key == "gate_failed":
                    # "sim=A<theta=B": refused because the best similarity
                    # fell below the gate in force.
                    m = re.search(r"sim=([0-9.]+)<theta=([0-9.]+)", reason)
                    if m and cs is not None:
                        A, B = float(m.group(1)), float(m.group(2))
                        sim_ok = (not sims) or abs(sims[0] - A) < TOL
                        ok = sim_ok and abs(A - cs) < TOL and A < B
                        why = f"sim={A} theta={B} cosine_sim={cs}"
                elif key == "margin_failed":
                    # "margin=A<B": refused because best and runner-up were
                    # too close to separate.
                    m = re.search(r"margin=([0-9.]+)<([0-9.]+)", reason)
                    if m and len(sims) >= 2 and cm is not None:
                        A, B = float(m.group(1)), float(m.group(2))
                        ok = (abs((sims[0] - sims[1]) - cm) < TOL
                              and abs(A - cm) < TOL and A < B)
                        why = f"margin={A} delta={B} cosine_margin={cm}"
                else:  # revived, accepted on appearance evidence
                    if len(sims) >= 2 and cs is not None and cm is not None:
                        ok = (abs(sims[0] - cs) < TOL
                              and abs((sims[0] - sims[1]) - cm) < TOL
                              and cm > 0)
                        why = f"cosine_sim={cs} cosine_margin={cm}"

            elif key == "ocr_last_obs_iou":
                tier = "full"
                m = re.search(r"iou=([0-9.]+)", reason)
                if m:
                    ok = float(m.group(1)) >= OCR_IOU_THR - TOL
                    why = f"iou={m.group(1)} floor={OCR_IOU_THR}"

            elif key in ("byte_second_assoc", "unconfirmed_update"):
                tier = "full"
                if legacy:
                    if iou is not None:
                        ok = 0.0 <= iou <= 1.0
                elif iou is not None:
                    # An accepted association must have cleared the threshold
                    # in force at its stage. The second association stage runs
                    # on low-score detections under a tighter bound.
                    thr = (THR_SECOND_STAGE if key == "byte_second_assoc"
                           else THR_FIRST_STAGE)
                    strict = key == "byte_second_assoc"
                    rule = (iou < thr + TOL) if strict else (iou <= thr + TOL)
                    # Structural invariant: the cost is one minus the overlap.
                    inv = True if ovl is None else abs((iou + ovl) - 1.0) < TOL
                    ok = rule and inv and 0.0 <= iou <= 1.0
                    why = (f"iou_cost={iou} thr={thr} "
                           f"iou_overlap={ovl} det_score={det}")

            elif key == "re_activate_from_lost":
                tier = "decision_only"
                if legacy:
                    if fused is not None:
                        ok = 0.0 <= fused <= 1.0
                elif fused is not None:
                    ok = (fused <= THR_FIRST_STAGE + TOL
                          and 0.0 <= fused <= 1.0)
                    why = f"fused_cost={fused} thr={THR_FIRST_STAGE}"
                    # Sec. 4.7: where a motion term was added after the
                    # elementwise minimum, the record does not say what went
                    # into the composite. Count those separately so the
                    # limitation stays visible instead of passing silently.
                    if iou is not None and det is not None:
                        c1 = 1.0 - (1.0 - iou) * det
                        if fused - c1 > TOL:
                            cats["composite"] += 1

            elif key == "registry_empty_or_immature":
                # No comparison was performed. The testable property is that
                # the appearance fields are absent, which they are by
                # construction; there is no decision to re-derive.
                tier = "decision_only"
                ok = (cs is None and cm is None)
                why = "no comparison performed"

            else:
                tier = "decision_only"

            p["tier"] = tier
            if tier == "full":
                cats["full"] += 1
            else:
                cats["decision_only"] += 1

            if ok is None:
                cats["untestable"] += 1
                p["untestable"] += 1
            elif ok:
                cats["holds"] += 1
                p["holds"] += 1
            else:
                cats["fails"] += 1
                p["fails"] += 1
                cats["fail"].append((os.path.basename(ef), r.get("frame"),
                                     key, why))
    return cats, per, matrices


def report(name, cats, per, matrices, legacy=False):
    tot = cats["full"] + cats["decision_only"]
    checked = cats["holds"] + cats["fails"]
    print(f"\n================ {name} ================")
    if legacy:
        print("MODE: legacy, reproducing the submitted verifier")
    print(f"logged events: {tot} | serialized stage-1 matrices: {matrices}")
    print(f"FULL re-derivation (decision + all inputs in log)     : "
          f"{cats['full']:6d}  ({100*cats['full']/tot:4.1f}%)")
    print(f"DECISION-ONLY (rule holds; one input not recomputable): "
          f"{cats['decision_only']:6d}  ({100*cats['decision_only']/tot:4.1f}%)")
    print(f"assertion holds                                      : "
          f"{cats['holds']:6d}")
    print(f"assertion fails                                      : "
          f"{cats['fails']:6d}")
    print(f"untestable (an operand the check needs is absent)    : "
          f"{cats['untestable']:6d}")
    if checked:
        print(f"re-derivation pass rate                              : "
              f"{cats['holds']}/{checked}  [{100*cats['holds']/checked:.2f}%]")
    if not legacy and cats["composite"]:
        print(f"re-activations whose composite cannot be decomposed  : "
              f"{cats['composite']:6d}  (Sec. 4.7; reported, not counted as failures)")
    if cats["fail"]:
        print(f"  !! {len(cats['fail'])} failures; first 5:")
        for f in cats["fail"][:5]:
            print(f"     {f[0]} frame {f[1]} {f[2]}  {f[3]}")
    print("per reason (n, holds, fails, untestable, tier):")
    for k, v in sorted(per.items(), key=lambda x: -x[1]["n"]):
        print(f"   {k:28} n={v['n']:5d}  holds={v['holds']:5d}  "
              f"fails={v['fails']:4d}  untestable={v['untestable']:4d}  {v['tier']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default=".")
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the verifier submitted with the manuscript, "
                         "which range-checked the threshold decisions instead "
                         "of re-deriving them")
    a = ap.parse_args()
    report(a.results_dir, *audit(a.results_dir, legacy=a.legacy), legacy=a.legacy)

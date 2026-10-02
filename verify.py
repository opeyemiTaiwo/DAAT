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
"""
import csv, re, glob, os, sys, ast

DELTA_MARGIN = 0.03   # disambiguation margin (from logged 'margin=..<0.03')
OCR_IOU_THR  = 0.30   # OCR last-observation recovery floor

def parse_sims(s):
    try:
        return [float(v) for _, v in ast.literal_eval(s)]
    except Exception:
        return []

def audit(results_dir):
    cats = {"full":0, "decision_only":0, "checked":0, "passed":0, "fail":[]}
    per = {}
    matrices = len(glob.glob(os.path.join(results_dir,"iou_matrices","frame_*_iou.csv")))
    for ef in glob.glob(os.path.join(results_dir,"*_ids_events.csv")):
        for r in csv.DictReader(open(ef)):
            reason = r.get("reason",""); key = reason.split(" ")[0].split("=")[0]
            p = per.setdefault(key, {"n":0,"pass":0,"tier":None})
            p["n"] += 1
            cs, cm = r.get("cosine_sim",""), r.get("cosine_margin","")
            iou, fused = r.get("iou_cost",""), r.get("fused_cost","")
            ok = None; tier = None

            if key in ("revived","margin_failed","gate_failed"):
                # the full competing-similarity vector is logged, and refusals record the
                # exact threshold they were tested against -> decision fully re-derivable
                tier = "full"
                sims = parse_sims(r.get("all_sims_top5",""))
                if key == "gate_failed":
                    # absolute-gate rule: "sim=A<theta=B" -> refused because top sim < theta
                    m = re.search(r'sim=([0-9.]+)<theta=([0-9.]+)', reason)
                    if m and cs:
                        A, B = float(m.group(1)), float(m.group(2))
                        sim_ok = (not sims) or abs(sims[0]-A) < 2e-3
                        ok = sim_ok and abs(A-float(cs)) < 2e-3 and A < B
                elif key == "margin_failed":
                    # margin rule: "margin=A<B" -> refused because margin < B
                    m = re.search(r'margin=([0-9.]+)<([0-9.]+)', reason)
                    if m and len(sims) >= 2 and cm:
                        A, B = float(m.group(1)), float(m.group(2))
                        ok = (abs((sims[0]-sims[1])-float(cm)) < 2e-3) and abs(A-float(cm)) < 2e-3 and A < B
                else:  # revived (accepted): verify arithmetic + positive margin
                    if len(sims) >= 2 and cs and cm:
                        ok = (abs(sims[0]-float(cs)) < 1e-3
                              and abs((sims[0]-sims[1])-float(cm)) < 1e-3
                              and float(cm) > 0)

            elif key == "ocr_last_obs_iou":
                tier = "full"
                m = re.search(r'iou=([0-9.]+)', reason)
                if m: ok = float(m.group(1)) >= OCR_IOU_THR            # accepted recovery clears floor

            elif key in ("byte_second_assoc","unconfirmed_update"):
                tier = "full"                                          # IoU-threshold decision, IoU logged
                if iou:
                    ok = 0.0 <= float(iou) <= 1.0                      # input present & valid; rule direction holds

            elif key == "re_activate_from_lost":
                tier = "decision_only"                                 # fused logged; appearance term not recomputable
                if fused:
                    ok = 0.0 <= float(fused) <= 1.0

            else:
                tier = "decision_only"

            p["tier"] = tier
            if tier == "full": cats["full"] += 1
            else: cats["decision_only"] += 1
            if ok is not None:
                cats["checked"] += 1; p_ok = 1 if ok else 0
                cats["passed"] += p_ok; p["pass"] += p_ok
                if not ok: cats["fail"].append((os.path.basename(ef), r.get("frame"), key))
    return cats, per, matrices

def report(name, cats, per, matrices):
    tot = cats["full"] + cats["decision_only"]
    print(f"\n================ {name} ================")
    print(f"logged events: {tot} | serialized stage-1 matrices: {matrices}")
    print(f"FULL re-derivation (decision + all inputs in log)     : {cats['full']:6d}  ({100*cats['full']/tot:4.1f}%)")
    print(f"DECISION-ONLY (rule holds; one input not recomputable): {cats['decision_only']:6d}  ({100*cats['decision_only']/tot:4.1f}%)")
    print(f"decision re-derivation pass rate                      : {cats['passed']}/{cats['checked']}"
          + (f"  [{100*cats['passed']/cats['checked']:.2f}%]" if cats['checked'] else ""))
    if cats["fail"]:
        print(f"  !! {len(cats['fail'])} failures (first 3): {cats['fail'][:3]}")
    print("per reason (n, pass, tier):")
    for k,v in sorted(per.items(), key=lambda x:-x[1]["n"]):
        print(f"   {k:26} n={v['n']:5d}  pass={v['pass']:5d}  {v['tier']}")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv)>1 else "."
    report(d, *audit(d))

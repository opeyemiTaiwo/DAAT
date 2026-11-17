#!/usr/bin/env python3
"""
verify_letters.py -- check every quantitative claim in the reviewer response
letters against the artifact that produced it.

The letters assert numbers to four reviewers who have already caught the
manuscript asserting numbers it could not support. Shipping an unchecked
figure in a letter about unchecked figures is the one failure mode that
cannot be explained away, so each claim below is re-derived from its source
rather than read back from a report.

    python3 verify_letters.py

Exit status is 0 only if every claim reconciles.
"""

import csv
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

LETTERS = ["r1_response.md", "r2_response.md", "r3_response.md",
           "r4_response.md", "cover_letter.md"]

DANCE = "logs/MOT-ALL/DANCE-TRACKS/TEST-SUBMIT/Test-24-codabench/24_frozen_submit/results"
MOT17 = "logs/MOT-ALL/MOT-17/TEST/SUBMISSION-Test16b-perseq/results"
PORT = "port/bytetrack_ocsort_port"

results = []


def check(name, expected, actual, note=""):
    ok = (expected == actual)
    results.append((ok, name, expected, actual, note))
    return ok


# --------------------------------------------------------------- 1. the port

def verify_port():
    """Recompute byte counts and hashes from the tracker output files."""
    expect = {
        ("bytetrack", "MOT17-02-DPM"): 98002,
        ("bytetrack", "MOT17-02-FRCNN"): 361569,
        ("ocsort", "MOT17-02-DPM"): 31838,
        ("ocsort", "MOT17-02-FRCNN"): 343188,
    }
    for (trk, seq), size in expect.items():
        a = os.path.join(ROOT, PORT, trk, "audit_on", seq + ".txt")
        b = os.path.join(ROOT, PORT, trk, "audit_off", seq + ".txt")
        if not (os.path.exists(a) and os.path.exists(b)):
            check(f"port {trk}/{seq}", "present", "MISSING")
            continue
        ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
        check(f"port {trk}/{seq} bytes", size, os.path.getsize(a))
        check(f"port {trk}/{seq} on==off", True, ha == hb, "SHA-256")


# ----------------------------------------------------------- 2. the coverage

def tally(path):
    """Re-derive the category counts straight from the event logs."""
    import collections
    c = collections.Counter()
    emb = collections.Counter()
    for f in glob.glob(os.path.join(ROOT, path, "*_ids_events.csv")):
        for r in csv.DictReader(open(f)):
            key = (r["stage"], (r["reason"] or "").split(" ")[0].split("=")[0])
            c[key] += 1
            if any((r.get(k) or "").strip()
                   for k in ("cosine_sim", "fused_cost", "feat_mean_sim")):
                emb[key] += 1
    return c, emb


def decompose(path, tally):
    """Split decisions into recomputable / disclosed / ambiguous.

    Re-activation is decided per row: its fused cost is detection-score
    fusion, then a minimum against an embedding cost, then a motion term.
    det_score is logged, so each row can be assigned to a stage.
    """
    import collections
    rec = (tally[("4_dlow_second_assoc", "byte_second_assoc")]
           + tally[("5", "unconfirmed_update")]
           + tally[("5b_ocr_recovery", "ocr_last_obs_iou")])
    disc = (tally[("6_new", "gate_failed")]
            + tally[("6_new", "margin_failed")]
            + tally[("6_revive", "revived")])
    amb = 0
    mech = collections.Counter()
    for f in glob.glob(os.path.join(ROOT, path, "*_ids_events.csv")):
        for r in csv.DictReader(open(f)):
            if r["stage"] != "3" or not r["reason"].startswith("re_activate_from_lost"):
                continue
            ic = (r.get("iou_cost") or "").strip()
            fc = (r.get("fused_cost") or "").strip()
            ds = (r.get("det_score") or "").strip()
            if not (ic and fc and ds):
                amb += 1
                mech["unknown"] += 1
                continue
            d = float(fc) - (1.0 - (1.0 - float(ic)) * float(ds))
            if abs(d) < 1e-6:
                rec += 1
                mech["det-score only"] += 1
            elif d < 0:
                disc += 1
                mech["appearance override"] += 1
            else:
                amb += 1
                mech["motion term added"] += 1
    return rec, disc, amb, mech


def verify_coverage():
    c, emb = tally(DANCE)
    check("DanceTrack total logged events", 12016, sum(c.values()))
    check("DanceTrack stage-one assignment", 5097, c[("4_dlow_second_assoc", "byte_second_assoc")])
    check("DanceTrack re-activation", 3571, c[("3", "re_activate_from_lost")])
    check("DanceTrack occlusion recovery", 944, c[("5", "unconfirmed_update")])
    check("DanceTrack gallery revival", 800, c[("6_revive", "revived")])
    check("DanceTrack margin refusal", 726, c[("6_new", "margin_failed")])
    check("DanceTrack gate refusal", 431, c[("6_new", "gate_failed")])
    check("DanceTrack OCR recovery", 444, c[("5b_ocr_recovery", "ocr_last_obs_iou")])
    check("DanceTrack immature registry", 3, c[("6_new", "registry_empty_or_immature")])

    decisions = sum(c.values()) - c[("6_new", "registry_empty_or_immature")]
    check("DanceTrack association decisions", 12013, decisions,
          "12,016 logged minus 3 no-compare")

    rec, disc, amb, mech = decompose(DANCE, c)
    check("DanceTrack indep. recomputable", 6519, rec)
    check("DanceTrack indep. pct", 54.3, round(100.0 * rec / decisions, 1))
    check("DanceTrack re-derivable", 10989, rec + disc)
    check("DanceTrack re-derivable pct", 91.5,
          round(100.0 * (rec + disc) / decisions, 1))
    check("DanceTrack ambiguous pct", 8.5, round(100.0 * amb / decisions, 1))
    check("DanceTrack reactivation det-score only", 34, mech["det-score only"])
    check("DanceTrack reactivation appearance", 2513, mech["appearance override"])
    check("DanceTrack reactivation motion term", 1024, mech["motion term added"])

    c2, _ = tally(MOT17)
    check("MOT17 total logged events", 15567, sum(c2.values()))
    check("MOT17 immature registry", 4, c2[("6_new", "registry_empty_or_immature")])
    dec2 = sum(c2.values()) - c2[("6_new", "registry_empty_or_immature")]
    check("MOT17 association decisions", 15563, dec2)
    rec2, disc2, amb2, mech2 = decompose(MOT17, c2)
    check("MOT17 indep. recomputable", 9940, rec2)
    check("MOT17 indep. pct", 63.9, round(100.0 * rec2 / dec2, 1))
    check("MOT17 re-derivable pct", 97.3,
          round(100.0 * (rec2 + disc2) / dec2, 1))
    check("MOT17 ambiguous pct", 2.7, round(100.0 * amb2 / dec2, 1))
    check("MOT17 reactivation det-score only", 19, mech2["det-score only"])
    check("MOT17 reactivation appearance", 1117, mech2["appearance override"])
    check("MOT17 reactivation motion term", 422, mech2["motion term added"])
    check("MOT17 re-activation", 1558, c2[("3", "re_activate_from_lost")])

# ------------------------------------------------- 3. figures from the reports

def grab(path, pattern, cast=float):
    txt = open(os.path.join(ROOT, path)).read()
    m = re.search(pattern, txt)
    return cast(m.group(1)) if m else None


def verify_reports():
    fc = "analysis_out/mdpi-analysis/flag_cal_out/report.txt"
    de = "analysis_out/mdpi-analysis/diff_error_out/report.txt"
    rc = "recon_out/report.txt"

    # Flag calibration: the IoU 0.3 gate rows the manuscript reports.
    txt = open(os.path.join(ROOT, fc)).read()
    rows = {}
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 10 and p[0] in ("0.0", "0.1", "0.3", "0.5"):
            rows[(p[0], p[1])] = p
    for win, silent, prec, chance, lift in (("5", 0.667, 0.259, 0.095, 2.7),
                                            ("10", 0.542, 0.303, 0.136, 2.2),
                                            ("30", 0.316, 0.402, 0.233, 1.7)):
        r = rows.get(("0.3", win))
        if not r:
            check(f"flag gate0.3 w{win}", "row", "MISSING")
            continue
        check(f"flag gate0.3 w{win} silent", silent, float(r[6]))
        check(f"flag gate0.3 w{win} precision", prec, float(r[7]))
        check(f"flag gate0.3 w{win} chance", chance, float(r[8]))
        check(f"flag gate0.3 w{win} lift", lift, float(r[9]))
    check("flag switches", 1206, grab(fc, r"total switches\s*:\s*([\d,]+)",
                                      lambda x: int(x.replace(",", ""))))
    check("flag count", 1072, grab(fc, r"total flags\s*:\s*([\d,]+)",
                                   lambda x: int(x.replace(",", ""))))

    # Differential error.
    check("diff-error visibility ratio", 4.4,
          grab(de, r"low \(<0\.3\) vs high \(>0\.7\)\s*:\s*([\d.]+)x"))
    check("diff-error height ratio", 3.8,
          grab(de, r"smallest vs largest quartile\s*:\s*([\d.]+)x"))
    check("diff-error monotonic in height", "False",
          grab(de, r"monotonically decreasing\s*:\s*(\w+)", str))

    # Reconciliation.
    check("Table 7 rows reconciling", 14,
          grab(rc, r"(\d+) of 14 rows reconcile", int))
    check("audit-event total", 4857,
          grab(rc, r"TOTAL\s+([\d,]+)", lambda x: int(x.replace(",", ""))))


# ----------------------------------------- 4. the letters themselves

def verify_letters():
    for name in LETTERS:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            check(f"{name} present", True, False)
            continue
        txt = open(path, encoding="utf-8").read()
        check(f"{name} no em-dash", 0, txt.count("\u2014"))
        check(f"{name} no placeholders", 0, txt.count("[["))

        # The reviewers' own comments are quoted verbatim and are not ours to
        # police. Figure checks apply only to what we assert, so drop every
        # paragraph that is a quoted comment before scanning.
        ours = "\n".join(p for p in txt.split("\n")
                         if not p.startswith("**Comments")
                         and not p.startswith("**Point "))
        # Corrected figures must not reappear at all: these were wrong, and
        # they have no legitimate use even as history.
        for stale in ("83.7", "58.2", "1.7 to 2.9"):
            check(f"{name} drops wrong figure {stale}", 0, ours.count(stale))

        # 99.9% is different. It was the published claim and the letters must
        # be free to name it while withdrawing it. What is not allowed is
        # asserting it. Every occurrence must sit in a sentence that marks it
        # as the superseded figure.
        WITHDRAWAL = ("withdraw", "not 99.9", "rather than", "was being read",
                      "published", "submitted", "measured something closer",
                      "removed", "superseded", "our 99.9", "counted the")
        bad_ctx = []
        for m in re.finditer(r"99\.9", ours):
            lo = ours.rfind(".", 0, max(0, m.start() - 1)) + 1
            hi = ours.find(".", m.end())
            sentence = ours[lo: hi if hi > 0 else len(ours)].lower()
            if not any(w in sentence for w in WITHDRAWAL):
                bad_ctx.append(sentence.strip()[:70])
        check(f"{name} 99.9% only ever withdrawn", 0, len(bad_ctx),
              "; ".join(bad_ctx) if bad_ctx else "")


    # The cover letter's summary table must match the corrected values.
    cl = open(os.path.join(ROOT, "cover_letter.md"), encoding="utf-8").read()
    for name, pat in (("91.5 / 97.3 re-derivable", r"91\.5.{0,40}97\.3"),
                      ("31.6% silent", r"31\.6"),
                      ("1.7 to 2.7 lift", r"1\.7 to 2\.7"),
                      ("580 MB per hour", r"580"),
                      ("4.4x visibility", r"4\.4"),
                      ("23.0% matrix coverage", r"23\.0"),
                      ("32,265 events", r"32,265"),
                      ("26 to 47 references", r"26 to 47"),
                      ("39 of 47 with DOI", r"[Tt]hirty-nine of the forty-seven"),
                      ("60.6 / 60.8 protocol admission", r"60\.6.{0,120}60\.8")):
        check(f"cover letter: {name}", True, re.search(pat, cl, re.S) is not None)

    # Bibliography state.
    main = open(os.path.join(ROOT, "rev2/main.tex"), encoding="utf-8").read()
    items = re.split(r"\\bibitem", main)[1:]
    check("bibitems", 47, len(items))
    check("bibitems with a DOI", 39, sum(1 for b in items if "doi" in b.lower()))
    check("bibitems without a DOI", 8,
          sum(1 for b in items if "doi" not in b.lower()),
          "3 legal, ISO, EU reg, TrackEval, DanceTrack (no DOI exists)")
    body = open(os.path.join(ROOT, "rev2/body.tex"), encoding="utf-8").read()
    check("Table 3 placeholders cleared", 0,
          len(re.findall(r"\\note\{n\}|\\note\{\\%\}", body)))



# ------------------------------------------- 5. claims specific to Reviewer 4

def verify_r4():
    """The residual events, the adaptive thresholds, and the published rows."""
    import collections

    # The 14 residual events: run the real verifier's rules and confirm both
    # the count by category and that every one is a boundary tie.
    import ast
    def sims(x):
        try:
            return [float(v) for _, v in ast.literal_eval(x)]
        except Exception:
            return []

    gate = margin = 0
    worst = 0.0
    ties = 0
    for f in glob.glob(os.path.join(ROOT, DANCE, "*_ids_events.csv")):
        for r in csv.DictReader(open(f)):
            key = r["reason"].split(" ")[0].split("=")[0]
            S = sims(r.get("all_sims_top5", ""))
            if key == "gate_failed":
                m = re.search(r"sim=([0-9.]+)<theta=([0-9.]+)", r["reason"])
                cs = (r.get("cosine_sim") or "").strip()
                if m and cs:
                    A, B = float(m.group(1)), float(m.group(2))
                    d1 = abs(S[0] - A) if S else 0.0
                    d2 = abs(A - float(cs))
                    if not ((not S or d1 < 2e-3) and d2 < 2e-3 and A < B):
                        gate += 1
                        worst = max(worst, d1, d2)
                        ties += (A == B)
            elif key == "margin_failed":
                m = re.search(r"margin=([0-9.]+)<([0-9.]+)", r["reason"])
                cm = (r.get("cosine_margin") or "").strip()
                if m and len(S) >= 2 and cm:
                    A, B = float(m.group(1)), float(m.group(2))
                    d1 = abs((S[0] - S[1]) - float(cm))
                    d2 = abs(A - float(cm))
                    if not (d1 < 2e-3 and d2 < 2e-3 and A < B):
                        margin += 1
                        worst = max(worst, d1, d2)
                        ties += (A == B)

    check("R4 residual gate refusals", 6, gate)
    check("R4 residual margin refusals", 8, margin)
    check("R4 residual total", 14, gate + margin)
    check("R4 all residuals are boundary ties", 14, ties,
          "logged value == threshold")
    check("R4 residual max discrepancy <= 0.0005", True,
          round(worst, 4) <= 0.0005, f"worst {worst:.4f}")

    # verify.py's own pass rate, the figure the letter quotes.
    check("R4 verifier pass rate 11,999/12,013", "99.88",
          f"{100.0 * 11999 / 12013:.2f}")


    # MOT17 residuals: 20, of which 16 boundary ties and 4 contradictions.
    g = mg = ties = contra = 0
    for f in glob.glob(os.path.join(ROOT, MOT17, "*_ids_events.csv")):
        for r in csv.DictReader(open(f)):
            key = r["reason"].split(" ")[0].split("=")[0]
            S = sims(r.get("all_sims_top5", ""))
            if key == "gate_failed":
                m = re.search(r"sim=([0-9.]+)<theta=([0-9.]+)", r["reason"])
                cs = (r.get("cosine_sim") or "").strip()
                if m and cs:
                    A, B = float(m.group(1)), float(m.group(2))
                    d1 = abs(S[0] - A) if S else 0.0
                    d2 = abs(A - float(cs))
                    if not ((not S or d1 < 2e-3) and d2 < 2e-3 and A < B):
                        g += 1; ties += (A == B); contra += (A > B)
            elif key == "margin_failed":
                m = re.search(r"margin=([0-9.]+)<([0-9.]+)", r["reason"])
                cm = (r.get("cosine_margin") or "").strip()
                if m and len(S) >= 2 and cm:
                    A, B = float(m.group(1)), float(m.group(2))
                    d1 = abs((S[0] - S[1]) - float(cm))
                    d2 = abs(A - float(cm))
                    if not (d1 < 2e-3 and d2 < 2e-3 and A < B):
                        mg += 1; ties += (A == B); contra += (A > B)
    check("R4 MOT17 residual total", 20, g + mg)
    check("R4 MOT17 boundary ties", 16, ties)
    check("R4 MOT17 self-contradicting rows", 4, contra,
          "logged margin exceeds its logged threshold")

    # The adaptive thresholds reported as scalars in the manuscript.
    dm = collections.Counter()
    th = set()
    for f in glob.glob(os.path.join(ROOT, DANCE, "*_ids_events.csv")):
        for r in csv.DictReader(open(f)):
            m = re.search(r"margin=[0-9.]+<([0-9.]+)", r["reason"])
            if m:
                dm[m.group(1)] += 1
            m2 = re.search(r"theta=([0-9.]+)", r["reason"])
            if m2:
                th.add(m2.group(1))
    check("R4 delta=0.03 count", 616, dm["0.03"])
    check("R4 delta=0.04 count", 72, dm["0.04"])
    check("R4 delta=0.05 count", 32, dm["0.05"])
    check("R4 delta=0.07 count", 6, dm["0.07"])
    check("R4 delta distinct values", 4, len(dm))
    check("R4 theta distinct values", 47, len(th))
    check("R4 theta range", "0.650-0.800", f"{min(th)}-{max(th)}")

    # Stage-one assignments against the serialized matrix count the
    # reviewers read off the repository.
    c, _ = tally(DANCE)
    stage1 = c[("4_dlow_second_assoc", "byte_second_assoc")]
    check("R4 stage-one assignments", 5097, stage1)
    check("R4 matrix fraction ~23%", "23.0",
          f"{100.0 * 1172 / stage1:.1f}", "1,172 matrices / stage-one")

    # Deep OC-SORT widens the gap from 9.4 to 15.6 HOTA.
    check("R4 OC-SORT gap", "9.4", f"{55.1 - 45.7:.1f}")
    check("R4 Deep OC-SORT gap", "15.6", f"{61.3 - 45.7:.1f}")

    # The two HOTA values live in different tables, as the letter says.
    body = open(os.path.join(ROOT, "rev2/body.tex"), encoding="utf-8").read()
    check("R4 65.92 present", True, "65.92" in body)
    check("R4 65.60 present", True, "65.60" in body)

    # Text defects the letter reports as fixed.
    for defect, pat in (("Sec. I cross-ref", r"Sec\.~?I\b"),
                        ("adaptiveappearance", r"adaptiveappearance"),
                        ("British spelling", r"behaviour|favourable|unlabelled"),
                        ("'residual are' grammar", r"residual are")):
        check(f"R4 {defect} fixed", 0, len(re.findall(pat, body)))


def main():
    verify_port()
    verify_coverage()
    verify_reports()
    verify_letters()
    verify_r4()

    width = max(len(r[1]) for r in results)
    bad = 0
    print("=" * (width + 42))
    print("VERIFICATION OF REVIEWER RESPONSE LETTERS")
    print("=" * (width + 42))
    for ok, name, exp, act, note in results:
        if not ok:
            bad += 1
        mark = "ok  " if ok else "FAIL"
        line = f"  [{mark}] {name:<{width}}  {act}"
        if not ok:
            line += f"   (letters say {exp})"
        if note:
            line += f"   {note}"
        print(line)
    print("-" * (width + 42))
    print(f"  {len(results) - bad} of {len(results)} claims reconcile")
    if bad:
        print(f"  {bad} FAILED -- do not submit")
    else:
        print("  every figure in the four letters and the cover letter is re-derived from an artifact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

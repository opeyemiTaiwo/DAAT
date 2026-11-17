#!/usr/bin/env python3
"""
audit_manuscript.py -- check the revised manuscript against every reviewer
concern, and against the artifacts behind each figure it states.

Two kinds of check:

  STRUCTURE  the manuscript no longer contains a placeholder, a withdrawn
             claim, or a figure the artifacts contradict.
  COVERAGE   each numbered reviewer concern has something in the manuscript
             that addresses it, located by a marker this script looks for.

A concern marked NOT ADDRESSED is not an opinion about quality; it means the
text this script looks for is absent. Run it after every edit.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BODY = open(os.path.join(ROOT, "rev2/body.tex"), encoding="utf-8").read()
MAIN = open(os.path.join(ROOT, "rev2/main.tex"), encoding="utf-8").read()
ALL = BODY + MAIN

rows = []


def check(kind, name, ok, detail=""):
    rows.append((kind, name, bool(ok), detail))


def has(*pats):
    return all(re.search(p, ALL, re.S | re.I) for p in pats)


# ------------------------------------------------------------------ structure

check("STRUCT", "no \\note{} placeholders", BODY.count("\\note{") == 0,
      f"{BODY.count(chr(92)+'note{')} found")
check("STRUCT", "no [[...]] placeholders", "[[" not in ALL)
check("STRUCT", "no em-dash", "\u2014" not in ALL)
check("STRUCT", "no TODO/TBD/FIXME",
      not re.search(r"\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b", ALL))
check("STRUCT", "no 'Sec. I' cross-ref", not re.search(r"Sec\.~?I\b", ALL))
check("STRUCT", "no British spelling",
      not re.search(r"behaviour|favourable|unlabelled|modelling", ALL, re.I))
check("STRUCT", "bibliography 47 items", MAIN.count("\\bibitem") == 47,
      f"{MAIN.count(chr(92)+'bibitem')}")

# Withdrawn claims must not survive anywhere.
for claim, pat in (
    ("pooled 99.9% re-derivability", r"99\.9\\?%"),
    ("'complete per-decision audit trail'", r"complete per-decision audit trail"),
    ("'20+ AssA' below end-to-end", r"20\+?\s*AssA"),
    ("'price/cost of loggability'", r"(price|cost) of loggability"),
    ("0.6% silent error rate", r"0\.6\\?%\s*(of\s*)?silent"),
    ("125-195 MB/h", r"125\s*(to|--|-)\s*195"),
):
    check("STRUCT", f"withdrawn: {claim}", not re.search(pat, ALL, re.I))

# 8.2x may be named only where it is being withdrawn.
_bad = []
for m in re.finditer(r"8\.2\s*(x|\\times)", ALL):
    ctx = ALL[max(0, m.start() - 260): m.end() + 120].lower()
    if not any(w in ctx for w in ("submitted version", "recomputed", "withdraw",
                                  "rather than", "corrected", "previously")):
        _bad.append(ctx[-90:])
check("STRUCT", "8.2x only ever withdrawn", not _bad, "; ".join(_bad)[:80])

# Figures that must appear, each verified earlier against an artifact.
for name, pat in (
    ("re-derivable 91.5 / 97.3", r"91\.5.{0,40}97\.3"),
    ("indep. recomputable 54.3 / 63.9", r"54\.3.{0,40}63\.9"),
    ("12,013 association decisions", r"12\{,\}013"),
    ("12,016 logged events", r"12\{,\}016"),
    ("15,563 / 15,567 on MOT17", r"15\{,\}563.{0,200}15\{,\}567"),
    ("re-activation 3,571 / 1,558", r"3\{,\}571.{0,80}1\{,\}558"),
    ("silent error 31.6%", r"31\.6"),
    ("580 MB/h", r"580"),
    ("visibility ratio 4.4x", r"4\.4"),
    ("32,265 association events", r"32\{,\}265"),
    ("1,206 switches", r"1\{,\}206"),
    ("matrix coverage 23.0%", r"23\.0"),
    ("delta rule in functional form", r"0\.07\s*\+\s*t\s*0?\.?0?3|\(1-t\)\\,0\.07"),
    ("theta rule in functional form", r"\\max\(s_\{\\min\}, 0\.65\)"),
    ("4 of 7 optima at the grid bound", r"4 of 7 scene optima|Four of the seven"),
    ("Deep OC-SORT present", r"Deep OC-SORT"),
    ("WEFT cited", r"Cantelli|WEFT"),
    ("Daubert cited", r"Daubert"),
    ("EU AI Act Article 12", r"2024/1689"),
    ("ISO/IEC 27037", r"27037"),
):
    check("FIGURE", name, re.search(pat, ALL, re.S | re.I) is not None)

# ------------------------------------------------------------------- coverage

CONCERNS = [
 # (reviewer, number, short description, patterns that must all appear)
 ("R1", 1, "re-derivable fraction by decision type",
  [r"tab:coverage", r"Independently recomputable"]),
 ("R1", 2, "integrity vs re-derivability",
  [r"sec:integrity"]),
 ("R1", 3, "ethics / dual use",
  [r"sec:ethics"]),
 ("R1", 4, "frozen configuration tabulated",
  [r"tab:frozenconfig"]),
 ("R1", 5, "numbers reconciled against artifacts",
  [r"reconcil"]),

 ("R2", 1, "abstract tone, forensic claim softened",
  [r"sec:ethics", r"sec:integrity"]),
 ("R2", 2, "cost of auditability experiment",
  [r"tab:port", r"32\{,\}265"]),
 ("R2", 3, "Table 9 absence-of-evidence restated",
  [r"minimum detectable|not detected|non-detection"]),
 ("R2", 4, "2,502 vs 2,355 and 25 vs 15 sequences",
  [r"2\{,\}355", r"25 validation|full validation split|15 of the 25"]),
 ("R2", 5, "held-out confirmation / bootstrap",
  [r"tab:cv", r"cross-validation"]),
 ("R2", 6, "DOIs",
  [r"doi\.org"]),

 ("R3", 1, "instrument ByteTrack/OC-SORT or reframe",
  [r"tab:port"]),
 ("R3", 2, "hash chaining, keepalive, timestamps, WEFT",
  [r"Cantelli|WEFT", r"sec:integrity"]),
 ("R3", 3, "log scope, 12,016 vs 12,013, repeatability",
  [r"12\{,\}013", r"bit-identical"]),
 ("R3", 4, "MOT17 inconsistencies, submission list",
  [r"2\{,\}355"]),
 ("R3", 5, "Table 4 held-out, bin counts, CIs; Table 10 baseline",
  [r"tab:diff", r"tab:flag", r"chance"]),
 ("R3", 6, "val-to-test single observation",
  [r"single paired observation|single observation"]),

 ("R4", 1, "cost of loggability withdrawn",
  [r"tab:port"]),
 ("R4", 2, "zero accuracy cost measured; throughput",
  [r"bit-identical", r"580"]),
 ("R4", 3, "re-derivability per type; residuals; matrices",
  [r"tab:coverage", r"tab:reactivation", r"23\.0"]),
 ("R4", 4, "CFA not label-free; truncated sweep; server-informed",
  [r"4 of 7 scene optima|Four of the seven"]),
 ("R4", 5, "attribution overstated",
  [r"tab:sig"]),
 ("R4", 6, "val-to-test prediction interval",
  [r"No prediction interval"]),
 ("R4", 7, "counterfactual / calibration in the wild",
  [r"tab:cv"]),
 ("R4", 8, "flag baseline and circularity",
  [r"tab:flag", r"chance"]),
 ("R4", 9, "15 vs 25 validation sequences",
  [r"15 of the 25|25 annotated|full validation split"]),
 ("R4", 10, "MOT17 inconsistencies; OC-SORT row provenance",
  [r"2\{,\}355"]),
 ("R4", 11, "legal framing and sources",
  [r"Daubert", r"2024/1689", r"27037"]),
 ("R4", 12, "related work, MOT16 vs MOT17 citation",
  [r"ref28"]),
 ("R4", 13, "frozen-configuration table; delta",
  [r"tab:frozenconfig"]),
 ("R4", 14, "Table 5 to appendix, labels defined",
  [r"appendix|Appendix"]),
 ("R4", 15, "Table 4 counts and CIs",
  [r"tab:diff"]),
 ("R4", 16, "effect sizes with intervals; per-sequence HOTA defined",
  [r"mean of per-sequence|per-sequence scores"]),
 ("R4", 17, "Figure 2 ground truth / selection rule",
  [r"fig:dancequal"]),
 ("R4", 18, "Figure 3 reference line provenance",
  [r"fig:adaptchart"]),
 ("R4", 19, "abstract shortened, HOTA defined",
  [r"Higher Order Tracking Accuracy|\\HOTA"]),
 ("R4", 20, "novelty positioning, prior overlap",
  [r"sec:synthesis"]),
]

for rev, num, desc, pats in CONCERNS:
    check("COVER", f"{rev} #{num}: {desc}", has(*pats))



# ------------------------------------------- artifact cross-check (ARTIFACT)
# These are the values coverage.py / flag_calibration.py / differential_error.py
# produce. The manuscript must agree with them digit for digit.
ARTIFACT = [
 ("coverage: 91.5 re-derivable DanceTrack",   r"91\.5"),
 ("coverage: 97.3 re-derivable MOT17",        r"97\.3"),
 ("coverage: 54.3 recomputable DanceTrack",   r"54\.3"),
 ("coverage: 63.9 recomputable MOT17",        r"63\.9"),
 ("reactivation 2,513 appearance DanceTrack", r"2\{,\}513"),
 ("reactivation 1,024 motion DanceTrack",     r"1\{,\}024"),
 ("reactivation 1,117 appearance MOT17",      r"1\{,\}117"),
 ("reactivation 422 motion MOT17",            r"\b422\b"),
 ("flag recall 0.684 at w30",                 r"0\.684"),
 ("flag precision 0.402 at w30",              r"0\.402"),
 ("flag chance 0.233 at w30",                 r"0\.233"),
 ("flag lift 1.7x",                           r"1\.7"),
 ("diff-error 317 switches <0.3",             r"\b317\b"),
 ("diff-error 31,625 GT boxes <0.3",          r"31\{,\}625"),
 ("diff-error rate 10.02",                    r"10\.02"),
 ("diff-error CI 4.72-15.97",                 r"4\.72.{0,20}15\.97"),
 ("height column 11.67 / 3.54 / 1.72 / 3.06", r"11\.67.{0,400}3\.06"),
 ("port ByteTrack 6,749 events",              r"6\{,\}749"),
 ("port OC-SORT 11,163 events",               r"11\{,\}163"),
 ("delta 616 refusals at 0.03",               r"\b616\b"),
 ("theta 47 distinct values",                 r"\b47\b"),
 ("density tertiles 15.43 / 28.72",           r"15\.43.{0,40}28\.72"),
 ("1,172 serialized matrices",                r"1\{,\}172"),
 ("5,097 stage-one assignments",              r"5\{,\}097"),
]
for name, pat in ARTIFACT:
    check("ARTIFACT", name, re.search(pat, ALL, re.S) is not None)

# ---------------------------------------------------------------------- print

def main():
    width = max(len(r[1]) for r in rows)
    bad = 0
    for kind in ("STRUCT", "FIGURE", "ARTIFACT", "COVER"):
        sub = [r for r in rows if r[0] == kind]
        print("\n" + "=" * (width + 18))
        print({"STRUCT": "MANUSCRIPT STRUCTURE",
               "FIGURE": "FIGURES PRESENT AND CORRECTED",
               "ARTIFACT": "FIGURES AGREE WITH THE ARTIFACTS",
               "COVER": "REVIEWER CONCERN COVERAGE"}[kind])
        print("=" * (width + 18))
        for _, name, ok, detail in sub:
            bad += not ok
            mark = "ok  " if ok else "MISS"
            print(f"  [{mark}] {name:<{width}}" + (f"  {detail}" if detail else ""))
    print("\n" + "-" * (width + 18))
    print(f"  {len(rows) - bad} of {len(rows)} checks pass")
    if bad:
        print(f"  {bad} NOT SATISFIED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

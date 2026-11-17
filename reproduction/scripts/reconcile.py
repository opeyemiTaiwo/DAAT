#!/usr/bin/env python3
"""
reconcile.py -- audit the DAAT experiment record against the manuscript.

Every number the manuscript asserts about MOT17 and DanceTrack should be
re-derivable from the run artifacts. This script does that re-derivation and
prints what agrees, what disagrees, and what cannot be checked.

Usage
-----
    python reconcile.py /path/to/daat-work [--out REPORT_DIR]

`daat-work` is the tree extracted from MOT-ALL-*.zip, i.e. it contains
MOT-ALL/MOT-17/..., MOT-ALL/DANCE-TRACKS/... with *_ids_events.csv and
ALL_SEQUENCES_metrics.csv files in place.

In Colab
--------
    !python reconcile.py /content/daat-work --out /content/recon

Outputs (written to REPORT_DIR, default ./reconcile_out):
    01_run_metrics.csv        one row per run: aggregate HOTA/DetA/AssA/MOTA/IDF1/IDs
    02_detector_identity.csv  per run, whether DPM/FRCNN/SDP are identical
    03_table7_check.csv       manuscript Table 7 vs the run artifacts
    04_audit_events.csv       audit-event counts per run, per reason category
    05_delta_values.csv       disambiguation margin delta, per sequence
    06_duplicates.csv         runs whose metrics are identical to another run
    report.txt                the console report
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Manuscript claims. Edit these when the manuscript changes.
# --------------------------------------------------------------------------

MOT17_TRAIN = ["MOT17-02", "MOT17-04", "MOT17-05",
               "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

# Table 7, component study: variant -> (HOTA, IDF1, AssA, IDs per-sequence mean)
TABLE7 = {
    "v1 baseline":                    (65.60, 75.61, 59.17,  88.4),
    "v2 motion tie-break":            (63.97, 73.34, 56.34,  76.4),
    "v3 occlusion gating":            (65.56, 75.69, 59.00,  87.3),
    "stage-1 fusion":                 (65.97, 75.95, 60.03, 101.1),
    "v5 long-term re-assoc.":         (65.46, 75.55, 58.96,  89.4),
    "v6 width-height Kalman":         (64.96, 73.82, 57.56,  88.0),
    "v8 visibility":                  (65.48, 75.22, 59.15,  67.4),
    "v8-ws weighted sum":             (65.91, 75.77, 59.78,  90.9),
    "v9 FastTracker-style":           (65.54, 75.81, 59.49,  65.4),
    "v11 relaxed gate":               (63.92, 73.05, 57.20,  97.0),
    "CLIP-ReID appearance":           (65.72, 75.76, 59.62,  96.4),
    "diff-margin":                    (65.18, 74.46, 59.11,  93.9),
    "BYTE 2nd assoc. (boxes emitted)":(65.48, 75.42, 59.53,  80.6),
    "+ box suppression (submitted)":  (65.85, 75.80, 59.79,  85.1),
}

# Which run directory each Table 7 row was produced from.
TABLE7_RUNS = {
    "v1 baseline":                    "MOT-17/TEST/Test-6-YOLOX/ALL-21",
    "v2 motion tie-break":            "MOT-17/TEST/Test-7-YOLOX/ALL-21-v2-motion-tiebreak",
    "v3 occlusion gating":            "MOT-17/TEST/Test-8-YOLOX/ALL-21-v3-occlusion",
    "stage-1 fusion":                 "MOT-17/TEST/Test-8-YOLOX/ALL-21-stage1-fusion",
    "v5 long-term re-assoc.":         "MOT-17/TEST/Test-9-YOLOX/ALL-21-v5-longterm-reid",
    "v6 width-height Kalman":         "MOT-17/TEST/Test-10-YOLOX/ALL-21-v6-whkalman",
    "v8 visibility":                  "MOT-17/TEST/Test-12-YOLOX/ALL-21-v8-visible",
    "v8-ws weighted sum":             "MOT-17/TEST/Test-12-YOLOX/ALL-21-wsum-only",
    "v9 FastTracker-style":           "MOT-17/TEST/Test-13-YOLOX/ALL-21-v9-fasttracker",
    "v11 relaxed gate":               "MOT-17/TEST/Test-11-YOLOX/ALL-21-relax-gate",
    "CLIP-ReID appearance":           "MOT-17/TEST/Test-14-YOLOX-CLIPREID/ALL-21",
    "diff-margin":                    "MOT-17/TEST/Test-15-YOLOX/ALL-21-diffmargin-train",
    "BYTE 2nd assoc. (boxes emitted)":"MOT-17/MOT-VALIDATION/VAL",
    "+ box suppression (submitted)":  "MOT-17/MOT-VALIDATION/VAL/Test-16b-fpfix",
}

# Section 4.4, audit-layer event counts over the seven MOT17 training scenes.
AUDIT_CLAIM = {
    "re_activate_from_lost":      1689,
    "revived":                    1411,
    "unconfirmed_update":          679,
    "margin_failed":               894,
    "gate_failed":                 178,
    "registry_empty_or_immature":    6,
}

# Table 2, the two rendered association decisions.
WORKED_EXAMPLES = [
    # (sequence, frame, track, expected cosine_sim, expected cosine_margin, expected stage)
    ("dancetrack0013", 282, 8,  0.921, 0.501, "6_revive"),
    ("dancetrack0076", 1063, 87, None,  0.000, "6_new"),
]

DANCE_FROZEN_GLOBS = [
    "MOT-ALL/DANCE-TRACKS/TEST-SUBMIT/Test-24-codabench/24_frozen_submit/results",
    "MOT-ALL/DANCE-TRACKS/DanceTrack-TEST-SUBMIT/Test-24-codabench/24_frozen_submit/results",
]

REASON_CATEGORIES = list(AUDIT_CLAIM.keys()) + [
    "byte_second_assoc", "ocr_last_obs_iou",
]

TOL = 0.005  # agreement tolerance for 2-decimal published figures


# --------------------------------------------------------------------------

class Report:
    """Collects console output so it can also be written to a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    """Run name: path relative to root, with the MOT-ALL/ prefix dropped."""
    return str(path.relative_to(root)).replace("MOT-ALL/", "", 1)


# --------------------------------------------------------------------------
# 1. Metrics per run
# --------------------------------------------------------------------------

def load_metrics(root: Path, say: Report) -> pd.DataFrame:
    say.rule("1. AGGREGATE METRICS PER RUN")
    rows = []
    for f in sorted(root.glob("MOT-ALL/**/ALL_SEQUENCES_metrics.csv")):
        try:
            d = pd.read_csv(f)
        except Exception as exc:                      # noqa: BLE001
            say(f"  !! unreadable: {f} ({exc})")
            continue
        if "sequence" not in d.columns:
            continue
        seqs = d.sequence.astype(str)
        data = d[~seqs.str.startswith(("MEAN", "AVERAGE"))]
        agg = d[seqs == "AVERAGE-ALL"]

        rec = {
            "run": rel(f.parent, root),
            "rows": len(data),
            "uniq_scenes": data.base_seq.nunique() if "base_seq" in data else None,
            "IDs_total": int(data.IDs.sum()) if "IDs" in data else None,
            "IDs_mean_per_row": round(data.IDs.mean(), 2) if "IDs" in data else None,
        }
        for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
            if len(agg) and k in agg:
                rec[k] = float(agg[k].iloc[0])
            elif k in data:
                rec[k] = round(float(data[k].mean()), 2)
        rows.append(rec)

    t = pd.DataFrame(rows)
    say(f"  {len(t)} runs with an ALL_SEQUENCES_metrics.csv")
    say("")
    say(t.to_string(index=False))
    return t


# --------------------------------------------------------------------------
# 2. Are the three detector variants the same run?
# --------------------------------------------------------------------------

def detector_identity(root: Path, say: Report) -> pd.DataFrame:
    say.rule("2. DETECTOR VARIANTS: ARE DPM / FRCNN / SDP DISTINCT RUNS?")
    say("  Under private detections the pipeline runs its own detector on the")
    say("  images, so the three public-detection variants of a scene should be")
    say("  the same run. If so, the effective sample size is 7, not 21.")
    say("")
    rows = []
    for f in sorted(root.glob("MOT-ALL/**/ALL_SEQUENCES_metrics.csv")):
        d = pd.read_csv(f)
        if "base_seq" not in d.columns or "sequence" not in d.columns:
            continue
        data = d[~d.sequence.astype(str).str.startswith(("MEAN", "AVERAGE"))]
        if data.empty or data.base_seq.nunique() == len(data):
            continue
        cols = [c for c in ("HOTA", "MOTA", "IDF1", "IDs") if c in data]
        nuniq = data.groupby("base_seq")[cols].nunique()
        identical = bool((nuniq == 1).all().all())
        rows.append({
            "run": rel(f.parent, root),
            "rows": len(data),
            "uniq_scenes": int(data.base_seq.nunique()),
            "variants_identical": identical,
        })
    t = pd.DataFrame(rows)
    if t.empty:
        say("  no multi-variant runs found")
        return t
    n_id = int(t.variants_identical.sum())
    say(f"  {n_id} of {len(t)} multi-variant runs have byte-equal detector variants")
    differ = t[~t.variants_identical]
    if len(differ):
        say("")
        say("  runs where the variants genuinely DIFFER (expected only for public detections):")
        for r in differ.run:
            say(f"    {r}")
    say("")
    say(f"  VERDICT: effective n for MOT17 training comparisons = "
        f"{int(t.uniq_scenes.mode().iloc[0])}, not {int(t.rows.mode().iloc[0])}")
    return t


# --------------------------------------------------------------------------
# 3. Table 7
# --------------------------------------------------------------------------

def check_table7(metrics: pd.DataFrame, say: Report) -> pd.DataFrame:
    say.rule("3. MANUSCRIPT TABLE 7 (COMPONENT STUDY) vs THE RUN ARTIFACTS")
    idx = metrics.set_index("run")
    rows = []
    for variant, (hota, idf1, assa, ids) in TABLE7.items():
        run = TABLE7_RUNS[variant]
        rec = {"variant": variant, "run": run}
        if run not in idx.index:
            rec["status"] = "RUN NOT FOUND"
            rows.append(rec)
            continue
        r = idx.loc[run]
        got = (r.get("HOTA"), r.get("IDF1"), r.get("AssA"), r.get("IDs_mean_per_row"))
        want = (hota, idf1, assa, ids)
        diffs = []
        for name, w, g in zip(("HOTA", "IDF1", "AssA", "IDs"), want, got):
            rec[f"paper_{name}"] = w
            rec[f"found_{name}"] = g
            if g is None or pd.isna(g) or abs(float(g) - w) > (TOL if name != "IDs" else 0.06):
                diffs.append(f"{name} {w} vs {g}")
        rec["status"] = "OK" if not diffs else "MISMATCH: " + "; ".join(diffs)
        rows.append(rec)

    t = pd.DataFrame(rows)
    ok = (t.status == "OK").sum()
    say(f"  {ok} of {len(t)} rows reconcile")
    say("")
    for _, r in t.iterrows():
        mark = "OK  " if r.status == "OK" else "FAIL"
        say(f"  [{mark}] {r.variant:<33} {r.status if r.status != 'OK' else ''}")
    return t


# --------------------------------------------------------------------------
# 4. Audit-event counts
# --------------------------------------------------------------------------

def categorize(reason: str) -> str:
    for cat in REASON_CATEGORIES:
        if reason.startswith(cat):
            return cat
    return "other:" + reason.split()[0].split("=")[0]


def audit_events(root: Path, say: Report) -> pd.DataFrame:
    say.rule("4. AUDIT-EVENT COUNTS, MOT17 TRAINING SCENES")
    say("  Section 4.4 states, over the seven MOT17 training scenes:")
    for k, v in AUDIT_CLAIM.items():
        say(f"    {k:<28} {v:>6,}")
    say(f"    {'TOTAL':<28} {sum(AUDIT_CLAIM.values()):>6,}")
    say("")
    say("  Counting the same categories in every MOT17 training-scene run.")
    say("  Only the -DPM variant is read, since the variants are identical.")
    say("")

    runs: dict[Path, list[Path]] = collections.defaultdict(list)
    for f in root.glob("MOT-ALL/**/*-DPM_ids_events.csv"):
        if f.name.split("-DPM")[0] in MOT17_TRAIN:
            runs[f.parent].append(f)

    rows = []
    for d, files in runs.items():
        if len(files) != len(MOT17_TRAIN):
            continue
        c: collections.Counter = collections.Counter()
        for f in files:
            for reason in pd.read_csv(f, usecols=["reason"]).reason.astype(str):
                c[categorize(reason)] += 1
        rec = {"run": rel(d, root)}
        rec.update({k: c.get(k, 0) for k in AUDIT_CLAIM})
        rec["TOTAL"] = sum(rec[k] for k in AUDIT_CLAIM)
        rec["categories_matching_paper"] = sum(
            1 for k in AUDIT_CLAIM if rec[k] == AUDIT_CLAIM[k])
        rows.append(rec)

    t = (pd.DataFrame(rows)
         .sort_values("categories_matching_paper", ascending=False)
         .reset_index(drop=True))
    if t.empty:
        say("  no complete MOT17 training-scene runs found")
        return t

    say(t.to_string(index=False))
    say("")
    exact = t[t.categories_matching_paper == len(AUDIT_CLAIM)]
    if exact.empty:
        say("  !! NO run reproduces the published counts.")
    else:
        say("  Runs reproducing all six published counts:")
        for r in exact.run:
            say(f"    {r}")
        say("")
        submitted = TABLE7_RUNS["+ box suppression (submitted)"]
        sub = t[t.run == submitted]
        if len(sub) and submitted not in set(exact.run):
            s = sub.iloc[0]
            say(f"  !! The SUBMITTED pipeline ({submitted}) does not match.")
            say(f"     Its counts: " + ", ".join(
                f"{k}={s[k]:,}" for k in AUDIT_CLAIM) + f", TOTAL={s.TOTAL:,}")
            say("     The published figures describe a different configuration.")
    return t


# --------------------------------------------------------------------------
# 5. Disambiguation margin
# --------------------------------------------------------------------------

DELTA_RE = re.compile(r"margin=[0-9.]+<([0-9.]+)")


def delta_values(root: Path, say: Report) -> pd.DataFrame:
    say.rule("5. DISAMBIGUATION MARGIN delta, FROZEN DANCETRACK TEST SUBMISSION")
    say("  Table 6 lists delta as a single frozen scalar. Each margin refusal")
    say("  records the threshold it was tested against, so delta is readable")
    say("  per sequence directly from the log.")
    say("")

    results_dir = None
    for g in DANCE_FROZEN_GLOBS:
        p = root / g
        if p.is_dir():
            results_dir = p
            break
    if results_dir is None:
        say("  !! frozen DanceTrack submission not found")
        return pd.DataFrame()

    say(f"  reading {rel(results_dir, root)}")
    rows = []
    for f in sorted(results_dir.glob("*_ids_events.csv")):
        seq = f.name.split("_ids")[0]
        seen: collections.Counter = collections.Counter()
        for reason in pd.read_csv(f, usecols=["reason"]).reason.astype(str):
            m = DELTA_RE.search(reason)
            if m:
                seen[m.group(1)] += 1
        if seen:
            rows.append({
                "sequence": seq,
                "deltas": ",".join(sorted(seen)),
                "n_distinct": len(seen),
                "n_refusals": sum(seen.values()),
            })

    t = pd.DataFrame(rows)
    if t.empty:
        say("  no margin refusals recorded")
        return t

    allv: collections.Counter = collections.Counter()
    for _, r in t.iterrows():
        for v in r.deltas.split(","):
            allv[v] += 1

    say("")
    say(f"  sequences with margin refusals : {len(t)}")
    say(f"  distinct delta values observed : {sorted(allv)}")
    say("")
    say("  delta value -> number of sequences using it")
    for v in sorted(allv):
        say(f"    {v:<6} {allv[v]:>3}")
    say("")
    mixed = t[t.n_distinct > 1]
    if len(mixed):
        say(f"  !! delta CHANGES WITHIN a sequence in {len(mixed)} sequences:")
        for _, r in mixed.iterrows():
            say(f"       {r.sequence}  ({r.deltas})")
    if len(allv) > 1:
        say("")
        say("  VERDICT: delta is not a single frozen scalar. Table 6 cannot")
        say("  report it as one without stating the rule that varies it.")
    return t


# --------------------------------------------------------------------------
# 6. Worked examples
# --------------------------------------------------------------------------

def worked_examples(root: Path, say: Report) -> None:
    say.rule("6. TABLE 2 WORKED EXAMPLES")
    results_dir = None
    for g in DANCE_FROZEN_GLOBS:
        p = root / g
        if p.is_dir():
            results_dir = p
            break
    if results_dir is None:
        say("  !! frozen DanceTrack submission not found")
        return

    for seq, frame, track, sim, margin, stage in WORKED_EXAMPLES:
        f = results_dir / f"{seq}_ids_events.csv"
        say("")
        say(f"  {seq} frame {frame}, track {track}")
        if not f.exists():
            say("    !! log not found")
            continue
        d = pd.read_csv(f)
        hit = d[(d.frame == frame) & ((d.old_id == track) | (d.new_id == track))]
        if hit.empty:
            say("    !! NO EVENT RECORDED AT THIS FRAME FOR THIS TRACK")
            continue
        r = hit.iloc[0]
        say(f"    stage={r.stage}  old_id={r.old_id} new_id={r.new_id}")
        say(f"    reason={r.reason}")
        say(f"    cosine_sim={r.cosine_sim}  cosine_margin={r.cosine_margin}")
        checks = []
        if stage and str(r.stage) != stage:
            checks.append(f"stage {stage} vs {r.stage}")
        if sim is not None and abs(float(r.cosine_sim) - sim) > 0.001:
            checks.append(f"sim {sim} vs {r.cosine_sim}")
        if margin is not None and abs(float(r.cosine_margin) - margin) > 0.001:
            checks.append(f"margin {margin} vs {r.cosine_margin}")
        say(f"    -> {'OK, matches the caption' if not checks else 'MISMATCH: ' + '; '.join(checks)}")


# --------------------------------------------------------------------------
# 7. Duplicate runs
# --------------------------------------------------------------------------

def duplicates(metrics: pd.DataFrame, say: Report) -> pd.DataFrame:
    say.rule("7. DUPLICATE AND DEGENERATE RUNS")
    key = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDs_total"]
    key = [k for k in key if k in metrics.columns]
    full = metrics.dropna(subset=key)
    groups = full.groupby(key).run.apply(list)
    dups = [g for g in groups if len(g) > 1]

    rows = []
    if dups:
        say("  Runs whose aggregate metrics are identical to another run:")
        for g in dups:
            say("")
            for r in g:
                say(f"    {r}")
            rows.append({"group": " | ".join(g), "n": len(g)})
    else:
        say("  no duplicate runs")

    say("")
    bad = metrics[(metrics.get("MOTA", pd.Series(dtype=float)) < 0)
                  | (metrics.get("HOTA", pd.Series(dtype=float)) < 20)]
    if len(bad):
        say("  Runs that look degenerate (negative MOTA or HOTA < 20):")
        for _, r in bad.iterrows():
            say(f"    {r.run}   HOTA={r.get('HOTA')}  MOTA={r.get('MOTA')}")
        say("  These cannot serve as baselines in any comparison.")
    else:
        say("  no degenerate runs")

    say("")
    partial = metrics[(metrics.rows < 21) & (metrics.rows > 0)]
    if len(partial):
        say("  Incomplete runs (fewer than 21 sequence rows), i.e. abandoned early:")
        for _, r in partial.iterrows():
            say(f"    {r.run}   rows={r.rows}")
        say("  These explain the gaps in the v-numbering in Table 7.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="path to the daat-work tree (contains MOT-ALL/)")
    ap.add_argument("--out", default="reconcile_out", help="directory for CSV outputs")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not (root / "MOT-ALL").is_dir():
        print(f"error: {root} does not contain a MOT-ALL/ directory", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 200)

    say = Report()
    say(f"reconcile.py   root={root}")
    say(f"               out ={out}")

    metrics = load_metrics(root, say)
    metrics.to_csv(out / "01_run_metrics.csv", index=False)

    det = detector_identity(root, say)
    det.to_csv(out / "02_detector_identity.csv", index=False)

    t7 = check_table7(metrics, say)
    t7.to_csv(out / "03_table7_check.csv", index=False)

    ev = audit_events(root, say)
    ev.to_csv(out / "04_audit_events.csv", index=False)

    dv = delta_values(root, say)
    dv.to_csv(out / "05_delta_values.csv", index=False)

    worked_examples(root, say)

    dup = duplicates(metrics, say)
    dup.to_csv(out / "06_duplicates.csv", index=False)

    say.rule("SUMMARY")
    say(f"  runs with metrics        : {len(metrics)}")
    if len(t7):
        say(f"  Table 7 rows reconciling : {(t7.status == 'OK').sum()} of {len(t7)}")
    if len(ev):
        exact = (ev.categories_matching_paper == len(AUDIT_CLAIM)).sum()
        say(f"  runs matching the audit-event counts : {exact}")
    if len(dv):
        n = len({v for row in dv.deltas for v in row.split(",")})
        say(f"  distinct delta values in the frozen submission : {n}")
    say("")
    say(f"  CSV outputs written to {out}")

    say.save(out / "report.txt")
    print(f"\nreport.txt written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

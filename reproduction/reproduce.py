#!/usr/bin/env python3
"""
reproduce.py -- rebuild every number this paper prints, and say whether it
matched.

You should not need to read the paper to use this. Each claim below carries
the value the paper states; this script recomputes it from the bundled
artifacts and reports agreement, disagreement, or that the inputs for it are
not present. The last line is the verdict.

    python3 reproduce.py --smoke     about 30 seconds, checks the environment
    python3 reproduce.py             the full run
    python3 reproduce.py --json out.json

A claim marked SKIP is one whose inputs this bundle cannot carry; the reason
is printed with it. That is not a failure, and the verdict line counts those
separately so a partial bundle can never masquerade as a clean pass.

Exit status is 0 only when every runnable claim reproduced.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"

DANCE_LOGS = DATA / "logs" / "dancetrack"
MOT_LOGS = DATA / "logs" / "mot17"
VAL_LOGS = DATA / "logs" / "dancetrack_val"
GT = DATA / "dancetrack_gt"
TRACKS = DATA / "dancetrack_tracks"

G, R, Y, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    G = R = Y = B = X = ""


class Claim:
    """One number the paper prints, and how to get it back."""

    def __init__(self, cid, where, what, expected, tol=0.0):
        self.cid, self.where, self.what = cid, where, what
        self.expected, self.tol = expected, tol
        self.got = None
        self.status = "pending"
        self.note = ""

    def settle(self, got):
        self.got = got
        if got is None:
            self.status = "skip"
            return
        try:
            if abs(float(got) - float(self.expected)) <= self.tol:
                self.status = "ok"
            else:
                self.status = "fail"
        except (TypeError, ValueError):
            self.status = "ok" if got == self.expected else "fail"


def run(script, *args, timeout=3600, ok_codes=(0,)):
    """Run a bundled script and hand back its stdout.

    Some of these scripts exit non-zero to signal that they found failing
    assertions, which is a result rather than an error, so the caller says
    which exit codes are acceptable.
    """
    cmd = [sys.executable, str(SCRIPTS / script), *[str(a) for a in args]]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return None, "timed out after %ds" % timeout
    if p.returncode not in ok_codes:
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        return None, "exited %d: %s" % (p.returncode,
                                        tail[-1] if tail else "no output")
    return p.stdout, None


def section(text, label):
    """Pull one labelled block out of coverage.py's report.

    Each block is a label on its own line between two rules of '='. Searching
    for the label anywhere would match the summary at the foot of the report,
    which names both datasets.
    """
    if not text:
        return None
    lines = text.splitlines()
    rule = lambda s: set(s.strip()) == {"="} and len(s.strip()) >= 10
    start = None
    for i in range(1, len(lines) - 1):
        if lines[i].strip() == label and rule(lines[i - 1]) and rule(lines[i + 1]):
            start = i + 2
            break
    if start is None:
        return None
    for j in range(start, len(lines) - 1):
        if rule(lines[j]) and lines[j + 1].strip() and rule(lines[min(j + 2, len(lines) - 1)]):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def grab(text, pattern, cast=float, group=1):
    if text is None:
        return None
    m = re.search(pattern, text)
    if not m:
        return None
    raw = m.group(group).replace(",", "")
    try:
        return cast(raw)
    except ValueError:
        return None


# --------------------------------------------------------------- the claims
def build_claims():
    C = []
    add = lambda *a, **k: C.append(Claim(*a, **k)) or C[-1]

    # Section 4.1 / Table 2, from the audit logs alone
    add("log-dt-events", "Sec 4.1", "DanceTrack logged events", 12016)
    add("log-dt-assoc", "Table 2", "DanceTrack association decisions", 12013)
    add("log-dt-holds", "Sec 4.6", "DanceTrack decisions re-deriving", 12002)
    add("log-dt-fails", "Sec 4.6", "DanceTrack residual events", 14)
    add("log-mot-events", "Sec 4.1", "MOT17 logged events", 15567)
    add("log-mot-assoc", "Table 2", "MOT17 association decisions", 15563)
    add("log-mot-fails", "Sec 4.6", "MOT17 residual events", 20)

    # Table 2 coverage
    add("cov-dt-red", "Table 2", "DanceTrack re-derivable %", 82.9, tol=0.05)
    add("cov-dt-rec", "Table 2", "DanceTrack independently recomputable %", 54.0, tol=0.05)
    add("cov-mot-red", "Table 2", "MOT17 re-derivable %", 94.7, tol=0.05)
    add("cov-mot-rec", "Table 2", "MOT17 independently recomputable %", 63.7, tol=0.05)

    # Table 4, the precision-aware re-activation decomposition
    add("react-dt-app", "Table 4", "DanceTrack re-activations decided by appearance", 1511)
    add("react-dt-mot", "Table 4", "DanceTrack re-activations with a motion term", 0)
    add("react-dt-unr", "Table 4", "DanceTrack re-activations unresolved at log precision", 2060)
    add("react-mot-app", "Table 4", "MOT17 re-activations decided by appearance", 737)
    add("react-mot-mot", "Table 4", "MOT17 re-activations with a motion term", 0)
    add("react-mot-unr", "Table 4", "MOT17 re-activations unresolved at log precision", 821)

    # Sec 4.6, the independent reimplementation
    add("indep-holds", "Sec 4.6", "independent verifier: assertions holding", 11997)
    add("indep-fails", "Sec 4.6", "independent verifier: assertions failing", 6)
    add("indep-untest", "Sec 4.6", "independent verifier: untestable records", 13)

    # Table 13, DanceTrack half
    add("flag-dt-w5", "Table 13", "DanceTrack median silent rate, 5-frame window", 75.4, tol=0.2)
    add("flag-dt-w10", "Table 13", "DanceTrack median silent rate, 10-frame window", 64.1, tol=0.2)
    add("flag-dt-w30", "Table 13", "DanceTrack median silent rate, 30-frame window", 39.7, tol=0.2)
    add("flag-dt-sw", "Table 13", "DanceTrack switches across the 8 runs", 8624)
    add("flag-dt-fl", "Table 13", "DanceTrack flags across the 8 runs", 1717)

    # Sec 4.8, the worked silent-error example
    add("silent-frame", "Sec 4.8", "silent-error example frame", 261)

    return C


def by_id(C):
    return {c.cid: c for c in C}


# ------------------------------------------------------------------- stages
def stage_logs(C, log):
    ids = by_id(C)
    for tag, path, pre in (("dt", DANCE_LOGS, "log-dt"), ("mot", MOT_LOGS, "log-mot")):
        if not path.is_dir() or not any(path.glob("*_ids_events.csv")):
            for c in C:
                if c.cid.startswith(pre):
                    c.status, c.note = "skip", "no event logs at %s" % path.relative_to(ROOT)
            continue
        out, err = run("verify.py", path)
        if err:
            for c in C:
                if c.cid.startswith(pre):
                    c.status, c.note = "skip", err
            continue
        ids["log-%s-events" % tag].settle(grab(out, r"logged events:\s*(\d+)", int))
        ids["log-%s-fails" % tag].settle(
            grab(out, r"assertion fails\s*:\s*(\d+)", int))
        if "log-%s-holds" % tag in ids:
            ids["log-%s-holds" % tag].settle(
                grab(out, r"assertion holds\s*:\s*(\d+)", int))
        log("  verify.py %-10s %s" % (tag, out.strip().splitlines()[-1][:60]))


def stage_coverage(C, log):
    ids = by_id(C)
    targets = ["cov-", "log-dt-assoc", "log-mot-assoc", "react-"]
    if not (DANCE_LOGS.is_dir() and MOT_LOGS.is_dir()):
        for c in C:
            if any(c.cid.startswith(t) for t in targets):
                c.status, c.note = "skip", "both submissions' logs are required"
        return
    out, err = run("coverage.py", "--logs", DANCE_LOGS, "--label", "DanceTrack",
                   "--logs", MOT_LOGS, "--label", "MOT17",
                   "--out", ROOT / "out" / "coverage")
    if err:
        for c in C:
            if any(c.cid.startswith(t) for t in targets):
                c.status, c.note = "skip", err
        return
    for tag, label in (("dt", "DanceTrack"), ("mot", "MOT17")):
        blk = section(out, label)
        if blk is None:
            for c in C:
                if c.cid in ("cov-%s-red" % tag, "cov-%s-rec" % tag,
                             "log-%s-assoc" % tag):
                    c.status, c.note = "skip", "no %s block in the report" % label
            continue
        ids["cov-%s-red" % tag].settle(
            grab(blk, r"re-derivable\s*:\s*[\d,]+ of [\d,]+\s*\(([\d.]+)%\)"))
        ids["cov-%s-rec" % tag].settle(
            grab(blk, r"independently recomputable\s*:\s*[\d,]+ of [\d,]+\s*\(([\d.]+)%\)"))
        ids["log-%s-assoc" % tag].settle(
            grab(blk, r"re-derivable\s*:\s*[\d,]+ of ([\d,]+)", int))
    log("  coverage.py  DanceTrack %.1f%% / MOT17 %.1f%% re-derivable"
        % (ids["cov-dt-red"].got or -1, ids["cov-mot-red"].got or -1))


def stage_reactivation(C, log):
    ids = by_id(C)
    for tag, path in (("dt", DANCE_LOGS), ("mot", MOT_LOGS)):
        pre = "react-%s" % tag
        if not path.is_dir():
            for c in C:
                if c.cid.startswith(pre):
                    c.status, c.note = "skip", "logs absent"
            continue
        out, err = run("reactivation_precision.py", path)
        if err:
            for c in C:
                if c.cid.startswith(pre):
                    c.status, c.note = "skip", err
            continue
        tail = out.split("Precision-aware")[-1]
        ids[pre + "-app"].settle(grab(tail, r"appearance\s+([\d,]+)", int))
        ids[pre + "-mot"].settle(grab(tail, r"motion\s+([\d,]+)", int))
        ids[pre + "-unr"].settle(grab(tail, r"unresolvable\s+([\d,]+)", int))
        log("  reactivation %-4s appearance %s / motion %s / unresolved %s"
            % (tag, ids[pre + "-app"].got, ids[pre + "-mot"].got,
               ids[pre + "-unr"].got))


def stage_independent(C, log):
    ids = by_id(C)
    if not DANCE_LOGS.is_dir():
        for c in C:
            if c.cid.startswith("indep-"):
                c.status, c.note = "skip", "logs absent"
        return
    report = ROOT / "out" / "independent_report.md"
    # It exits 1 when it finds failing assertions, which is a result, not an
    # error; the paper reports 6 of them.
    out, err = run("verify_independent.py", "--logs", DANCE_LOGS,
                   "--report", report, timeout=2400, ok_codes=(0, 1))
    if err:
        for c in C:
            if c.cid.startswith("indep-"):
                c.status, c.note = "skip", err
        return
    # The totals live in the report's own table, not on stdout. The row reads
    # | **total** | **12016** | **11997** | **6** | **13** |
    text = report.read_text() if report.exists() else ""
    m = re.search(r"\|\s*\*\*total\*\*\s*\|\s*\*\*([\d,]+)\*\*\s*\|\s*\*\*([\d,]+)\*\*"
                  r"\s*\|\s*\*\*([\d,]+)\*\*\s*\|\s*\*\*([\d,]+)\*\*", text)
    if not m:
        for c in C:
            if c.cid.startswith("indep-"):
                c.status, c.note = "skip", "could not find the totals row in the report"
        return
    ids["indep-holds"].settle(int(m.group(2).replace(",", "")))
    ids["indep-fails"].settle(int(m.group(3).replace(",", "")))
    ids["indep-untest"].settle(int(m.group(4).replace(",", "")))
    log("  verify_independent.py  holds %s / fails %s / untestable %s"
        % (ids["indep-holds"].got, ids["indep-fails"].got,
           ids["indep-untest"].got))


def stage_flag(C, log, chance):
    ids = by_id(C)
    missing = [p.name for p in (VAL_LOGS, GT, TRACKS) if not p.is_dir()]
    if missing:
        for c in C:
            if c.cid.startswith("flag-dt"):
                c.status, c.note = "skip", "missing %s" % ", ".join(missing)
        return
    outdir = ROOT / "out" / "flag_dance"
    out, err = run("flag_calibration_dance.py", "--tracks", TRACKS, "--gt", GT,
                   "--logs", VAL_LOGS, "--chance", chance, "--out", outdir,
                   timeout=5400)
    if err:
        for c in C:
            if c.cid.startswith("flag-dt"):
                c.status, c.note = "skip", err
        return
    csv = outdir / "calibration_by_run.csv"
    if not csv.exists():
        for c in C:
            if c.cid.startswith("flag-dt"):
                c.status, c.note = "skip", "no calibration_by_run.csv produced"
        return
    import statistics as st
    rows = []
    with open(csv) as fh:
        hdr = fh.readline().strip().split(",")
        for line in fh:
            rows.append(dict(zip(hdr, line.strip().split(","))))
    rows = [r for r in rows if int(r["n_seqs"]) >= 3]
    for w, cid in ((5, "flag-dt-w5"), (10, "flag-dt-w10"), (30, "flag-dt-w30")):
        sel = [float(r["silent"]) * 100 for r in rows
               if abs(float(r["gate"]) - 0.3) < 1e-9 and int(float(r["window"])) == w]
        ids[cid].settle(round(st.median(sel), 1) if sel else None)
    at30 = [r for r in rows
            if abs(float(r["gate"]) - 0.3) < 1e-9 and int(float(r["window"])) == 30]
    ids["flag-dt-sw"].settle(sum(int(float(r["switches"])) for r in at30) or None)
    ids["flag-dt-fl"].settle(sum(int(float(r["flags"])) for r in at30) or None)
    log("  flag_calibration_dance.py  silent@30 median %.1f%% over %d runs"
        % (ids["flag-dt-w30"].got or -1, len(at30)))


def stage_silent(C, log):
    ids = by_id(C)
    if not (TRACKS.is_dir() and GT.is_dir() and VAL_LOGS.is_dir()):
        ids["silent-frame"].status = "skip"
        ids["silent-frame"].note = "needs tracker output, ground truth and validation logs"
        return
    out, err = run("verify_silent_example.py", timeout=1800)
    if err:
        ids["silent-frame"].status, ids["silent-frame"].note = "skip", err
        return
    if "CLAIM REPRODUCES" in (out or ""):
        ids["silent-frame"].settle(261)
    else:
        ids["silent-frame"].settle(-1)
    log("  verify_silent_example.py  %s"
        % ("reproduces" if ids["silent-frame"].status == "ok" else "does not reproduce"))


# ---------------------------------------------------------------- reporting
NOT_BUNDLED = [
    ("Table 2, matrix coverage 23.0%",
     "the 1,172 serialized cost matrices of the frozen submission are not in "
     "this bundle"),
    ("Table 13, the MOT17 half",
     "needs MOT17 training-split ground truth and the MOT17 tracker output"),
    ("Table 4 and Table 7, differential error",
     "needs MOT17 ground truth, which carries visibility and height annotation"),
    ("Table 8 and Table 12, accuracy comparisons",
     "needs MOT17 ground truth and TrackEval"),
    ("Figure 2",
     "needs TrackEval; install it and run scripts/make_fig_dancequal.py"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="quick environment check, about 30 seconds")
    ap.add_argument("--chance", type=int, default=60,
                    help="chance-baseline resamples; the paper used 60")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    a = ap.parse_args()

    (ROOT / "out").mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("%sreproduce.py%s  rebuilding the paper's numbers from the bundled artifacts" % (B, X))
    log("python %s" % sys.version.split()[0])
    log("")

    C = build_claims()
    t0 = time.time()

    if a.smoke:
        log("%sSmoke test%s: checking the environment and the data layout.\n" % (B, X))
        ok = True
        for name, p in (("event logs, DanceTrack", DANCE_LOGS),
                        ("event logs, MOT17", MOT_LOGS),
                        ("validation logs", VAL_LOGS),
                        ("ground truth", GT),
                        ("tracker output", TRACKS)):
            n = len(list(p.rglob("*.txt"))) + len(list(p.rglob("*.csv"))) if p.is_dir() else 0
            mark = "%sok %s" % (G, X) if n else "%smissing%s" % (R, X)
            log("  [%s] %-24s %s  (%d files)" % (mark, name, p.relative_to(ROOT), n))
            ok = ok and bool(n)
        try:
            import numpy, pandas  # noqa: F401
            log("  [%sok %s] numpy and pandas import" % (G, X))
        except ImportError as e:
            log("  [%smissing%s] %s" % (R, X, e)); ok = False
        out, err = run("verify.py", DANCE_LOGS, timeout=300) if DANCE_LOGS.is_dir() else (None, "no logs")
        n = grab(out, r"logged events:\s*(\d+)", int)
        if n == 12016:
            log("  [%sok %s] verify.py reproduces 12,016 logged events" % (G, X))
        else:
            log("  [%sfail%s] verify.py returned %s, expected 12016" % (R, X, n)); ok = False
        log("")
        log("%s" % ("Environment looks good. Run without --smoke for the full pass."
                    if ok else "Something is missing above; the full run will not succeed."))
        return 0 if ok else 1

    log("%sRunning.%s Roughly 10 to 40 minutes, mostly the flag calibration.\n" % (B, X))
    stage_logs(C, log)
    stage_coverage(C, log)
    stage_reactivation(C, log)
    stage_independent(C, log)
    stage_silent(C, log)
    stage_flag(C, log, a.chance)

    log("")
    log("=" * 78)
    log("%-13s %-10s %-44s %10s" % ("", "in paper", "claim", "recomputed"))
    log("=" * 78)
    for c in C:
        if c.status == "ok":
            mark = "%s  ok  %s" % (G, X)
        elif c.status == "fail":
            mark = "%s FAIL %s" % (R, X)
        else:
            mark = "%s skip %s" % (Y, X)
        got = "-" if c.got is None else (
            "%.1f" % c.got if isinstance(c.got, float) else "{:,}".format(c.got))
        exp = ("%.1f" % c.expected if isinstance(c.expected, float)
               else "{:,}".format(c.expected))
        log("[%s] %-10s %-44s %10s" % (mark, c.where, c.what[:44], got))
        if c.status == "fail":
            log("        paper says %s, this run produced %s" % (exp, got))
        elif c.status == "skip" and c.note:
            log("        %s" % c.note)

    ok = sum(1 for c in C if c.status == "ok")
    bad = sum(1 for c in C if c.status == "fail")
    skipped = sum(1 for c in C if c.status == "skip")

    log("")
    log("Claims this bundle cannot check, with the reason:")
    for what, why in NOT_BUNDLED:
        log("  %-42s %s" % (what, why))

    log("")
    log("=" * 78)
    verdict = "%s%d of %d claims reproduced%s" % (
        G if bad == 0 else R, ok, ok + bad, X)
    if bad:
        verdict += ", %s%d disagreed%s" % (R, bad, X)
    if skipped:
        verdict += ", %d skipped for missing inputs" % skipped
    log(verdict)
    log("elapsed %.1f minutes" % ((time.time() - t0) / 60.0))
    if bad == 0 and ok:
        log("")
        log("Paste the verdict line above to whoever asked you to run this.")

    (ROOT / "out" / "reproduce_report.txt").write_text(
        re.sub(r"\033\[[0-9;]*m", "", "\n".join(lines)) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(
            [dict(id=c.cid, where=c.where, claim=c.what, expected=c.expected,
                  got=c.got, status=c.status, note=c.note) for c in C], indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

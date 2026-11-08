#!/usr/bin/env python3
"""
verify_independent.py -- independent verifier for the DAAT association-layer
audit log, written from SPEC.md v1.0 (frozen 7 November 2026) alone.

Question answered per record: does the recorded outcome follow from the
operands recorded alongside it, under the rule the record names?

Three outcomes per record, never two:
    HOLDS      the assertion follows from the logged operands
    FAILS      the logged operands contradict the assertion
    UNTESTABLE an operand the check needs is absent from the record

UNTESTABLE is never counted as a pass.

Two further buckets are reported separately and are not pass/fail:
    COMPOSITE  SPEC.md 4.7 says the composite cost cannot be decomposed
    ADVISORY   checks implied by SPEC.md but not stated as an assertion in S6

Standard library only.  Usage:
    python3 verify_independent.py [--logs DIR] [--report FILE] [--max-items N]
"""

import argparse
import ast
import collections
import csv
import glob
import math
import os
import re
import sys

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
# SPEC.md Section 2: numeric columns are printed at four decimal places,
# operands embedded in the reason string at three.  A printed value v
# therefore stands for a true value in [v - u, v + u] with u the half-ulp of
# the print width:
U_COL = 0.5e-4      # half-ulp of a 4-decimal print
U_REASON = 0.5e-3   # half-ulp of a 3-decimal print
U_EXACT = 0.0       # value known exactly (config constant, or full precision)
U_FLOAT = 1e-7      # float32 embedding round-off, for all_sims_top5 values

# Uncertainties add when two printed quantities are compared, so the effective
# tolerance is derived per comparison rather than being one global constant.
# The widest comparison in this verifier (3-decimal operand vs 4-decimal
# column) carries 5.5e-4; the narrowest (full-precision similarity vs an exact
# configuration constant) carries ~1e-7.

# SPEC.md 4.3: the disambiguation margin actually in force takes one of these
# exact values.  A reason-string right-hand side that rounds to one of them is
# therefore treated as that exact constant rather than as a 3-decimal print.
DELTA_SET = (0.03, 0.04, 0.05, 0.07)
DELTA_EARLY = 0.05          # SPEC.md 4.3, first 30 frames
EARLY_FRAME_LIMIT = 30
# SPEC.md 4.3: delta = (1-t)*0.07 + t*0.03 with t = clip(...,0,1) per 4.1, so
# whatever t was, the margin in force lies in this closed range.  Used to
# bound `revived` where the per-sequence value cannot be recovered.
DELTA_RANGE = (0.03, 0.07)

THETA_CLAMP = (0.65, 0.80)  # SPEC.md 4.2: min(max(s_min,0.65),0.80)
COST_ACCEPT_FIRST = 0.80    # SPEC.md 4.6, first stage, "at or below"
COST_ACCEPT_SECOND = 0.50   # SPEC.md 4.6, second stage, "strictly below"
DET_THRESHOLD = 0.50        # SPEC.md 4.6, detection threshold
OCM_WEIGHT = 0.2            # SPEC.md 4.7 / 4.9
OCR_IOU_ASSUMED = 0.30      # NOT IN SPEC -- see report, advisory only

HOLDS, FAILS, UNTESTABLE = "holds", "fails", "untestable"

# SPEC.md Section 3: each stage pairs with exactly one reason keyword.
STAGE_REASON = {
    "3": "re_activate_from_lost",
    "4_dlow_second_assoc": "byte_second_assoc",
    "5": "unconfirmed_update",
    "5b_ocr_recovery": "ocr_last_obs_iou",
    "6_revive": "revived",
    "6_new": ("gate_failed", "margin_failed", "registry_empty_or_immature"),
}

# SPEC.md Section 3 documents these reason-string grammars.
DOCUMENTED_FORMS = {
    "gate_failed": re.compile(r"^gate_failed sim=([\d.]+)<theta=([\d.]+)$"),
    "margin_failed": re.compile(r"^margin_failed margin=([\d.]+)<([\d.]+)$"),
    "ocr_last_obs_iou": re.compile(r"^ocr_last_obs_iou iou=([\d.]+)$"),
}
# Grammars the data actually uses (see report section 4).
ACTUAL_FORMS = {
    "gate_failed": re.compile(r"^gate_failed sim=([\d.]+)<theta=([\d.]+)$"),
    "margin_failed": re.compile(r"^margin_failed margin=([\d.]+)<([\d.]+)$"),
    "ocr_last_obs_iou": re.compile(r"^ocr_last_obs_iou=([\d.]+) gap=([-\d.]+)$"),
    "byte_second_assoc": re.compile(r"^byte_second_assoc iou=([\d.]+) score=([\d.]+)$"),
}

APPEARANCE_COMPARISON_FIELDS = ("cosine_sim", "cosine_theta", "cosine_margin",
                                "all_sims_top5")


# ---------------------------------------------------------------------------
# Measured value: a number plus the uncertainty its print width implies
# ---------------------------------------------------------------------------
class M:
    __slots__ = ("v", "u", "src")

    def __init__(self, v, u, src):
        self.v, self.u, self.src = v, u, src

    def __repr__(self):
        return "%.6g" % self.v


def strictly_less(a, b):
    """Decide a.v < b.v given both uncertainties.

    Returns (outcome, slack) where outcome is HOLDS or FAILS and slack is the
    signed violation a-b.  'borderline' is True when the ordering cannot be
    resolved within the print precision; such records are counted as HOLDS
    (the operands do not demonstrably contradict the assertion) but are
    itemized in the report because an exact comparison would reject them.
    """
    tol = a.u + b.u
    d = a.v - b.v
    if d < -tol:
        return HOLDS, d, False
    if d > tol:
        return FAILS, d, False
    return HOLDS, d, True


def at_least(a, b):
    """Decide a.v >= b.v given both uncertainties."""
    tol = a.u + b.u
    d = a.v - b.v
    if d > tol:
        return HOLDS, d, False
    if d < -tol:
        return FAILS, d, False
    return HOLDS, d, True


def at_most(a, b):
    """Decide a.v <= b.v given both uncertainties."""
    tol = a.u + b.u
    d = a.v - b.v
    if d < -tol:
        return HOLDS, d, False
    if d > tol:
        return FAILS, d, False
    return HOLDS, d, True


def equal(a, b):
    tol = a.u + b.u
    d = a.v - b.v
    return (HOLDS if abs(d) <= tol else FAILS), d


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def num(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def col(row, name, u=U_COL):
    v = num(row.get(name))
    return None if v is None else M(v, u, name)


def parse_box(s):
    if not s or not s.strip():
        return None
    try:
        v = ast.literal_eval(s.strip())
    except (ValueError, SyntaxError):
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def parse_top5(s):
    """all_sims_top5 -> list of (identity, similarity) at full precision."""
    if not s or not s.strip():
        return None
    try:
        v = ast.literal_eval(s.strip())
    except (ValueError, SyntaxError):
        return None
    if not isinstance(v, (list, tuple)):
        return None
    out = []
    for item in v:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                out.append((item[0], float(item[1])))
            except (TypeError, ValueError):
                return None
        else:
            return None
    return out


def geom_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else None


def reason_keyword(reason):
    """SPEC.md Section 3 keyword.  Tolerates the keyword=value form the
    ocr records actually use (see report section 4)."""
    if not reason:
        return ""
    return reason.split()[0].split("=")[0]


def snap_delta(x):
    """Return (value, uncertainty) for a margin right-hand side."""
    for d in DELTA_SET:
        if abs(x - d) <= U_REASON:
            return M(d, U_EXACT, "delta(4.3 exact)")
    return M(x, U_REASON, "delta(reason 3dp)")


def snap_theta(x):
    """cosine_theta at a clamp boundary of 4.2 is an exact constant."""
    for t in THETA_CLAMP:
        if abs(x - t) <= U_COL:
            return M(t, U_EXACT, "cosine_theta(4.2 clamp)")
    return M(x, U_COL, "cosine_theta")


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
class Finding:
    def __init__(self, kind, path, frame, reason, detail, operands):
        self.kind = kind
        self.file = os.path.basename(path)
        self.frame = frame
        self.reason = reason
        self.detail = detail
        self.operands = operands

    def line(self):
        ops = "  ".join("%s=%s" % (k, v) for k, v in self.operands)
        return "`%s` frame %s | %s | %s | %s" % (
            self.file, self.frame, self.reason, self.detail, ops)


class Verifier:
    def __init__(self):
        self.counts = collections.defaultdict(
            lambda: collections.Counter())          # reason -> outcome counter
        self.untestable_why = collections.defaultdict(collections.Counter)
        self.failures = []
        self.borderline = []
        self.composite = collections.Counter()
        self.composite_items = []
        self.advisory = collections.defaultdict(list)
        self.advisory_n = collections.Counter()
        self.structural = collections.defaultdict(list)
        self.structural_n = collections.Counter()
        self.examined = 0
        self.notes = collections.Counter()
        self.delta_by_file = {}

    # -- bookkeeping ------------------------------------------------------
    def record(self, reason, outcome):
        self.counts[reason][outcome] += 1

    def fail(self, path, frame, reason, detail, operands):
        self.failures.append(Finding("fail", path, frame, reason, detail,
                                     operands))

    def border(self, path, frame, reason, detail, operands):
        self.borderline.append(Finding("borderline", path, frame, reason,
                                       detail, operands))

    def advise(self, tag, path, frame, reason, detail, operands):
        self.advisory_n[tag] += 1
        self.advisory[tag].append(Finding("advisory", path, frame, reason,
                                          detail, operands))

    def struct(self, tag, path, frame, reason, detail, operands):
        self.structural_n[tag] += 1
        self.structural[tag].append(Finding("structural", path, frame, reason,
                                            detail, operands))

    # -- pass 1: recover the per-sequence disambiguation margin -----------
    def recover_deltas(self, files):
        """SPEC.md 4.1 measures t once at frame 30, so 4.3's delta is one
        constant per sequence after frame 30.  margin_failed records print it
        as the reason right-hand side, so it can be read off the log itself
        and then used to test the `revived` records, whose reason string
        carries no operands."""
        for path in files:
            seen = set()
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if reason_keyword(row.get("reason", "")) != "margin_failed":
                        continue
                    fr = num(row.get("frame"))
                    if fr is None or fr <= EARLY_FRAME_LIMIT:
                        continue
                    m = ACTUAL_FORMS["margin_failed"].match(row["reason"].strip())
                    if m:
                        seen.add(snap_delta(float(m.group(2))).v)
            if len(seen) == 1:
                self.delta_by_file[path] = seen.pop()
            elif len(seen) > 1:
                self.delta_by_file[path] = None
                self.notes["delta_not_constant:" + os.path.basename(path)] += 1

    def feasible_delta(self, files):
        """Per sequence, the interval the true late-frame delta must lie in if
        every record in that file is faithful.

        Every `margin_failed` record requires delta > its margin; every
        `revived` record requires delta <= its margin (4.3 is one constant per
        sequence after frame 30, because 4.1 measures t once at frame 30).
        An empty interval proves the file cannot be explained by any single
        threshold, independently of what the log prints."""
        rows = []
        for path in files:
            lo, hi, printed = 0.0, 1.0, set()
            nmf = nrev = 0
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    kw = reason_keyword(row.get("reason", ""))
                    fr = num(row.get("frame"))
                    if fr is None or fr <= EARLY_FRAME_LIMIT:
                        continue
                    mg = self.margin_value(row)
                    if mg is None:
                        continue
                    if kw == "margin_failed":
                        m = ACTUAL_FORMS["margin_failed"].match(
                            row["reason"].strip())
                        if m:
                            printed.add(float(m.group(2)))
                        lo = max(lo, mg.v)
                        nmf += 1
                    elif kw == "revived":
                        hi = min(hi, mg.v)
                        nrev += 1
            if not printed:
                rows.append((path, None, lo, hi, nmf, nrev, None))
                continue
            p = min(printed)
            plo, phi = p - U_REASON, p + U_REASON
            L, H = max(lo, plo), min(hi, phi)
            rows.append((path, p, lo, hi, nmf, nrev, L <= H))
        return rows

    def delta_in_force(self, path, frame):
        """(M, why) -- the margin in force, or (None, reason-absent)."""
        if frame is not None and frame <= EARLY_FRAME_LIMIT:
            return M(DELTA_EARLY, U_EXACT, "delta(4.3 first-30)"), None
        d = self.delta_by_file.get(path)
        if d is None:
            return None, "delta_in_force (no margin_failed record in file)"
        return M(d, U_EXACT, "delta(recovered per-file)"), None

    # -- main -------------------------------------------------------------
    def run(self, files):
        self.recover_deltas(files)
        for path in files:
            prev_frame = None
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.examined += 1
                    prev_frame = self.check_row(path, row, prev_frame)

    def check_row(self, path, row, prev_frame):
        frame_raw = row.get("frame", "")
        frame = num(frame_raw)
        reason = (row.get("reason") or "").strip()
        kw = reason_keyword(reason)
        stage = (row.get("stage") or "").strip()

        self.check_invariants(path, frame_raw, reason, row, frame, prev_frame)
        self.check_stage_pairing(path, frame_raw, reason, stage, kw)
        self.check_reason_format(path, frame_raw, reason, kw)

        handler = {
            "gate_failed": self.check_gate_failed,
            "margin_failed": self.check_margin_failed,
            "revived": self.check_revived,
            "ocr_last_obs_iou": self.check_ocr,
            "byte_second_assoc": self.check_byte,
            "unconfirmed_update": self.check_unconfirmed,
            "re_activate_from_lost": self.check_reactivate,
            "registry_empty_or_immature": self.check_immature,
        }.get(kw)

        if handler is None:
            self.record("(unrecognized reason)", UNTESTABLE)
            self.untestable_why["(unrecognized reason)"]["reason keyword not in SPEC.md Section 3"] += 1
            self.struct("unknown_reason", path, frame_raw, reason,
                        "reason keyword not in SPEC.md Section 3", [])
        else:
            handler(path, frame_raw, frame, reason, row)
        return frame if frame is not None else prev_frame

    # -- Section 5 invariants (reported separately from the S6 assertions) --
    def check_invariants(self, path, frame_raw, reason, row, frame, prev_frame):
        ic = col(row, "iou_cost")
        io = col(row, "iou_overlap")
        fc = col(row, "fused_cost")
        cs = col(row, "cosine_sim")
        ct = col(row, "cosine_theta")
        cm = col(row, "cosine_margin")
        top5 = parse_top5(row.get("all_sims_top5"))

        # 1. iou_cost + iou_overlap = 1
        if ic and io and abs(ic.v + io.v - 1.0) > 2 * U_COL:
            self.struct("inv1_iou_sum", path, frame_raw, reason,
                        "iou_cost + iou_overlap != 1 (off by %+.5f)"
                        % (ic.v + io.v - 1.0),
                        [("iou_cost", ic.v), ("iou_overlap", io.v)])
        # 2. ranges
        for nm, mv in (("iou_cost", ic), ("fused_cost", fc), ("cosine_sim", cs)):
            if mv and not (-U_COL <= mv.v <= 1 + U_COL):
                self.struct("inv2_range", path, frame_raw, reason,
                            "%s outside [0,1]" % nm, [(nm, mv.v)])
        # 3. cosine_sim == top of all_sims_top5
        if top5 and cs and abs(top5[0][1] - cs.v) > U_COL + U_FLOAT:
            self.struct("inv3_sim_is_top", path, frame_raw, reason,
                        "cosine_sim != highest similarity in all_sims_top5",
                        [("cosine_sim", cs.v), ("top5[0]", top5[0][1])])
        # 4. cosine_margin == top - runner-up
        if top5 and len(top5) >= 2 and cm:
            d = top5[0][1] - top5[1][1]
            if abs(d - cm.v) > U_COL + 2 * U_FLOAT:
                self.struct("inv4_margin", path, frame_raw, reason,
                            "cosine_margin != top-minus-runner-up",
                            [("cosine_margin", cm.v), ("top5 diff", d)])
        # 5. all_sims_top5 descending
        if top5 and any(top5[i][1] < top5[i + 1][1] - U_FLOAT
                        for i in range(len(top5) - 1)):
            self.struct("inv5_sorted", path, frame_raw, reason,
                        "all_sims_top5 not sorted descending",
                        [("sims", [round(s, 6) for _, s in top5])])
        # 6. frame non-decreasing
        if frame is not None and prev_frame is not None and frame < prev_frame:
            self.struct("inv6_frame_order", path, frame_raw, reason,
                        "frame decreased within file",
                        [("frame", frame), ("previous", prev_frame)])
        # 7. theta in [0.65, 0.80]
        if ct and not (THETA_CLAMP[0] - U_COL <= ct.v <= THETA_CLAMP[1] + U_COL):
            self.struct("inv7_theta_range", path, frame_raw, reason,
                        "cosine_theta outside [0.65,0.80]",
                        [("cosine_theta", ct.v)])

        # Section 2's own definition of iou_cost: "1 minus the IoU overlap
        # between det_box and track_box".  Testable wherever both boxes and
        # iou_overlap are present.  Reported structurally, not as an S6
        # assertion, because S6 takes iou_cost/iou_overlap as given operands.
        db, tb = parse_box(row.get("det_box")), parse_box(row.get("track_box"))
        if db and tb and io:
            g = geom_iou(db, tb)
            if g is not None:
                # boxes print at 1 decimal; 0.01 is far wider than that implies
                if abs(g - io.v) > 0.01:
                    self.struct("sec2_box_iou_mismatch", path, frame_raw, reason,
                                "IoU(det_box,track_box)=%.4f but iou_overlap=%.4f (%+.4f)"
                                % (g, io.v, g - io.v),
                                [("det_box", db), ("track_box", tb),
                                 ("iou_overlap", io.v)])
                else:
                    self.structural_n["sec2_box_iou_agrees"] += 1

    def check_stage_pairing(self, path, frame_raw, reason, stage, kw):
        want = STAGE_REASON.get(stage)
        if want is None:
            self.struct("stage_unknown", path, frame_raw, reason,
                        "stage not in SPEC.md Section 3", [("stage", stage)])
        elif (kw != want) if isinstance(want, str) else (kw not in want):
            self.struct("stage_reason_mismatch", path, frame_raw, reason,
                        "stage %r does not pair with reason %r" % (stage, kw),
                        [("stage", stage), ("reason keyword", kw)])

    def check_reason_format(self, path, frame_raw, reason, kw):
        doc = DOCUMENTED_FORMS.get(kw)
        act = ACTUAL_FORMS.get(kw)
        if doc is not None and not doc.match(reason):
            self.struct("reason_format_undocumented", path, frame_raw, reason,
                        "reason string does not match the grammar in Section 3",
                        [("reason", reason)])
        if act is not None and not act.match(reason):
            self.struct("reason_format_unparsed", path, frame_raw, reason,
                        "reason string matches no known grammar",
                        [("reason", reason)])
        if doc is None and act is not None:
            self.structural_n["reason_operands_undocumented"] += 1

    # -- best similarity, preferring full precision ----------------------
    def best_sim(self, row):
        top5 = parse_top5(row.get("all_sims_top5"))
        if top5:
            return M(top5[0][1], U_FLOAT, "all_sims_top5[0]"), top5
        cs = col(row, "cosine_sim")
        return cs, None

    def margin_value(self, row):
        top5 = parse_top5(row.get("all_sims_top5"))
        if top5 and len(top5) >= 2:
            return M(top5[0][1] - top5[1][1], 2 * U_FLOAT,
                     "all_sims_top5 top-minus-runner-up")
        return col(row, "cosine_margin")

    # -- Section 6 assertions --------------------------------------------
    def check_gate_failed(self, path, frame_raw, frame, reason, row):
        """S6: 'the best similarity is strictly below the gate'."""
        R = "gate_failed"
        sim, _ = self.best_sim(row)
        ct = col(row, "cosine_theta")
        if sim is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["cosine_sim and all_sims_top5 both absent"] += 1
            return
        if ct is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["cosine_theta absent"] += 1
            return
        theta = snap_theta(ct.v)
        outcome, slack, bl = strictly_less(sim, theta)
        ops = [("best_sim", "%.6f (%s)" % (sim.v, sim.src)),
               ("cosine_theta", "%.6f" % theta.v),
               ("sim-theta", "%+.3e" % slack),
               ("tolerance", "%.1e" % (sim.u + theta.u))]
        self.record(R, outcome)
        if outcome == FAILS:
            self.fail(path, frame_raw, reason,
                      "best similarity is NOT strictly below the gate", ops)
        elif bl:
            self.border(path, frame_raw, reason,
                        "sim vs theta unresolved within print precision", ops)
        # reason-string operands must agree with the columns
        m = ACTUAL_FORMS[R].match(reason)
        if m:
            for label, rv, cv in (("sim", float(m.group(1)), sim),
                                  ("theta", float(m.group(2)), theta)):
                o, d = equal(M(rv, U_REASON, "reason " + label), cv)
                if o == FAILS:
                    self.struct("reason_operand_disagrees", path, frame_raw,
                                reason,
                                "reason %s= disagrees with the column (%+.4f)"
                                % (label, d),
                                [("reason " + label, rv),
                                 ("column", "%.6f" % cv.v)])

    def check_margin_failed(self, path, frame_raw, frame, reason, row):
        """S6: 'the margin is strictly below the margin in force'."""
        R = "margin_failed"
        mg = self.margin_value(row)
        m = ACTUAL_FORMS[R].match(reason)
        if mg is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["cosine_margin and all_sims_top5 both absent"] += 1
            return
        if not m:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["margin in force absent (reason unparsed)"] += 1
            return
        delta = snap_delta(float(m.group(2)))
        outcome, slack, bl = strictly_less(mg, delta)
        ops = [("margin", "%.6f (%s)" % (mg.v, mg.src)),
               ("delta_in_force", "%.6f (%s)" % (delta.v, delta.src)),
               ("margin-delta", "%+.3e" % slack),
               ("tolerance", "%.1e" % (mg.u + delta.u)),
               ("cosine_margin_col", row.get("cosine_margin")),
               ("reason_margin", m.group(1))]
        self.record(R, outcome)
        if outcome == FAILS:
            note = "margin is NOT strictly below the margin in force"
            if delta.src == "delta(4.3 exact)" and slack <= U_REASON:
                note += (" [reading-dependent: fails because 4.3 enumerates "
                         "delta exactly; would hold if the reason right-hand "
                         "side were a 3-decimal print of a continuous delta]")
            self.fail(path, frame_raw, reason, note, ops)
        elif bl:
            self.border(path, frame_raw, reason,
                        "margin vs delta unresolved within print precision", ops)
        # SPEC.md 4.3: delta is 0.05 for the first 30 frames.
        if frame is not None and frame <= EARLY_FRAME_LIMIT \
                and abs(delta.v - DELTA_EARLY) > U_REASON:
            self.advise("delta_early_mismatch", path, frame_raw, reason,
                        "frame <= 30 but delta in force is not 0.05 (SPEC 4.3)",
                        [("frame", frame), ("delta", delta.v)])
        if delta.src.startswith("delta(reason"):
            self.advise("delta_off_enumerated_set", path, frame_raw, reason,
                        "delta in force is not one of {0.03,0.04,0.05,0.07} (SPEC 4.3)",
                        [("delta", float(m.group(2)))])

    def check_revived(self, path, frame_raw, frame, reason, row):
        """S6: 'the similarity cleared the gate and the margin was decisive'."""
        R = "revived"
        sim, _ = self.best_sim(row)
        ct = col(row, "cosine_theta")
        mg = self.margin_value(row)
        missing = []
        if sim is None:
            missing.append("cosine_sim and all_sims_top5 both absent")
        if ct is None:
            missing.append("cosine_theta absent")
        delta, why = self.delta_in_force(path, frame)
        if mg is None:
            missing.append("cosine_margin and all_sims_top5 both absent")
        if delta is None:
            missing.append(why)

        results, ops, bl_any = [], [], False
        if sim is not None and ct is not None:
            theta = snap_theta(ct.v)
            o, d, bl = at_least(sim, theta)
            bl_any = bl_any or bl
            results.append(("gate cleared", o, d))
            ops += [("best_sim", "%.6f (%s)" % (sim.v, sim.src)),
                    ("cosine_theta", "%.6f" % theta.v),
                    ("sim-theta", "%+.3e" % d)]
        if mg is not None and delta is not None:
            o, d, bl = at_least(mg, delta)
            bl_any = bl_any or bl
            results.append(("margin decisive", o, d))
            ops += [("margin", "%.6f (%s)" % (mg.v, mg.src)),
                    ("delta_in_force", "%.6f (%s)" % (delta.v, delta.src)),
                    ("margin-delta", "%+.3e" % d)]
        elif mg is not None and delta is None:
            # The per-sequence value is unrecoverable, but 4.3 bounds it to
            # [0.03,0.07].  A margin at or above the upper bound is decisive
            # whatever t was; one below the lower bound is decisive under no
            # admissible t.  Between them the record stays untestable.
            if mg.v >= DELTA_RANGE[1] - mg.u:
                results.append(("margin decisive under any admissible delta "
                                "(4.3 range)", HOLDS, mg.v - DELTA_RANGE[1]))
                missing = [x for x in missing if x != why]
                ops += [("margin", "%.6f (%s)" % (mg.v, mg.src)),
                        ("delta upper bound", "%.2f" % DELTA_RANGE[1])]
            elif mg.v < DELTA_RANGE[0] - mg.u:
                results.append(("margin decisive under no admissible delta "
                                "(4.3 range)", FAILS, mg.v - DELTA_RANGE[0]))
                missing = [x for x in missing if x != why]
                ops += [("margin", "%.6f (%s)" % (mg.v, mg.src)),
                        ("delta lower bound", "%.2f" % DELTA_RANGE[0])]
            else:
                ops += [("margin", "%.6f (%s)" % (mg.v, mg.src)),
                        ("delta admissible range", "[0.03,0.07]")]

        bad = [r for r in results if r[1] == FAILS]
        if bad:
            self.record(R, FAILS)
            self.fail(path, frame_raw, reason,
                      "; ".join("%s does NOT hold (%+.3e)" % (n, d)
                                for n, _, d in bad), ops)
        elif missing:
            self.record(R, UNTESTABLE)
            for w in missing:
                self.untestable_why[R][w] += 1
        else:
            self.record(R, HOLDS)
            if bl_any:
                self.border(path, frame_raw, reason,
                            "gate or margin unresolved within print precision",
                            ops)

    def check_ocr(self, path, frame_raw, frame, reason, row):
        """S6: recovered by overlap against its own last observation.
        Operands: the iou= value in the reason string, iou_cost, iou_overlap.
        SPEC.md states no acceptance threshold for this stage, so the testable
        assertion is the mutual consistency of the three operands."""
        R = "ocr_last_obs_iou"
        m = ACTUAL_FORMS[R].match(reason)
        ic, io = col(row, "iou_cost"), col(row, "iou_overlap")
        if not m:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["reason-string iou= operand absent/unparsed"] += 1
            return
        riou = M(float(m.group(1)), U_REASON, "reason iou=")
        if io is None and ic is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["iou_cost and iou_overlap both absent"] += 1
            return
        checks, ops = [], [("reason iou", "%.4f" % riou.v)]
        if io is not None:
            o, d = equal(riou, io)
            checks.append(("reason iou == iou_overlap", o, d))
            ops.append(("iou_overlap", "%.4f" % io.v))
        if ic is not None and io is not None:
            o, d = equal(M(ic.v + io.v, 2 * U_COL, "sum"), M(1.0, U_EXACT, "1"))
            checks.append(("iou_cost + iou_overlap == 1", o, d))
            ops.append(("iou_cost", "%.4f" % ic.v))
        elif ic is not None:
            o, d = equal(riou, M(1.0 - ic.v, U_COL, "1-iou_cost"))
            checks.append(("reason iou == 1-iou_cost", o, d))
            ops.append(("iou_cost", "%.4f" % ic.v))
        bad = [c for c in checks if c[1] == FAILS]
        if bad:
            self.record(R, FAILS)
            self.fail(path, frame_raw, reason,
                      "; ".join("%s does NOT hold (%+.4f)" % (n, d)
                                for n, _, d in bad), ops)
        else:
            self.record(R, HOLDS)
        # Advisory: SPEC.md gives no OCR acceptance threshold.  0.30 is the
        # verifier's own assumption, reported, never counted as pass/fail.
        if riou.v < OCR_IOU_ASSUMED - U_REASON:
            self.advise("ocr_below_assumed_threshold", path, frame_raw, reason,
                        "recovery IoU below the ASSUMED 0.30 gate (not in SPEC)",
                        [("reason iou", riou.v)])

    def check_byte(self, path, frame_raw, frame, reason, row):
        """S6: 'the cost was acceptable at that stage (Section 4.6)' --
        strictly below 0.50."""
        R = "byte_second_assoc"
        ic, io, ds = (col(row, "iou_cost"), col(row, "iou_overlap"),
                      col(row, "det_score"))
        if ic is None and io is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["iou_cost and iou_overlap both absent"] += 1
            return
        cost = ic if ic is not None else M(1.0 - io.v, U_COL, "1-iou_overlap")
        outcome, slack, bl = strictly_less(cost, M(COST_ACCEPT_SECOND,
                                                  U_EXACT, "0.50"))
        ops = [("iou_cost", "%.4f (%s)" % (cost.v, cost.src)),
               ("iou_overlap", None if io is None else "%.4f" % io.v),
               ("det_score", None if ds is None else "%.4f" % ds.v),
               ("cost-0.50", "%+.3e" % slack)]
        self.record(R, outcome)
        if outcome == FAILS:
            self.fail(path, frame_raw, reason,
                      "second-stage cost is NOT strictly below 0.50", ops)
        elif bl:
            self.border(path, frame_raw, reason,
                        "cost vs 0.50 unresolved within print precision", ops)
        if ic is not None and io is not None:
            o, d = equal(M(ic.v + io.v, 2 * U_COL, "sum"), M(1.0, U_EXACT, "1"))
            if o == FAILS:
                self.struct("inv1_iou_sum", path, frame_raw, reason,
                            "iou_cost + iou_overlap != 1 (%+.4f)" % d, ops)
        m = ACTUAL_FORMS[R].match(reason)
        if m:
            if io is not None:
                o, d = equal(M(float(m.group(1)), U_REASON, "reason iou="), io)
                if o == FAILS:
                    self.struct("reason_operand_disagrees", path, frame_raw,
                                reason, "reason iou= disagrees with iou_overlap (%+.4f)" % d,
                                [("reason iou", m.group(1)),
                                 ("iou_overlap", "%.4f" % io.v)])
            if ds is not None:
                o, d = equal(M(float(m.group(2)), U_REASON, "reason score="), ds)
                if o == FAILS:
                    self.struct("reason_operand_disagrees", path, frame_raw,
                                reason, "reason score= disagrees with det_score (%+.4f)" % d,
                                [("reason score", m.group(2)),
                                 ("det_score", "%.4f" % ds.v)])
        # Advisory: SPEC.md 4.6 says this stage "operates on detections below
        # the 0.50 detection threshold".  S6 names only the cost as the
        # assertion, so a det_score at or above 0.50 is reported, not failed.
        if ds is not None and ds.v >= DET_THRESHOLD - U_COL:
            self.advise("byte_det_score_not_low", path, frame_raw, reason,
                        "det_score is not below the 0.50 detection threshold (SPEC 4.6)",
                        [("det_score", "%.4f" % ds.v),
                         ("iou_cost", "%.4f" % cost.v)])

    def check_unconfirmed(self, path, frame_raw, frame, reason, row):
        """S6 states no threshold for this reason, only that the track was
        'updated from a detection on overlap'.  Tested: an overlap exists, and
        the cost was acceptable to the first-stage assignment (4.6, <= 0.80).
        See report section 4 for the ambiguity."""
        R = "unconfirmed_update"
        ic, io, ds = (col(row, "iou_cost"), col(row, "iou_overlap"),
                      col(row, "det_score"))
        if ic is None and io is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["iou_cost and iou_overlap both absent"] += 1
            return
        cost = ic if ic is not None else M(1.0 - io.v, U_COL, "1-iou_overlap")
        ovl = io if io is not None else M(1.0 - ic.v, U_COL, "1-iou_cost")
        ops = [("iou_cost", "%.4f" % cost.v), ("iou_overlap", "%.4f" % ovl.v),
               ("det_score", None if ds is None else "%.4f" % ds.v)]
        checks = []
        o, d, bl1 = at_most(cost, M(COST_ACCEPT_FIRST, U_EXACT, "0.80"))
        checks.append(("cost <= 0.80 (4.6)", o, d))
        o2, d2, bl2 = strictly_less(M(0.0, U_EXACT, "0"), ovl)
        checks.append(("a positive overlap exists", o2, d2))
        bad = [c for c in checks if c[1] == FAILS]
        if bad:
            self.record(R, FAILS)
            self.fail(path, frame_raw, reason,
                      "; ".join("%s does NOT hold (%+.4f)" % (n, d)
                                for n, _, d in bad), ops)
        else:
            self.record(R, HOLDS)
            if bl1 or bl2:
                self.border(path, frame_raw, reason,
                            "cost or overlap unresolved within print precision",
                            ops)
        if ic is not None and io is not None:
            o, d = equal(M(ic.v + io.v, 2 * U_COL, "sum"), M(1.0, U_EXACT, "1"))
            if o == FAILS:
                self.struct("inv1_iou_sum", path, frame_raw, reason,
                            "iou_cost + iou_overlap != 1 (%+.4f)" % d, ops)

    def check_reactivate(self, path, frame_raw, frame, reason, row):
        """S6: 'recovered from the lost pool at a cost the assignment stage
        accepted' -- 4.6, first stage, at or below 0.80.  The 4.7 decomposition
        is reported in its own category, never as pass or fail."""
        R = "re_activate_from_lost"
        fc = col(row, "fused_cost")
        ic, ds = col(row, "iou_cost"), col(row, "det_score")
        if fc is None:
            self.record(R, UNTESTABLE)
            self.untestable_why[R]["fused_cost absent"] += 1
            return
        outcome, slack, bl = at_most(fc, M(COST_ACCEPT_FIRST, U_EXACT, "0.80"))
        ops = [("fused_cost", "%.4f" % fc.v),
               ("iou_cost", None if ic is None else "%.4f" % ic.v),
               ("det_score", None if ds is None else "%.4f" % ds.v),
               ("fused-0.80", "%+.3e" % slack)]
        self.record(R, outcome)
        if outcome == FAILS:
            self.fail(path, frame_raw, reason,
                      "fused_cost exceeds the 0.80 first-stage acceptance", ops)
        elif bl:
            self.border(path, frame_raw, reason,
                        "fused_cost vs 0.80 unresolved within print precision",
                        ops)

        # SPEC.md 4.7 decomposition.  c1 = 1 - (1-IoU)*det_score
        #                                = 1 - iou_cost*det_score
        # c2 = min(c1, c_emb) <= c1 ; c3 = c2 + 0.2*a_OCM with a_OCM >= 0.
        if ic is None or ds is None:
            self.composite["operands absent: cannot attempt 4.7 decomposition"] += 1
            return
        c1 = 1.0 - ic.v * ds.v
        u_c1 = U_COL * ds.v + U_COL * ic.v      # first-order propagation
        tol = u_c1 + fc.u
        d = fc.v - c1
        if d > tol:
            # c3 > c1 >= c2, so 0.2*a_OCM = c3 - c2 > 0: a motion term was
            # certainly added and the composite cannot be decomposed.
            key = "motion term provably present (fused_cost > c1): not decomposable"
            self.composite[key] += 1
            self.composite_items.append(Finding(
                "composite", path, frame_raw, reason, key,
                [("fused_cost", "%.4f" % fc.v), ("c1", "%.6f" % c1),
                 ("fused-c1", "%+.4f" % d), ("iou_cost", "%.4f" % ic.v),
                 ("det_score", "%.4f" % ds.v)]))
        elif abs(d) <= tol:
            # consistent with c2 = c1 and a_OCM = 0; appearance not selected.
            self.composite["fused_cost == c1: consistent with no appearance "
                           "and no motion term"] += 1
        else:
            # c2 <= c3 < c1, so the minimum selected c_emb -- but a_OCM cannot
            # be shown to be zero, so c_emb is a lower bound only.
            key = ("appearance selected (fused_cost < c1) but motion term "
                   "indeterminable: not decomposable")
            self.composite[key] += 1
            # 4.7 claims sim = 1 - 2*fused_cost in the a_OCM = 0 case; check
            # whether that would even be a valid similarity.
            if 1.0 - 2.0 * fc.v < -U_COL:
                self.composite["  of which 1-2*fused_cost would be a negative "
                               "similarity (contradicts invariant 2)"] += 1

    def check_immature(self, path, frame_raw, frame, reason, row):
        """S6: 'no comparison was performed because no template was mature.
        There is no operand to check; the testable property is that the
        appearance fields are absent.'"""
        R = "registry_empty_or_immature"
        present = [f for f in APPEARANCE_COMPARISON_FIELDS
                   if (row.get(f) or "").strip() != ""]
        ops = [(f, row.get(f)) for f in APPEARANCE_COMPARISON_FIELDS]
        if present:
            self.record(R, FAILS)
            self.fail(path, frame_raw, reason,
                      "appearance comparison fields present though no "
                      "comparison was performed: " + ", ".join(present), ops)
        else:
            self.record(R, HOLDS)
        if (row.get("feat_norm") or "").strip() != "":
            self.advise("immature_feat_norm_present", path, frame_raw, reason,
                        "feat_norm is populated; whether it counts as an "
                        "'appearance field' is not stated in SPEC.md",
                        [("feat_norm", row.get("feat_norm")),
                         ("n_registry", row.get("n_registry"))])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def emit(v, files, out, max_items):
    w = out.write
    order = ["byte_second_assoc", "re_activate_from_lost", "unconfirmed_update",
             "revived", "margin_failed", "gate_failed", "ocr_last_obs_iou",
             "registry_empty_or_immature"]
    order += [r for r in sorted(v.counts) if r not in order]

    w("## Scope\n\n")
    w("- Logs read: **%d** files, **%d** records.\n" % (len(files), v.examined))
    w("- Verifier: `verify_independent.py`, standard library only, "
      "written from `SPEC.md` v1.0 alone.\n\n")

    w("## 2. Per-reason results\n\n")
    w("| reason | examined | holds | fails | untestable |\n")
    w("|---|---:|---:|---:|---:|\n")
    tot = collections.Counter()
    for r in order:
        c = v.counts[r]
        n = c[HOLDS] + c[FAILS] + c[UNTESTABLE]
        if n == 0:
            continue
        tot.update(c)
        w("| `%s` | %d | %d | %d | %d |\n"
          % (r, n, c[HOLDS], c[FAILS], c[UNTESTABLE]))
    w("| **total** | **%d** | **%d** | **%d** | **%d** |\n\n"
      % (tot[HOLDS] + tot[FAILS] + tot[UNTESTABLE], tot[HOLDS], tot[FAILS],
         tot[UNTESTABLE]))
    if any(v.untestable_why.values()):
        w("Untestable records, by the operand that is missing:\n\n")
        for r in order:
            for why, n in v.untestable_why[r].most_common():
                w("- `%s`: %d -- %s\n" % (r, n, why))
        w("\n")
    else:
        w("No record was untestable: every operand each Section 6 check "
          "needs was present on every record of that reason.\n\n")

    w("## 3. Failures, itemized\n\n")
    if not v.failures:
        w("None.\n\n")
    else:
        w("%d failing record(s).\n\n" % len(v.failures))
        for f in v.failures:
            w("%d. %s\n" % (v.failures.index(f) + 1, f.line()))
        w("\n")

    w("### Borderline, decided by the tolerance\n\n")
    w("Records whose assertion an exact comparison of the printed values "
      "would reject, but which the stated tolerance cannot resolve. Counted "
      "as **holds** -- the operands do not demonstrably contradict the rule.\n\n")
    if not v.borderline:
        w("None.\n\n")
    else:
        for f in v.borderline:
            w("- %s\n" % f.line())
        w("\n")

    w("### Composite cost, SPEC.md 4.7 (neither pass nor fail)\n\n")
    if not v.composite:
        w("None.\n\n")
    else:
        for k, n in sorted(v.composite.items(), key=lambda x: -x[1]):
            w("- **%d** -- %s\n" % (n, k))
        w("\n")

    w("### Advisory checks (implied by SPEC.md, not stated as assertions in S6)\n\n")
    if not v.advisory_n:
        w("None triggered.\n\n")
    else:
        for tag, n in v.advisory_n.most_common():
            w("- `%s`: **%d** record(s)\n" % (tag, n))
        w("\n")
        for tag, items in v.advisory.items():
            w("<details><summary><code>%s</code> -- %d record(s)</summary>\n\n"
              % (tag, len(items)))
            for f in items[:max_items]:
                w("- %s\n" % f.line())
            if len(items) > max_items:
                w("- ... and %d more\n" % (len(items) - max_items))
            w("\n</details>\n\n")

    w("### Structural checks (Section 2 definitions, Section 3 pairing, "
      "Section 5 invariants)\n\n")
    inv_tags = ["inv1_iou_sum", "inv2_range", "inv3_sim_is_top", "inv4_margin",
                "inv5_sorted", "inv6_frame_order", "inv7_theta_range"]
    w("| check | violations |\n|---|---:|\n")
    for t in inv_tags:
        w("| Section 5 `%s` | %d |\n" % (t, v.structural_n[t]))
    for t in sorted(v.structural_n):
        if t in inv_tags or t == "sec2_box_iou_agrees":
            continue
        w("| `%s` | %d |\n" % (t, v.structural_n[t]))
    w("\n")
    agree = v.structural_n["sec2_box_iou_agrees"]
    mism = v.structural_n["sec2_box_iou_mismatch"]
    if agree + mism:
        w("Section 2 defines `iou_cost` as *1 minus the IoU overlap between "
          "`det_box` and `track_box`*. Recomputing that IoU from the two "
          "logged boxes: **%d of %d** records with both boxes present "
          "disagree with the logged `iou_overlap` by more than 0.01 "
          "(%.1f%%).\n\n" % (mism, agree + mism,
                             100.0 * mism / (agree + mism)))
        for f in v.structural["sec2_box_iou_mismatch"][:max_items]:
            w("- %s\n" % f.line())
        if mism > max_items:
            w("- ... and %d more\n" % (mism - max_items))
        w("\n")
    if v.notes:
        w("Notes: %s\n\n" % ", ".join("%s (%d)" % (k, n)
                                      for k, n in v.notes.items()))

    w("### Recovered disambiguation margin, and its feasibility per sequence\n\n")
    w("SPEC.md 4.3's `delta` is not logged as a column. Section 4.1 measures "
      "`t` once at frame 30, so `delta` is one constant per sequence after "
      "frame 30; it is read off the right-hand side of that file's own "
      "`margin_failed` records and then used to test `revived`.\n\n")
    w("That also makes the whole file testable at once. Every `margin_failed` "
      "record requires `delta >` its margin; every `revived` record requires "
      "`delta <=` its margin. The intersection is the interval the true "
      "`delta` must occupy if every record in the file is faithful. An empty "
      "intersection proves the file cannot be explained by any single "
      "threshold -- a conclusion that does not depend on what the log "
      "prints.\n\n")
    w("| sequence | printed | required by the decisions | implied by the "
      "print | consistent |\n|---|---|---|---|---|\n")
    for path, p, lo, hi, nmf, nrev, ok in v.feasible_delta(files):
        nm = os.path.basename(path)
        if p is None:
            w("| `%s` | none | -- | -- | not testable (no `margin_failed`) |\n"
              % nm)
            continue
        w("| `%s` | %.3f | (%.4f, %.4f] | [%.4f, %.4f] | %s |\n"
          % (nm, p, lo, hi, p - U_REASON, p + U_REASON,
             "yes" if ok else "**NO -- EMPTY**"))
    w("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"))
    ap.add_argument("--report", default="-")
    ap.add_argument("--max-items", type=int, default=40)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.logs, "*.csv")))
    if not files:
        sys.stderr.write("no logs under %s\n" % a.logs)
        return 2

    v = Verifier()
    v.run(files)

    if a.report == "-":
        emit(v, files, sys.stdout, a.max_items)
    else:
        with open(a.report, "w") as fh:
            emit(v, files, fh, a.max_items)
        sys.stderr.write("wrote %s\n" % a.report)
    return 1 if v.failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
bytetrack_audit_port.py -- does the audit discipline port to a tracker we did
not write?

Reviewers 2 and 4 asked for ByteTrack and OC-SORT to be instrumented with the
audit layer. Taken literally the request cannot be met, and the reason is
worth stating rather than hiding: most of what our audit layer records is
appearance reasoning. Gallery revival, appearance-margin refusal, gate
refusal and immature registry all presuppose a template registry. ByteTrack
has none; it associates on IoU and detection score. OC-SORT has none; it is
observation-centric motion. Adding the appearance machinery first would mean
instrumenting a third tracker and calling it ByteTrack.

What does port is the discipline: record every association decision as an
append-only sink that nothing reads back, and show the tracker output is
unchanged. That is the property the reviewers' objection actually turns on,
and it is testable on a tracker we did not write.

This script:

  1. fetches stock ByteTrack and runs it UNMODIFIED. The audit layer is
     attached by wrapping `matching.linear_assignment` at the module
     boundary, so no line of ByteTrack is edited. If the wrapping changed
     behaviour, the diff below would catch it.
  2. drives it from MOT17 public detections, so no detector, no GPU, no
     weights. The association stage is the only part the logging touches,
     and this isolates it.
  3. runs each sequence twice, logging on and off, and compares the output
     byte for byte.

Usage
-----
    python bytetrack_audit_port.py --mot17 /path/to/MOT17/train [--out DIR]
                                   [--seqs MOT17-02-DPM,MOT17-04-DPM]

Outputs (default ./bytetrack_port_out):
    audit_on/<seq>.txt, audit_off/<seq>.txt   tracker output
    audit_on/<seq>_assoc_events.csv           the association record
    report.txt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

BYTETRACK_URL = "https://github.com/ifzhang/ByteTrack.git"
OCSORT_URL    = "https://github.com/noahcao/OC_SORT.git"

DEFAULT_SEQS = ["MOT17-02-DPM", "MOT17-04-DPM"]

# ByteTrack's own MOT17 defaults.
TRACK_THRESH = 0.5
MATCH_THRESH = 0.8
TRACK_BUFFER = 30

STAGE_NAMES = {0: "1_first_assoc", 1: "2_low_score", 2: "3_unconfirmed"}


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, t: str = "") -> None:
        print(t)
        self.lines.append(t)

    def rule(self, t: str) -> None:
        self("")
        self("=" * 78)
        self(t)
        self("=" * 78)

    def save(self, p: Path) -> None:
        p.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Shims. ByteTrack imports cv2 and torch but never calls them, and uses two
# compiled helpers that have pure-numpy equivalents. Shimming those is what
# lets stock ByteTrack run with nothing but numpy and scipy.
# --------------------------------------------------------------------------

def _bbox_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=float)
    b = np.ascontiguousarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1 + 1, 0, None)
    ih = np.clip(iy2 - iy1 + 1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    bb = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _lapjv(cost: np.ndarray, extend_cost: bool = False, cost_limit: float = np.inf):
    """scipy stand-in for lap.lapjv, matching its (cost, x, y) return shape.
    x[i] is the column matched to row i, or -1; y[j] the row matched to col j."""
    from scipy.optimize import linear_sum_assignment

    n, m = cost.shape
    x = np.full(n, -1, dtype=int)
    y = np.full(m, -1, dtype=int)
    if n == 0 or m == 0:
        return 0.0, x, y

    big = float(cost_limit) if np.isfinite(cost_limit) else float(cost.max() + 1)
    work = np.where(cost <= big, cost, big + 1.0)
    ri, ci = linear_sum_assignment(work)
    total = 0.0
    for i, j in zip(ri, ci):
        if cost[i, j] <= cost_limit:
            x[i] = j
            y[j] = i
            total += float(cost[i, j])
    return total, x, y


class _PermissiveStub(types.ModuleType):
    """Stands in for a module that is imported but never really used.
    Any attribute resolves to a fresh empty class, so existence checks and
    isinstance tests against it behave sanely."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls


def install_shims() -> list[str]:
    installed = []

    # ByteTrack predates numpy 1.24, which removed the scalar aliases.
    # Restoring the API the code was written against is preferable to
    # editing the code, and leaves the "unmodified source" claim intact.
    for alias, builtin in (("float", float), ("int", int), ("bool", bool),
                           ("object", object), ("str", str)):
        if not hasattr(np, alias):
            setattr(np, alias, builtin)
            installed.append(f"np.{alias}")
    for name in ("cv2", "torch", "torch.nn", "torch.nn.functional"):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:                              # noqa: BLE001
                sys.modules[name] = _PermissiveStub(name)
                installed.append(name)
    # make torch.nn.functional reachable as an attribute chain
    if "torch" in sys.modules and isinstance(sys.modules["torch"], types.ModuleType):
        t = sys.modules["torch"]
        if not hasattr(t, "nn"):
            t.nn = sys.modules.get("torch.nn", types.ModuleType("torch.nn"))
        if not hasattr(t.nn, "functional"):
            t.nn.functional = sys.modules.get("torch.nn.functional",
                                              types.ModuleType("torch.nn.functional"))

    try:
        import cython_bbox                                 # noqa: F401
    except Exception:                                      # noqa: BLE001
        m = types.ModuleType("cython_bbox")
        m.bbox_overlaps = _bbox_overlaps
        sys.modules["cython_bbox"] = m
        installed.append("cython_bbox")

    try:
        import lap                                         # noqa: F401
    except Exception:                                      # noqa: BLE001
        m = types.ModuleType("lap")
        m.lapjv = _lapjv
        sys.modules["lap"] = m
        installed.append("lap")
    return installed


# --------------------------------------------------------------------------
# The audit layer, attached from outside ByteTrack.
# --------------------------------------------------------------------------

class AssocLog:
    """Append-only sink. Nothing in the tracker reads it back, which is the
    whole point; the output diff is what proves that claim."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.frame = 0
        self.stage = 0

    def new_frame(self, f: int) -> None:
        self.frame = f
        self.stage = 0

    def record(self, cost: np.ndarray, thresh: float,
               matches, u_track, u_det) -> None:
        stage = STAGE_NAMES.get(self.stage, f"stage_{self.stage}")
        self.stage += 1
        n, m = (cost.shape if cost.size else (0, 0))
        for ti, di in matches:
            self.rows.append({
                "frame": self.frame, "stage": stage,
                "track_row": int(ti), "det_col": int(di),
                "iou_cost": round(float(cost[ti, di]), 6),
                "iou": round(1.0 - float(cost[ti, di]), 6),
                "thresh": thresh,
                "n_tracks": n, "n_dets": m,
                "reason": "matched",
            })
        for ti in u_track:
            self.rows.append({
                "frame": self.frame, "stage": stage,
                "track_row": int(ti), "det_col": -1,
                "iou_cost": None, "iou": None, "thresh": thresh,
                "n_tracks": n, "n_dets": m,
                "reason": "unmatched_track",
            })
        for di in u_det:
            self.rows.append({
                "frame": self.frame, "stage": stage,
                "track_row": -1, "det_col": int(di),
                "iou_cost": None, "iou": None, "thresh": thresh,
                "n_tracks": n, "n_dets": m,
                "reason": "unmatched_detection",
            })

    def record_pairs(self, stage: str, matched, iou, thresh,
                     u_track, u_det, n_tracks: int, n_dets: int) -> None:
        """Generic recorder for trackers whose association returns pairs."""
        self.stage += 1
        iou = np.asarray(iou)
        for pair in matched:
            ti, di = int(pair[0]), int(pair[1])
            v = float(iou[ti, di]) if iou.size and ti < iou.shape[0] \
                and di < iou.shape[1] else None
            self.rows.append({
                "frame": self.frame, "stage": stage,
                "track_row": ti, "det_col": di,
                "iou_cost": None if v is None else round(1.0 - v, 6),
                "iou": None if v is None else round(v, 6),
                "thresh": thresh, "n_tracks": n_tracks, "n_dets": n_dets,
                "reason": "matched"})
        for ti in u_track:
            self.rows.append({"frame": self.frame, "stage": stage,
                              "track_row": int(ti), "det_col": -1,
                              "iou_cost": None, "iou": None, "thresh": thresh,
                              "n_tracks": n_tracks, "n_dets": n_dets,
                              "reason": "unmatched_track"})
        for di in u_det:
            self.rows.append({"frame": self.frame, "stage": stage,
                              "track_row": -1, "det_col": int(di),
                              "iou_cost": None, "iou": None, "thresh": thresh,
                              "n_tracks": n_tracks, "n_dets": n_dets,
                              "reason": "unmatched_detection"})

    def save(self, path: str) -> int:
        import pandas as pd
        pd.DataFrame(self.rows).to_csv(path, index=False)
        return len(self.rows)


def instrument(matching_mod, tracker_cls, log: AssocLog | None):
    """Wrap at the module boundary. Returns an undo callable.

    With log=None nothing is installed at all, which is the honest form of
    'logging off': not a disabled branch inside the tracker, but no audit
    code in the process."""
    if log is None:
        return lambda: None

    orig_la = matching_mod.linear_assignment
    orig_update = tracker_cls.update

    def wrapped_la(cost_matrix, thresh):
        out = orig_la(cost_matrix, thresh)
        try:
            log.record(np.asarray(cost_matrix), thresh, out[0], out[1], out[2])
        except Exception:                                  # noqa: BLE001
            pass          # a sink must never affect the run, including by raising
        return out

    def wrapped_update(self, output_results, img_info, img_size):
        log.new_frame(self.frame_id + 1)
        return orig_update(self, output_results, img_info, img_size)

    matching_mod.linear_assignment = wrapped_la
    tracker_cls.update = wrapped_update

    def undo():
        matching_mod.linear_assignment = orig_la
        tracker_cls.update = orig_update

    return undo


# --------------------------------------------------------------------------

def read_det(path: Path) -> dict[int, np.ndarray]:
    """MOT17 det.txt -> {frame: (N,5) [x1,y1,x2,y2,score]}."""
    raw = np.loadtxt(path, delimiter=",", usecols=(0, 2, 3, 4, 5, 6))
    if raw.ndim == 1:
        raw = raw[None, :]
    out: dict[int, np.ndarray] = {}
    for f in np.unique(raw[:, 0]).astype(int):
        r = raw[raw[:, 0] == f]
        b = np.stack([r[:, 1], r[:, 2],
                      r[:, 1] + r[:, 3], r[:, 2] + r[:, 4], r[:, 5]], axis=1)
        out[int(f)] = b
    return out


def seq_size(seq_dir: Path) -> tuple[int, int, int]:
    import configparser
    c = configparser.ConfigParser()
    c.read(seq_dir / "seqinfo.ini")
    s = c["Sequence"]
    return int(s["imHeight"]), int(s["imWidth"]), int(s["seqLength"])


def normalise_scores(dets: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Public MOT17 detection confidences are not on [0,1] for every
    detector. ByteTrack's thresholds assume they are, so rescale per
    sequence. Applied identically to both passes, so it cannot affect the
    comparison."""
    allv = np.concatenate([d[:, 4] for d in dets.values()]) if dets else np.array([0.0])
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= 1.0 and lo >= 0.0:
        return dets
    rng = hi - lo if hi > lo else 1.0
    return {f: np.column_stack([d[:, :4], (d[:, 4] - lo) / rng]) for f, d in dets.items()}


def run_sequence(BYTETracker, matching_mod, basetrack_mod, seq_dir: Path,
                 out_txt: Path, log: AssocLog | None) -> int:
    class Args:
        track_thresh = TRACK_THRESH
        match_thresh = MATCH_THRESH
        track_buffer = TRACK_BUFFER
        mot20 = False

    h, w, n_frames = seq_size(seq_dir)
    dets = normalise_scores(read_det(seq_dir / "det" / "det.txt"))

    # ByteTrack numbers tracks from a class-level counter, so a second run in
    # the same process continues where the first stopped. Reset it, or the
    # two passes differ for a reason that has nothing to do with logging.
    basetrack_mod.BaseTrack._count = 0

    undo = instrument(matching_mod, BYTETracker, log)
    try:
        tracker = BYTETracker(Args(), frame_rate=30)
        lines = []
        for f in range(1, n_frames + 1):
            d = dets.get(f, np.zeros((0, 5)))
            targets = tracker.update(d, (h, w), (h, w))
            for t in targets:
                x, y, bw, bh = t.tlwh
                if bw * bh < 100:
                    continue
                lines.append(f"{f},{t.track_id},{x:.2f},{y:.2f},"
                             f"{bw:.2f},{bh:.2f},{t.score:.2f},-1,-1,-1")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        undo()
    return len(lines)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------
# OC-SORT. Same discipline, different association structure: observation-
# centric momentum in `associate`, then two recovery passes through
# `linear_assignment`. Both are wrapped in the ocsort module's own namespace,
# because `from .association import *` binds them there.
# --------------------------------------------------------------------------

OC_STAGES = {0: "1_associate_ocm", 1: "2_byte_recovery", 2: "3_ocr_recovery"}


def instrument_ocsort(ocsort_mod, log: AssocLog | None):
    if log is None:
        return lambda: None

    orig_assoc = ocsort_mod.associate
    orig_la = ocsort_mod.linear_assignment
    orig_update = ocsort_mod.OCSort.update

    def wrapped_assoc(detections, trackers, iou_threshold, velocities,
                      previous_obs, vdc_weight):
        out = orig_assoc(detections, trackers, iou_threshold, velocities,
                         previous_obs, vdc_weight)
        try:
            matched, u_det, u_trk = out
            iou = ocsort_mod.iou_batch(detections, trackers) \
                if len(detections) and len(trackers) else np.zeros((0, 0))
            log.record_pairs("1_associate_ocm", matched, iou,
                             iou_threshold, u_trk, u_det,
                             len(trackers), len(detections))
        except Exception:                                  # noqa: BLE001
            pass
        return out

    def wrapped_la(cost_matrix):
        out = orig_la(cost_matrix)
        try:
            c = np.asarray(cost_matrix)
            log.record_pairs(OC_STAGES.get(log.stage, f"stage_{log.stage}"),
                             [(b, a) for a, b in out], -c, None,
                             [], [], *c.shape)
        except Exception:                                  # noqa: BLE001
            pass
        return out

    def wrapped_update(self, output_results, img_info, img_size):
        log.new_frame(self.frame_count + 1)
        log.stage = 1          # associate() has already consumed stage 0
        return orig_update(self, output_results, img_info, img_size)

    ocsort_mod.associate = wrapped_assoc
    ocsort_mod.linear_assignment = wrapped_la
    ocsort_mod.OCSort.update = wrapped_update

    def undo():
        ocsort_mod.associate = orig_assoc
        ocsort_mod.linear_assignment = orig_la
        ocsort_mod.OCSort.update = orig_update

    return undo


def run_sequence_ocsort(ocsort_mod, seq_dir: Path, out_txt: Path,
                        log: AssocLog | None) -> int:
    h, w, n_frames = seq_size(seq_dir)
    dets = normalise_scores(read_det(seq_dir / "det" / "det.txt"))

    undo = instrument_ocsort(ocsort_mod, log)
    try:
        tracker = ocsort_mod.OCSort(det_thresh=TRACK_THRESH, iou_threshold=0.3)
        lines = []
        for f in range(1, n_frames + 1):
            d = dets.get(f, np.empty((0, 5)))
            targets = tracker.update(d, (h, w), (h, w))
            for row in targets:
                x1, y1, x2, y2, tid = row[:5]
                bw, bh = x2 - x1, y2 - y1
                if bw * bh < 100:
                    continue
                lines.append(f"{f},{int(tid)},{x1:.2f},{y1:.2f},"
                             f"{bw:.2f},{bh:.2f},1.00,-1,-1,-1")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        undo()
    return len(lines)


def load_ocsort(src: Path, say: Report):
    if not (src / "trackers" / "ocsort_tracker" / "ocsort.py").exists():
        say(f"  cloning stock OC-SORT -> {src}")
        r = subprocess.run(["git", "clone", "-q", "--depth", "1",
                            OCSORT_URL, str(src)], capture_output=True, text=True)
        if r.returncode:
            say("  clone failed: " + (r.stderr or "").strip()[:300])
            return None
    try:
        import filterpy                                   # noqa: F401
    except Exception:                                     # noqa: BLE001
        say("  installing filterpy (pure Python, OC-SORT's Kalman filter)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "filterpy"],
                       capture_output=True, text=True)
        try:
            import filterpy                               # noqa: F401
        except Exception as exc:                          # noqa: BLE001
            say(f"  !! filterpy unavailable: {exc}")
            return None

    rev = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    say(f"  OC-SORT source  : {src}")
    say(f"  revision        : {rev}  (unmodified)")

    sys.path.insert(0, str(src))
    import importlib
    for name, path in (("trackers", src / "trackers"),
                       ("trackers.ocsort_tracker", src / "trackers" / "ocsort_tracker")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    return importlib.import_module("trackers.ocsort_tracker.ocsort")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mot17", required=True, help="MOT17/train (or test) directory")
    ap.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    ap.add_argument("--tracker", default="both",
                    choices=["bytetrack", "ocsort", "both"])
    ap.add_argument("--src", default=None, help="existing ByteTrack checkout")
    ap.add_argument("--out", default="bytetrack_port_out")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    say = Report()

    say("bytetrack_audit_port.py")
    say(f"  MOT17 : {args.mot17}")
    say(f"  out   : {out}")

    # ---- fetch stock ByteTrack ------------------------------------------
    src = Path(args.src).expanduser().resolve() if args.src \
        else out.parent / "ByteTrack_src"
    if not (src / "yolox" / "tracker" / "byte_tracker.py").exists():
        say(f"\ncloning stock ByteTrack -> {src}")
        r = subprocess.run(["git", "clone", "-q", "--depth", "1",
                            BYTETRACK_URL, str(src)],
                           capture_output=True, text=True)
        if r.returncode:
            say("  clone failed: " + (r.stderr or "").strip()[:300])
            return 2
    say(f"  ByteTrack source: {src}")

    rev = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    say(f"  revision        : {rev}  (unmodified)")

    shims = install_shims()
    say(f"  shimmed modules : {shims or 'none needed'}")
    say("  cv2 and torch are imported by ByteTrack but never called; lap and")
    say("  cython_bbox are replaced by exact numpy/scipy equivalents. No line")
    say("  of ByteTrack is edited.")

    # yolox/__init__.py imports the training utilities, which need a real
    # torch. The tracker needs none of that, so bind the two package names to
    # the source directories and import the tracker modules directly. The
    # module files themselves are untouched.
    sys.path.insert(0, str(src))
    for name, path in (("yolox", src / "yolox"),
                       ("yolox.tracker", src / "yolox" / "tracker")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

    import importlib
    matching_mod = importlib.import_module("yolox.tracker.matching")
    basetrack_mod = importlib.import_module("yolox.tracker.basetrack")
    BYTETracker = importlib.import_module("yolox.tracker.byte_tracker").BYTETracker

    # ---- run both passes -------------------------------------------------
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    root = Path(args.mot17).expanduser().resolve()

    seqs = [x.strip() for x in args.seqs.split(",") if x.strip()]
    root = Path(args.mot17).expanduser().resolve()
    overall = True

    def do_tracker(label: str, runner, mod, say=say):
        nonlocal overall
        say.rule(f"TWO PASSES OVER STOCK {label.upper()}, PUBLIC DETECTIONS")
        res = []
        for sq in seqs:
            sd = root / sq
            if not (sd / "det" / "det.txt").exists():
                say(f"  !! {sq}: no det/det.txt under {sd}")
                continue
            lg = AssocLog()
            on = out / label / "audit_on" / f"{sq}.txt"
            n_on = runner(mod, sd, on, lg)
            n_ev = lg.save(str(out / label / "audit_on" / f"{sq}_assoc_events.csv"))
            off = out / label / "audit_off" / f"{sq}.txt"
            n_off = runner(mod, sd, off, None)
            say(f"  {sq:<16} on={n_on:>6,} rows  off={n_off:>6,} rows  "
                f"{n_ev:>7,} association events logged")
            res.append((sq, on, off))

        if not res:
            say("  no sequence produced output")
            overall = False
            return

        say("")
        say(f"  {'sequence':<18}{'bytes on':>11}{'bytes off':>11}{'sha256':>10}")
        say("  " + "-" * 50)
        ok = True
        for sq, a, b in res:
            same = sha(a) == sha(b)
            ok &= same
            say(f"  {sq:<18}{a.stat().st_size:>11,}{b.stat().st_size:>11,}"
                f"{'MATCH' if same else 'DIFFER':>10}")
        overall &= ok
        say("")
        say(f"  {label}: {'bit-identical with logging on and off'
                          if ok else 'OUTPUTS DIFFER, the wrapping is not a pure sink'}")

    if args.tracker in ("bytetrack", "both"):
        do_tracker("bytetrack",
                   lambda m, sd, o, lg: run_sequence(BYTETracker, matching_mod,
                                                     basetrack_mod, sd, o, lg),
                   None)

    if args.tracker in ("ocsort", "both"):
        say.rule("OC-SORT SOURCE")
        oc_src = (Path(args.src).parent if args.src else out.parent) / "OC_SORT_src"
        oc_mod = load_ocsort(oc_src, say)
        if oc_mod is None:
            say("  skipped")
            overall = False
        else:
            do_tracker("ocsort",
                       lambda m, sd, o, lg: run_sequence_ocsort(oc_mod, sd, o, lg),
                       oc_mod)

    all_same = overall

    say.rule("WHAT THIS DOES AND DOES NOT SHOW")
    say("""
  Shows: the output-neutral logging discipline is not a property of our
  implementation. Attached from outside, to unmodified third-party code, it
  records every association decision and leaves the tracker output unchanged
  byte for byte. Two trackers, two different association structures: IoU and
  detection score in ByteTrack, observation-centric momentum in OC-SORT.

  Shows: the discipline needs no detector, no weights and no GPU. Driven
  from MOT17 public detections, it isolates the association stage, which is
  the only part logging touches.

  Does not show: that our appearance-decision records port. They cannot.
  Gallery revival, appearance-margin refusal, gate refusal and immature
  registry all presuppose a template registry, and neither ByteTrack nor
  OC-SORT has one. Adding that machinery first would mean instrumenting a
  third tracker and calling it ByteTrack. What is logged here is the
  association record these trackers actually have: cost matrices,
  thresholds, matched pairs, and unmatched tracks and detections.

  Does not show: anything about the accuracy gap between these systems and
  ours. That comparison was withdrawn from Section 5 in this revision and is
  not reinstated here.
""".rstrip())

    say.save(out / "report.txt")
    print(f"\nreport.txt written to {out}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())

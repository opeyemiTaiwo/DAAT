#!/usr/bin/env python3
"""
make_fig_auditcard.py -- regenerate Figure 1 (the two worked audit records).

Reviewer 4: "Figure 1 has overlapping text under the axes, which should be
fixed." It does: the per-panel provenance line collided with the x-axis label.

Every value below is read from the released event logs, not transcribed from
the previous figure:

  dancetrack0013 frame 282   stage 6_revive, revived
  dancetrack0076 frame 1063  stage 6_new, margin_failed margin=0.000<0.03

Run:  python3 make_fig_auditcard.py <path-to-results-dir> <out.pdf>
With no arguments it falls back to the values asserted below, which are
checked against the logs when the directory is given.
"""

import ast
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (style parity)

# (sequence, frame, panel title, color, annotation lines)
PANELS = [
    dict(seq="dancetrack0013", frame="282",
         title="ACCEPTED: identity revived", color="#1a7a3c",
         sims=[(8, 0.9214766), (5, 0.4205317), (4, 0.3509156),
               (6, 0.3416868), (3, 0.2622857)],
         note=[r"margin $0.501 > \delta{=}0.03$",
               "decision: revive (gate cleared)"]),
    dict(seq="dancetrack0076", frame="1063",
         title="REFUSED: ambiguous, flagged", color="#b5122e",
         sims=[(87, 0.8744650), (52, 0.8744640), (104, 0.8739559),
               (53, 0.8702034), (48, 0.8680352)],
         note=[r"margin $0.000 < \delta{=}0.03$",
               r"decision: refuse $\rightarrow$ open new ID 122"]),
]


def verify(results_dir):
    """Re-read the two records so the figure cannot drift from the logs."""
    for p in PANELS:
        path = os.path.join(results_dir, p["seq"] + "_ids_events.csv")
        found = None
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["frame"] == p["frame"] and (r.get("all_sims_top5") or "").strip():
                    found = r
                    break
        if found is None:
            sys.exit("no logged record at %s frame %s" % (p["seq"], p["frame"]))
        logged = [(int(i), float(v))
                  for i, v in ast.literal_eval(found["all_sims_top5"])]
        for (a_id, a_v), (b_id, b_v) in zip(logged, p["sims"]):
            if a_id != b_id or abs(a_v - b_v) > 1e-6:
                sys.exit("figure disagrees with the log at %s frame %s: "
                         "logged %s, figure %s" % (p["seq"], p["frame"],
                                                   logged, p["sims"]))
        p["sims"] = logged
        print("verified %s frame %s against the log" % (p["seq"], p["frame"]))


def build(out_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "mathtext.fontset": "dejavuserif",
    })

    # Taller than the original and with explicit spacing, which is what fixes
    # the collision: the provenance line now has a row of its own.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    fig.subplots_adjust(top=0.80, bottom=0.30, wspace=0.34)

    fig.suptitle("Two association decisions read directly from the evidence log"
                 " (test submission)", y=0.97, fontsize=9.5)

    for ax, p in zip(axes, PANELS):
        ids = [f"ID {i}" for i, _ in p["sims"]]
        vals = [v for _, v in p["sims"]]
        ypos = list(range(len(vals)))[::-1]

        colors = [p["color"]] + ["#c9c9c9"] * (len(vals) - 1)
        ax.barh(ypos, vals, color=colors, edgecolor="#4d4d4d", height=0.62,
                linewidth=0.7)

        for y, v in zip(ypos, vals):
            ax.text(v + 0.015, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=8.2)

        ax.set_yticks(ypos)
        ax.set_yticklabels(ids, fontsize=8.4)
        ax.set_xlim(0.0, 1.14)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="x", labelsize=8.2)
        ax.set_xlabel("gallery cosine similarity", fontsize=8.6, labelpad=3)
        ax.set_title(p["title"], color=p["color"], fontweight="bold",
                     fontsize=9, pad=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-2.05, len(vals) - 0.35)

        ax.text(0.03, -1.42, "\n".join(p["note"]), fontsize=7.9,
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                          edgecolor=p["color"], linewidth=0.9))

        # The provenance line sits below the axis label in figure
        # coordinates, so it can no longer collide with it.
        ax.annotate(f"{p['seq']}, frame {p['frame']}  (track {p['sims'][0][0]})",
                    xy=(0.5, -0.33), xycoords="axes fraction",
                    ha="center", va="top", fontsize=7.4, color="#555555")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out_path)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
        verify(sys.argv[1])
        out = sys.argv[2] if len(sys.argv) > 2 else "auditcard.pdf"
    else:
        out = sys.argv[1] if len(sys.argv) > 1 else "auditcard.pdf"
    build(out)

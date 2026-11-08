#!/usr/bin/env python3
"""
make_fig_adaptchart.py -- regenerate Figure 3 (the attributed adaptation chain).

Reviewer 4: the "ByteTrack ~ 47" reference line should state its provenance.
On checking it could not, as drawn. The line was labelled "(val, same
detector)", but the only ByteTrack DanceTrack figure the paper has is 47.7,
which Table 8 reports as a *published* number cited to the ByteTrack paper,
not a validation run on our detector. The line is therefore relabelled as the
published figure it is, and the approximation is replaced by the exact value.

Every other value is taken from the component study and the DanceTrack test
table, and the script asserts the chain is internally consistent before it
draws anything:

    42.41  zero-shot
    +1.87  domain-trained detector      -> 44.28
    +0.96  stage-one appearance fusion  -> 45.24
    +0.59  observation-centric recovery -> 45.83   (cumulative +3.42)
    45.7   frozen configuration, test server
    41.25  with OCM added      (falsified, -4.58)
    45.53  with adaptive weighting (neutral, -0.30)

Run:  python3 make_fig_adaptchart.py <out.pdf>
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [
    ("zero-shot",            42.41, None),
    ("+detector",            44.28, +1.87),
    ("+re-ID\n(fusion)",     45.24, +0.96),
    ("+recovery",            45.83, +0.59),
]
TEST_SERVER = 45.70
OCM = 41.25          # falsified, -4.58 from 45.83
AW = 45.53           # neutral,   -0.30 from 45.83
BYTETRACK = 47.7     # published DanceTrack figure, Table 8, cited to [1]


def check():
    """The chain must add up, or the figure is drawing something untrue."""
    val = STEPS[0][1]
    for label, got, delta in STEPS[1:]:
        val = round(val + delta, 2)
        assert abs(val - got) < 1e-6, (label, val, got)
    assert abs(round(STEPS[-1][1] - STEPS[0][1], 2) - 3.42) < 1e-6, "cumulative"
    assert abs(round(STEPS[-1][1] - 4.58, 2) - OCM) < 1e-6, "OCM"
    assert abs(round(STEPS[-1][1] - 0.30, 2) - AW) < 1e-6, "AW"
    print("chain consistent: 42.41 +1.87 +0.96 +0.59 = 45.83 (+3.42 cumulative)")
    print("falsified OCM 41.25 (-4.58), neutral AW 45.53 (-0.30)")


def build(out_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "mathtext.fontset": "dejavuserif",
    })
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(top=0.90, bottom=0.21, left=0.14, right=0.97)

    xs = list(range(len(STEPS)))
    ys = [s[1] for s in STEPS]

    ax.plot(xs, ys, "-o", color="#1f6fb4", linewidth=2.0, markersize=6,
            zorder=3)

    for x, (label, y, delta) in zip(xs, STEPS):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.4)
        if delta is not None:
            ax.annotate(f"+{delta:.2f}", xy=(x - 0.5, (y + ys[x - 1]) / 2),
                        xytext=(6, -11), textcoords="offset points",
                        ha="left", fontsize=8.4, color="#1a7a3c",
                        fontweight="bold")

    # Reference line, now with its provenance stated on the figure.
    ax.axhline(BYTETRACK, linestyle="--", color="#808080", linewidth=1.2,
               zorder=1)
    ax.annotate(f"ByteTrack {BYTETRACK} (published, Table 8)",
                xy=(0.02, BYTETRACK), xytext=(0, 4),
                textcoords="offset points", fontsize=7.9, color="#555555")

    ax.plot([3.42], [TEST_SERVER], marker="*", markersize=17,
            color="#d9a520", zorder=4, linestyle="none")
    ax.annotate(f"test server\n{TEST_SERVER:.1f} (frozen)", xy=(3.42, TEST_SERVER),
                xytext=(6, -22), textcoords="offset points", fontsize=8.1,
                color="#9a7714", fontweight="bold")

    ax.plot([3.0], [AW], marker="X", markersize=9, color="#b5122e",
            linestyle="none", zorder=4)
    ax.annotate(f"+AW/DA {AW} (neutral)", xy=(3.0, AW), xycoords="data",
                xytext=(1.05, 46.75), textcoords="data",
                fontsize=7.9, color="#b5122e", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#b5122e",
                                linewidth=0.8, shrinkA=2, shrinkB=5))

    ax.plot([3.0], [OCM], marker="X", markersize=9, color="#b5122e",
            linestyle="none", zorder=4)
    ax.annotate(f"+OCM {OCM} (falsified)", xy=(3.0, OCM), xytext=(8, -3),
                textcoords="offset points", fontsize=7.9, color="#b5122e")

    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in STEPS], fontsize=8.6)
    ax.set_xlim(-0.35, 4.25)
    ax.set_ylim(40.8, 48.4)
    ax.set_ylabel("HOTA (DanceTrack val)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out_path)


if __name__ == "__main__":
    check()
    build(sys.argv[1] if len(sys.argv) > 1 else "adaptchart.pdf")

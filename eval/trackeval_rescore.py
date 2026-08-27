#!/usr/bin/env python3
"""
DAAT — two-pass TrackEval re-scoring of raw tracker outputs.

Pass A scores trackers that produced results for all 21 MOT17 train
sequence-detector pairs under the full protocol. Pass B scores the
buffer-size sweep trackers on their COMMON subset of sequences, with the
full-coverage trackers re-scored on that same subset so the two groups are
directly comparable.

Expected layout (override with environment variables):
  $DATA_ROOT/MOT17/train/<SEQ>/gt/gt.txt, seqinfo.ini   -- MOT17 ground truth
  $RESULTS_ROOT/<tracker_name>/train/<SEQ>.txt          -- raw MOT-format outputs

Usage:
  DATA_ROOT=/path/to/data RESULTS_ROOT=/path/to/TEST python eval/trackeval_rescore.py
"""
import os
import shutil

import numpy as np

# --- NumPy 2.0 compatibility shims (TrackEval/motmetrics use removed aliases) ---
np.float = float
np.int = int
np.bool = bool

import pandas as pd
import trackeval

DATA_ROOT = os.environ.get("DATA_ROOT", "/content/data")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/content/work/TEST")
OUT_CSV = os.environ.get("OUT_CSV", "appearance_module_results.csv")
MOT17_TRAIN_GT = f"{DATA_ROOT}/MOT17/train"

APP_FOLDERS = {
    "FastTracker-REPRO": f"{RESULTS_ROOT}/FastTracker-REPRO/train",
    "AppGallery": f"{RESULTS_ROOT}/AppGalleryTracker/train",
    "AppSwap": f"{RESULTS_ROOT}/AppSwapTracker/train",
    "AppSwap_b010": f"{RESULTS_ROOT}/AppSwap_b010/train",
    "AppSwap_b015": f"{RESULTS_ROOT}/AppSwap_b015/train",
    "AppSwap_b020": f"{RESULTS_ROOT}/AppSwap_b020/train",
    "AppSwap_b025": f"{RESULTS_ROOT}/AppSwap_b025/train",
    "AppGallery_b015": f"{RESULTS_ROOT}/AppGallery_b015/train",
    "AppGallery_b020": f"{RESULTS_ROOT}/AppGallery_b020/train",
    "AppGallery_b025": f"{RESULTS_ROOT}/AppGallery_b025/train",
    "AppGallery_b030": f"{RESULTS_ROOT}/AppGallery_b030/train",
}


def discover_coverage():
    cov = {}
    for name, fold in APP_FOLDERS.items():
        if os.path.isdir(fold):
            cov[name] = {
                os.path.splitext(f)[0] for f in os.listdir(fold) if f.endswith(".txt")
            }
    return cov


def run_eval(pass_name, trackers, seqs, workdir="/tmp"):
    ev_root = f"{workdir}/EVAL_{pass_name}"
    shutil.rmtree(ev_root, ignore_errors=True)
    gt = f"{ev_root}/gt/mot_challenge/MOT17-train"
    tr = f"{ev_root}/trackers/mot_challenge/MOT17-train"
    sm = f"{ev_root}/gt/mot_challenge/seqmaps"
    for d in (gt, tr, sm):
        os.makedirs(d, exist_ok=True)
    seqs = sorted(seqs)
    for s in seqs:
        os.makedirs(f"{gt}/{s}/gt", exist_ok=True)
        shutil.copy(f"{MOT17_TRAIN_GT}/{s}/gt/gt.txt", f"{gt}/{s}/gt/gt.txt")
        shutil.copy(f"{MOT17_TRAIN_GT}/{s}/seqinfo.ini", f"{gt}/{s}/seqinfo.ini")
    with open(f"{sm}/MOT17-train.txt", "w") as f:
        f.write("name\n" + "\n".join(seqs) + "\n")
    for name in trackers:
        os.makedirs(f"{tr}/{name}/data", exist_ok=True)
        for s in seqs:
            shutil.copy(f"{APP_FOLDERS[name]}/{s}.txt", f"{tr}/{name}/data/{s}.txt")

    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(
        dict(PRINT_CONFIG=False, TIME_PROGRESS=False, USE_PARALLEL=False)
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        dict(
            GT_FOLDER=f"{ev_root}/gt/mot_challenge",
            TRACKERS_FOLDER=f"{ev_root}/trackers/mot_challenge",
            BENCHMARK="MOT17",
            SPLIT_TO_EVAL="train",
            DO_PREPROC=True,
            PRINT_CONFIG=False,
            SEQMAP_FILE=f"{sm}/MOT17-train.txt",
        )
    )
    res, _ = trackeval.Evaluator(ev_cfg).evaluate(
        [trackeval.datasets.MotChallenge2DBox(ds_cfg)],
        [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()],
    )
    rows = []
    for name in trackers:
        r = res["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
        rows.append(
            dict(
                protocol=pass_name,
                exp=name,
                n_seqs=len(seqs),
                HOTA=100 * r["HOTA"]["HOTA"].mean(),
                AssA=100 * r["HOTA"]["AssA"].mean(),
                DetA=100 * r["HOTA"]["DetA"].mean(),
                MOTA=100 * r["CLEAR"]["MOTA"],
                IDF1=100 * r["Identity"]["IDF1"],
                IDs=int(r["CLEAR"]["IDSW"]),
            )
        )
    return rows


def main():
    cov = discover_coverage()
    full = {n for n, s in cov.items() if len(s) == 21}
    sweeps = {n for n in cov if n not in full}
    common = set.intersection(*[cov[n] for n in sweeps]) if sweeps else set()
    print("Full-coverage trackers :", sorted(full))
    print("Sweep trackers         :", sorted(sweeps))
    print(f"Common sweep subset ({len(common)}):", sorted(common))

    rows = run_eval("FULL21", sorted(full), cov[next(iter(full))])
    if common:
        rows += run_eval("SUBSET", sorted(full | sweeps), common)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print(table.to_string())
    for proto in table.protocol.unique():
        print(f"\n% ---- LaTeX rows: {proto} ----")
        for _, r in table[table.protocol == proto].iterrows():
            print(
                f"{r['exp'].replace('_', ' ')} & {r.HOTA:.2f} & {r.MOTA:.2f} "
                f"& {r.IDF1:.2f} & {r.AssA:.2f} & {r.IDs} \\\\"
            )


if __name__ == "__main__":
    main()

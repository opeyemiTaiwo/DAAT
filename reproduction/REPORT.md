# Independent verification of the DAAT association-layer audit log

Clean-room verifier written from `SPEC.md` v1.0 (frozen 7 November 2026) and the 35 event logs alone. The question answered per record is the one Section 6 poses: **does the recorded outcome follow from the operands recorded alongside it, under the rule the record names?** This is a check of log fidelity, not of tracking correctness.

## Scope

- Logs read: **35** files, **12016** records.
- Verifier: `verify_independent.py`, standard library only, written from `SPEC.md` v1.0 alone.

## 1. Tolerance, and why

SPEC.md Section 2 states that numeric columns print at four decimals and
reason-string operands at three, and that "both roundings are lossy". A single
global tolerance is the wrong instrument for that, because the two print widths
differ by a factor of ten and some of the quantities being compared are not
printed at all — they are exact constants from Section 4, or they arrive at
full precision inside `all_sims_top5`. So rather than one constant, this
verifier attaches an uncertainty to each operand and adds the two uncertainties
at each comparison:

| operand kind | uncertainty | justification |
|---|---|---|
| 4-decimal column (`cosine_sim`, `iou_cost`, `fused_cost`, …) | `5e-5` | half-ulp of a 4-decimal print |
| 3-decimal reason-string operand (`sim=`, `theta=`, `margin=`, `iou=`) | `5e-4` | half-ulp of a 3-decimal print |
| similarity from `all_sims_top5` | `1e-7` | printed at full repr; only float32 embedding round-off remains |
| configuration constant (`0.50`, `0.80`, `0.2`, the `delta` set, the `theta` clamp bounds) | `0` | Section 4 states them exactly; they are not measurements |

The effective tolerance is therefore the sum of the two operands'
uncertainties, which ranges from **`2e-7`** (a full-precision margin against an
exact Section 4.3 constant) to **`5.5e-4`** (a 3-decimal reason operand against
a 4-decimal column). The widest comparison any check in this verifier performs
is `5.5e-4`; the one that decides most of the interesting cases is `2e-7`.

Three further decisions follow from this, and they matter for the counts:

**Full precision is preferred wherever it exists.** `all_sims_top5` carries
similarities at full `repr` precision, so the best similarity and the
top-minus-runner-up margin are taken from it rather than from the 4-decimal
`cosine_sim` / `cosine_margin` columns. This is not cosmetic. Two `gate_failed`
records — `dancetrack0022` frame 1142 and `dancetrack0070` frame 1299 — print
`sim=0.650<theta=0.650` with `cosine_sim` and `cosine_theta` both `0.65`. On
the printed values the assertion "strictly below" is unresolvable, and an exact
comparison would reject both. At full precision the similarities are
`0.649987102` and `0.649970412`, so both genuinely clear the strict inequality
and are decided as **holds** with no appeal to tolerance at all. Conversely the
single `margin_failed` failure below is only *visible* at full precision: its
reason string prints `margin=0.040<0.04`, which looks fine, while the true
margin is `0.040419`.

**A threshold at a clamp boundary is treated as exact.** Section 4.2 defines
`theta = min(max(s_min, 0.65), 0.80)`. A logged `cosine_theta` of `0.65` or
`0.80` is therefore the clamp itself, an exact constant, not a rounded
measurement — so it carries zero uncertainty. 1526 of the 1957 appearance
records sit on the `0.65` floor and 11 on the `0.80` ceiling, so this applies
to 78% of them and is what makes their strict inequalities decidable.

**Strict inequalities fail only when the violation exceeds the tolerance.** For
an assertion `a < b`, the verdict is `fails` only when `a - b` exceeds the
summed uncertainty; when the ordering cannot be resolved within the print
precision the record is counted as **holds** and listed separately as
*borderline*, on the principle that operands which merely fail to confirm a
rule do not contradict it. That bucket turned out to be **empty** — every
comparison in all 35 logs resolved one way or the other, because full-precision
similarities and exact configuration constants eliminated every tie. This is
reported so the reader knows the result does not rest on a tolerance judgement.

**`untestable` is reserved for an absent operand** and nothing else. It is
never used for an indecisive comparison, and it is never counted as a pass.

## 2. Per-reason results

| reason | examined | holds | fails | untestable |
|---|---:|---:|---:|---:|
| `byte_second_assoc` | 5097 | 5097 | 0 | 0 |
| `re_activate_from_lost` | 3571 | 3571 | 0 | 0 |
| `unconfirmed_update` | 944 | 944 | 0 | 0 |
| `revived` | 800 | 782 | 5 | 13 |
| `margin_failed` | 726 | 725 | 1 | 0 |
| `gate_failed` | 431 | 431 | 0 | 0 |
| `ocr_last_obs_iou` | 444 | 444 | 0 | 0 |
| `registry_empty_or_immature` | 3 | 3 | 0 | 0 |
| **total** | **12016** | **11997** | **6** | **13** |

Untestable records, by the operand that is missing:

- `revived`: 13 -- delta_in_force (no margin_failed record in file)

## 3. Failures, itemized

6 failing record(s).

1. `dancetrack0003_ids_events.csv` frame 110 | revived | margin decisive does NOT hold (-1.515e-03) | best_sim=0.669012 (all_sims_top5[0])  cosine_theta=0.650000  sim-theta=+1.901e-02  margin=0.048485 (all_sims_top5 top-minus-runner-up)  delta_in_force=0.050000 (delta(recovered per-file))  margin-delta=-1.515e-03
2. `dancetrack0003_ids_events.csv` frame 144 | revived | margin decisive does NOT hold (-3.490e-03) | best_sim=0.738068 (all_sims_top5[0])  cosine_theta=0.678200  sim-theta=+5.987e-02  margin=0.046510 (all_sims_top5 top-minus-runner-up)  delta_in_force=0.050000 (delta(recovered per-file))  margin-delta=-3.490e-03
3. `dancetrack0003_ids_events.csv` frame 930 | revived | margin decisive does NOT hold (-3.521e-03) | best_sim=0.800557 (all_sims_top5[0])  cosine_theta=0.650000  sim-theta=+1.506e-01  margin=0.046479 (all_sims_top5 top-minus-runner-up)  delta_in_force=0.050000 (delta(recovered per-file))  margin-delta=-3.521e-03
4. `dancetrack0056_ids_events.csv` frame 896 | revived | margin decisive does NOT hold (-7.903e-04) | best_sim=0.768415 (all_sims_top5[0])  cosine_theta=0.650000  sim-theta=+1.184e-01  margin=0.039210 (all_sims_top5 top-minus-runner-up)  delta_in_force=0.040000 (delta(recovered per-file))  margin-delta=-7.903e-04
5. `dancetrack0070_ids_events.csv` frame 686 | revived | margin decisive does NOT hold (-1.966e-03) | best_sim=0.759042 (all_sims_top5[0])  cosine_theta=0.650000  sim-theta=+1.090e-01  margin=0.038034 (all_sims_top5 top-minus-runner-up)  delta_in_force=0.040000 (delta(recovered per-file))  margin-delta=-1.966e-03
6. `dancetrack0089_ids_events.csv` frame 408 | margin_failed margin=0.040<0.04 | margin is NOT strictly below the margin in force [reading-dependent: fails because 4.3 enumerates delta exactly; would hold if the reason right-hand side were a 3-decimal print of a continuous delta] | margin=0.040419 (all_sims_top5 top-minus-runner-up)  delta_in_force=0.040000 (delta(4.3 exact))  margin-delta=+4.192e-04  tolerance=2.0e-07  cosine_margin_col=0.0404  reason_margin=0.040

### Borderline, decided by the tolerance

Records whose assertion an exact comparison of the printed values would reject, but which the stated tolerance cannot resolve. Counted as **holds** -- the operands do not demonstrably contradict the rule.

None.

### Composite cost, SPEC.md 4.7 (neither pass nor fail)

- **2646** -- appearance selected (fused_cost < c1) but motion term indeterminable: not decomposable
- **923** -- motion term provably present (fused_cost > c1): not decomposable
- **376** --   of which 1-2*fused_cost would be a negative similarity (contradicts invariant 2)
- **2** -- fused_cost == c1: consistent with no appearance and no motion term

### Advisory checks (implied by SPEC.md, not stated as assertions in S6)

- `byte_det_score_not_low`: **814** record(s)
- `immature_feat_norm_present`: **3** record(s)
- `delta_early_mismatch`: **1** record(s)

<details><summary><code>byte_det_score_not_low</code> -- 814 record(s)</summary>

- `dancetrack0003_ids_events.csv` frame 11 | byte_second_assoc iou=0.892 score=0.515 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5146  iou_cost=0.1081
- `dancetrack0003_ids_events.csv` frame 102 | byte_second_assoc iou=0.652 score=0.537 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5368  iou_cost=0.3478
- `dancetrack0003_ids_events.csv` frame 139 | byte_second_assoc iou=0.625 score=0.508 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5077  iou_cost=0.3752
- `dancetrack0003_ids_events.csv` frame 200 | byte_second_assoc iou=0.718 score=0.514 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5137  iou_cost=0.2825
- `dancetrack0003_ids_events.csv` frame 234 | byte_second_assoc iou=0.553 score=0.514 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5144  iou_cost=0.4469
- `dancetrack0003_ids_events.csv` frame 236 | byte_second_assoc iou=0.515 score=0.531 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5307  iou_cost=0.4851
- `dancetrack0003_ids_events.csv` frame 306 | byte_second_assoc iou=0.665 score=0.507 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5070  iou_cost=0.3349
- `dancetrack0003_ids_events.csv` frame 328 | byte_second_assoc iou=0.613 score=0.520 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5196  iou_cost=0.3871
- `dancetrack0003_ids_events.csv` frame 410 | byte_second_assoc iou=0.829 score=0.542 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5416  iou_cost=0.1705
- `dancetrack0003_ids_events.csv` frame 455 | byte_second_assoc iou=0.747 score=0.527 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5270  iou_cost=0.2528
- `dancetrack0003_ids_events.csv` frame 615 | byte_second_assoc iou=0.844 score=0.550 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5495  iou_cost=0.1558
- `dancetrack0003_ids_events.csv` frame 624 | byte_second_assoc iou=0.589 score=0.524 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5243  iou_cost=0.4111
- `dancetrack0003_ids_events.csv` frame 626 | byte_second_assoc iou=0.953 score=0.514 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5136  iou_cost=0.0471
- `dancetrack0003_ids_events.csv` frame 633 | byte_second_assoc iou=0.847 score=0.543 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5425  iou_cost=0.1535
- `dancetrack0003_ids_events.csv` frame 744 | byte_second_assoc iou=0.588 score=0.506 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5058  iou_cost=0.4121
- `dancetrack0003_ids_events.csv` frame 766 | byte_second_assoc iou=0.834 score=0.540 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5399  iou_cost=0.1664
- `dancetrack0003_ids_events.csv` frame 767 | byte_second_assoc iou=0.956 score=0.506 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5059  iou_cost=0.0440
- `dancetrack0003_ids_events.csv` frame 799 | byte_second_assoc iou=0.761 score=0.521 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5210  iou_cost=0.2390
- `dancetrack0003_ids_events.csv` frame 801 | byte_second_assoc iou=0.849 score=0.544 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5441  iou_cost=0.1506
- `dancetrack0003_ids_events.csv` frame 810 | byte_second_assoc iou=0.869 score=0.510 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5105  iou_cost=0.1305
- `dancetrack0003_ids_events.csv` frame 810 | byte_second_assoc iou=0.633 score=0.548 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5477  iou_cost=0.3671
- `dancetrack0003_ids_events.csv` frame 821 | byte_second_assoc iou=0.640 score=0.509 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5085  iou_cost=0.3603
- `dancetrack0003_ids_events.csv` frame 825 | byte_second_assoc iou=0.779 score=0.525 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5251  iou_cost=0.2205
- `dancetrack0003_ids_events.csv` frame 855 | byte_second_assoc iou=0.622 score=0.505 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5050  iou_cost=0.3784
- `dancetrack0003_ids_events.csv` frame 868 | byte_second_assoc iou=0.821 score=0.504 | det_score is not below the 0.50 detection threshold (SPEC 4.6) | det_score=0.5043  iou_cost=0.1786
- ... and 789 more

</details>

<details><summary><code>immature_feat_norm_present</code> -- 3 record(s)</summary>

- `dancetrack0042_ids_events.csv` frame 3 | registry_empty_or_immature | feat_norm is populated; whether it counts as an 'appearance field' is not stated in SPEC.md | feat_norm=1.0  n_registry=0
- `dancetrack0042_ids_events.csv` frame 4 | registry_empty_or_immature | feat_norm is populated; whether it counts as an 'appearance field' is not stated in SPEC.md | feat_norm=1.0  n_registry=0
- `dancetrack0093_ids_events.csv` frame 5 | registry_empty_or_immature | feat_norm is populated; whether it counts as an 'appearance field' is not stated in SPEC.md | feat_norm=1.0  n_registry=0

</details>

<details><summary><code>delta_early_mismatch</code> -- 1 record(s)</summary>

- `dancetrack0093_ids_events.csv` frame 30 | margin_failed margin=0.012<0.03 | frame <= 30 but delta in force is not 0.05 (SPEC 4.3) | frame=30.0  delta=0.03

</details>

### Structural checks (Section 2 definitions, Section 3 pairing, Section 5 invariants)

| check | violations |
|---|---:|
| Section 5 `inv1_iou_sum` | 0 |
| Section 5 `inv2_range` | 0 |
| Section 5 `inv3_sim_is_top` | 0 |
| Section 5 `inv4_margin` | 0 |
| Section 5 `inv5_sorted` | 0 |
| Section 5 `inv6_frame_order` | 0 |
| Section 5 `inv7_theta_range` | 0 |
| `reason_format_undocumented` | 444 |
| `reason_operands_undocumented` | 5097 |
| `sec2_box_iou_mismatch` | 9537 |

Section 2 defines `iou_cost` as *1 minus the IoU overlap between `det_box` and `track_box`*. Recomputing that IoU from the two logged boxes: **9537 of 10140** records with both boxes present disagree with the logged `iou_overlap` by more than 0.01 (94.1%).

- `dancetrack0003_ids_events.csv` frame 4 | unconfirmed_update | IoU(det_box,track_box)=0.9726 but iou_overlap=0.9444 (+0.0282) | det_box=[1027.3, 571.4, 1080.0, 820.8]  track_box=[1026.9, 571.3, 1081.0, 820.5]  iou_overlap=0.9444
- `dancetrack0003_ids_events.csv` frame 7 | unconfirmed_update | IoU(det_box,track_box)=0.9834 but iou_overlap=0.9209 (+0.0625) | det_box=[738.5, 574.1, 849.2, 814.1]  track_box=[739.2, 573.1, 849.8, 813.9]  iou_overlap=0.9209
- `dancetrack0003_ids_events.csv` frame 11 | byte_second_assoc iou=0.892 score=0.515 | IoU(det_box,track_box)=0.9174 but iou_overlap=0.8919 (+0.0255) | det_box=[1024.0, 569.0, 1100.9, 817.4]  track_box=[1026.1, 568.7, 1097.1, 816.1]  iou_overlap=0.8919
- `dancetrack0003_ids_events.csv` frame 16 | unconfirmed_update | IoU(det_box,track_box)=0.8564 but iou_overlap=0.8311 (+0.0253) | det_box=[928.8, 646.3, 1038.2, 828.9]  track_box=[921.8, 645.9, 1049.2, 828.8]  iou_overlap=0.8311
- `dancetrack0003_ids_events.csv` frame 17 | byte_second_assoc iou=0.577 score=0.490 | IoU(det_box,track_box)=0.5913 but iou_overlap=0.5770 (+0.0143) | det_box=[928.1, 647.3, 1002.4, 828.2]  track_box=[907.2, 647.0, 1032.5, 828.4]  iou_overlap=0.577
- `dancetrack0003_ids_events.csv` frame 18 | re_activate_from_lost | IoU(det_box,track_box)=0.9603 but iou_overlap=0.9267 (+0.0336) | det_box=[1024.0, 568.4, 1094.2, 795.2]  track_box=[1025.4, 568.6, 1093.1, 796.0]  iou_overlap=0.9267
- `dancetrack0003_ids_events.csv` frame 19 | byte_second_assoc iou=0.776 score=0.115 | IoU(det_box,track_box)=0.8702 but iou_overlap=0.7757 (+0.0945) | det_box=[1028.0, 570.7, 1106.3, 799.2]  track_box=[1030.1, 570.0, 1098.9, 797.6]  iou_overlap=0.7757
- `dancetrack0003_ids_events.csv` frame 20 | byte_second_assoc iou=0.891 score=0.130 | IoU(det_box,track_box)=0.9136 but iou_overlap=0.8910 (+0.0226) | det_box=[1028.7, 567.3, 1104.3, 796.5]  track_box=[1031.3, 568.3, 1100.7, 796.4]  iou_overlap=0.891
- `dancetrack0003_ids_events.csv` frame 25 | byte_second_assoc iou=0.820 score=0.282 | IoU(det_box,track_box)=0.9343 but iou_overlap=0.8205 (+0.1138) | det_box=[1035.5, 570.0, 1109.7, 816.1]  track_box=[1033.9, 570.0, 1107.0, 813.7]  iou_overlap=0.8205
- `dancetrack0003_ids_events.csv` frame 28 | re_activate_from_lost | IoU(det_box,track_box)=0.6697 but iou_overlap=0.5360 (+0.1337) | det_box=[927.5, 645.0, 993.6, 828.9]  track_box=[910.8, 645.0, 1009.5, 828.9]  iou_overlap=0.536
- `dancetrack0003_ids_events.csv` frame 29 | byte_second_assoc iou=0.799 score=0.311 | IoU(det_box,track_box)=0.9199 but iou_overlap=0.7986 (+0.1213) | det_box=[922.7, 642.6, 1014.5, 830.2]  track_box=[915.7, 643.3, 1014.9, 829.7]  iou_overlap=0.7986
- `dancetrack0003_ids_events.csv` frame 32 | byte_second_assoc iou=0.918 score=0.232 | IoU(det_box,track_box)=0.9715 but iou_overlap=0.9176 (+0.0539) | det_box=[918.0, 631.8, 1009.8, 831.6]  track_box=[918.8, 633.1, 1010.6, 830.6]  iou_overlap=0.9176
- `dancetrack0003_ids_events.csv` frame 35 | re_activate_from_lost | IoU(det_box,track_box)=0.7601 but iou_overlap=0.4112 (+0.3489) | det_box=[913.3, 597.7, 997.0, 838.4]  track_box=[912.2, 597.7, 1020.7, 842.0]  iou_overlap=0.4112
- `dancetrack0003_ids_events.csv` frame 42 | re_activate_from_lost | IoU(det_box,track_box)=0.8355 but iou_overlap=0.5157 (+0.3198) | det_box=[1053.0, 569.0, 1170.5, 818.8]  track_box=[1062.5, 569.0, 1160.7, 818.9]  iou_overlap=0.5157
- `dancetrack0003_ids_events.csv` frame 43 | re_activate_from_lost | IoU(det_box,track_box)=0.8374 but iou_overlap=0.6852 (+0.1522) | det_box=[1053.7, 568.4, 1134.7, 811.4]  track_box=[1060.3, 568.4, 1128.2, 811.7]  iou_overlap=0.6852
- `dancetrack0003_ids_events.csv` frame 99 | byte_second_assoc iou=0.580 score=0.178 | IoU(det_box,track_box)=0.6268 but iou_overlap=0.5804 (+0.0464) | det_box=[1040.2, 589.3, 1149.5, 826.9]  track_box=[1059.2, 583.7, 1129.3, 825.0]  iou_overlap=0.5804
- `dancetrack0003_ids_events.csv` frame 100 | byte_second_assoc iou=0.709 score=0.143 | IoU(det_box,track_box)=0.8901 but iou_overlap=0.7092 (+0.1809) | det_box=[911.9, 593.3, 978.1, 816.1]  track_box=[913.7, 577.2, 977.8, 820.2]  iou_overlap=0.7092
- `dancetrack0003_ids_events.csv` frame 102 | byte_second_assoc iou=0.652 score=0.537 | IoU(det_box,track_box)=0.8546 but iou_overlap=0.6522 (+0.2024) | det_box=[1069.9, 547.1, 1141.4, 816.1]  track_box=[1063.7, 561.7, 1142.4, 818.9]  iou_overlap=0.6522
- `dancetrack0003_ids_events.csv` frame 102 | byte_second_assoc iou=0.732 score=0.246 | IoU(det_box,track_box)=0.8549 but iou_overlap=0.7317 (+0.1232) | det_box=[917.3, 561.6, 995.6, 827.6]  track_box=[920.3, 567.7, 989.5, 825.0]  iou_overlap=0.7317
- `dancetrack0003_ids_events.csv` frame 103 | byte_second_assoc iou=0.858 score=0.456 | IoU(det_box,track_box)=0.8772 but iou_overlap=0.8583 (+0.0189) | det_box=[1070.6, 548.1, 1142.1, 816.8]  track_box=[1065.3, 552.4, 1145.4, 817.6]  iou_overlap=0.8583
- `dancetrack0003_ids_events.csv` frame 111 | byte_second_assoc iou=0.913 score=0.465 | IoU(det_box,track_box)=0.9743 but iou_overlap=0.9127 (+0.0616) | det_box=[1058.4, 554.2, 1138.1, 810.0]  track_box=[1057.9, 552.9, 1138.0, 813.5]  iou_overlap=0.9127
- `dancetrack0003_ids_events.csv` frame 125 | re_activate_from_lost | IoU(det_box,track_box)=0.7550 but iou_overlap=0.6897 (+0.0653) | det_box=[792.5, 554.2, 889.7, 829.6]  track_box=[775.7, 555.2, 902.7, 832.6]  iou_overlap=0.6897
- `dancetrack0003_ids_events.csv` frame 137 | byte_second_assoc iou=0.647 score=0.293 | IoU(det_box,track_box)=0.8593 but iou_overlap=0.6469 (+0.2124) | det_box=[661.5, 594.0, 758.7, 827.6]  track_box=[665.0, 585.2, 766.5, 828.1]  iou_overlap=0.6469
- `dancetrack0003_ids_events.csv` frame 138 | byte_second_assoc iou=0.567 score=0.489 | IoU(det_box,track_box)=0.5976 but iou_overlap=0.5666 (+0.0310) | det_box=[657.5, 606.5, 826.2, 827.6]  track_box=[681.7, 600.7, 784.2, 827.9]  iou_overlap=0.5666
- `dancetrack0003_ids_events.csv` frame 139 | byte_second_assoc iou=0.625 score=0.508 | IoU(det_box,track_box)=0.6608 but iou_overlap=0.6248 (+0.0360) | det_box=[658.8, 604.8, 819.5, 827.6]  track_box=[684.2, 604.9, 790.5, 827.8]  iou_overlap=0.6248
- ... and 9512 more

### Recovered disambiguation margin, and its feasibility per sequence

SPEC.md 4.3's `delta` is not logged as a column. Section 4.1 measures `t` once at frame 30, so `delta` is one constant per sequence after frame 30; it is read off the right-hand side of that file's own `margin_failed` records and then used to test `revived`.

That also makes the whole file testable at once. Every `margin_failed` record requires `delta >` its margin; every `revived` record requires `delta <=` its margin. The intersection is the interval the true `delta` must occupy if every record in the file is faithful. An empty intersection proves the file cannot be explained by any single threshold -- a conclusion that does not depend on what the log prints.

| sequence | printed | required by the decisions | implied by the print | consistent |
|---|---|---|---|---|
| `dancetrack0003_ids_events.csv` | 0.050 | (0.0385, 0.0465] | [0.0495, 0.0505] | **NO -- EMPTY** |
| `dancetrack0009_ids_events.csv` | 0.030 | (0.0250, 0.0307] | [0.0295, 0.0305] | yes |
| `dancetrack0011_ids_events.csv` | 0.030 | (0.0217, 0.0313] | [0.0295, 0.0305] | yes |
| `dancetrack0013_ids_events.csv` | 0.050 | (0.0435, 0.0504] | [0.0495, 0.0505] | yes |
| `dancetrack0017_ids_events.csv` | 0.070 | (0.0651, 0.0751] | [0.0695, 0.0705] | yes |
| `dancetrack0021_ids_events.csv` | 0.030 | (0.0218, 0.0337] | [0.0295, 0.0305] | yes |
| `dancetrack0022_ids_events.csv` | 0.030 | (0.0285, 0.0303] | [0.0295, 0.0305] | yes |
| `dancetrack0028_ids_events.csv` | 0.030 | (0.0270, 0.0301] | [0.0295, 0.0305] | yes |
| `dancetrack0031_ids_events.csv` | 0.030 | (0.0286, 0.0303] | [0.0295, 0.0305] | yes |
| `dancetrack0036_ids_events.csv` | 0.030 | (0.0285, 0.0308] | [0.0295, 0.0305] | yes |
| `dancetrack0038_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0040_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0042_ids_events.csv` | 0.030 | (0.0296, 0.0302] | [0.0295, 0.0305] | yes |
| `dancetrack0046_ids_events.csv` | 0.030 | (0.0300, 0.0303] | [0.0295, 0.0305] | yes |
| `dancetrack0048_ids_events.csv` | 0.030 | (0.0291, 0.0308] | [0.0295, 0.0305] | yes |
| `dancetrack0050_ids_events.csv` | 0.030 | (0.0191, 0.0308] | [0.0295, 0.0305] | yes |
| `dancetrack0054_ids_events.csv` | 0.030 | (0.0200, 0.0408] | [0.0295, 0.0305] | yes |
| `dancetrack0056_ids_events.csv` | 0.040 | (0.0379, 0.0392] | [0.0395, 0.0405] | **NO -- EMPTY** |
| `dancetrack0059_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0060_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0064_ids_events.csv` | 0.030 | (0.0128, 0.0348] | [0.0295, 0.0305] | yes |
| `dancetrack0067_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0070_ids_events.csv` | 0.040 | (0.0276, 0.0380] | [0.0395, 0.0405] | **NO -- EMPTY** |
| `dancetrack0071_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0076_ids_events.csv` | 0.030 | (0.0285, 0.0304] | [0.0295, 0.0305] | yes |
| `dancetrack0078_ids_events.csv` | none | -- | -- | not testable (no `margin_failed`) |
| `dancetrack0084_ids_events.csv` | 0.030 | (0.0248, 0.0338] | [0.0295, 0.0305] | yes |
| `dancetrack0085_ids_events.csv` | 0.030 | (0.0299, 0.0301] | [0.0295, 0.0305] | yes |
| `dancetrack0088_ids_events.csv` | 0.040 | (0.0392, 0.0464] | [0.0395, 0.0405] | yes |
| `dancetrack0089_ids_events.csv` | 0.040 | (0.0404, 0.0426] | [0.0395, 0.0405] | yes |
| `dancetrack0091_ids_events.csv` | 0.050 | (0.0389, 0.0551] | [0.0495, 0.0505] | yes |
| `dancetrack0092_ids_events.csv` | 0.030 | (0.0274, 0.0347] | [0.0295, 0.0305] | yes |
| `dancetrack0093_ids_events.csv` | 0.030 | (0.0281, 0.0307] | [0.0295, 0.0305] | yes |
| `dancetrack0095_ids_events.csv` | 0.030 | (0.0299, 0.0318] | [0.0295, 0.0305] | yes |
| `dancetrack0100_ids_events.csv` | 0.030 | (0.0277, 0.0311] | [0.0295, 0.0305] | yes |


## 4. Where SPEC.md was ambiguous, incomplete, or wrong

The specification was **not** sufficient to write this verifier without
guessing. Eleven points required a decision. They are ordered by how much they
affect the counts above.

### 4.1 `revived` names no threshold for "the margin was decisive" — I had to reconstruct one

> Section 6: "**`revived`** asserts a lost track was re-associated on
> appearance evidence: the similarity cleared the gate and the margin was
> decisive. Operands: `cosine_sim`, `cosine_theta`, `cosine_margin`,
> `all_sims_top5`."

"Decisive" is not defined, and the margin it must beat is not among the listed
operands. Section 4.3 says the margin in force "is not logged as its own
column; the value in force appears as the right-hand side of the
`margin_failed` reason string" — but a `revived` record carries no reason-string
operands at all. So on a literal reading of Section 6 the second half of every
one of the 800 `revived` assertions is untestable, and the reason with the
richest appearance evidence in the log would be the one the spec makes hardest
to check.

**What I assumed.** Section 4.1 measures `t` "once at frame 30", and Section
4.3 derives `delta` from `t` alone, so `delta` is a single constant per
sequence after frame 30. I therefore recover it from the right-hand side of
that same file's `margin_failed` records and apply it to the file's `revived`
records. Where a file has no `margin_failed` record (7 of 35), I fall back to
the range Section 4.3's formula admits — `delta = 0.07 - 0.04t` with
`t = clip(...,0,1)` gives `delta` in `[0.03, 0.07]` — so a margin at or above
`0.07` is decisive whatever `t` was, and one below `0.03` is decisive under no
admissible `t`. That resolves 21 of those 34 records and leaves 13 genuinely
untestable.

**Alternatives I rejected.** (a) Leave all 800 untestable — defensible but
discards real evidence. (b) Assume the revive path uses its own, looser margin.
Nothing in the spec says it does, and the per-sequence feasibility table above
shows `revived` and `margin_failed` margins never overlap within a file, which
is what a *single* shared threshold predicts. I note, though, that if the
authors intended a separate revive margin, the 5 `revived` failures below
become untestable rather than failures. **This assumption is load-bearing for
5 of the 6 reported failures and the authors should confirm or deny it.**

### 4.2 Section 4.3's enumerated `delta` values conflict with the data

> Section 4.3: "In force: `{0.03, 0.04, 0.05, 0.07}`."

Taken literally this makes the margin an exact constant, which is what lets the
one `margin_failed` failure be adjudicated: `dancetrack0089` frame 408 has a
true margin of `0.040419`, which is not strictly below `0.04`.

But the same file's feasibility interval requires `delta` in
`(0.0404, 0.0426]`, and **no member of the enumerated set lies in it**. Since
`delta = (1-t)*0.07 + t*0.03` is continuous in `t`, a `delta` of about
`0.04045` (`t ≈ 0.7388`) explains every record in that file and prints as
`0.040` at three decimals. So the enumerated set appears to be the *printed*
values, not the values in force, and Section 4.3 conflates the two.

**What I did.** I kept the literal reading, so the record is reported as a
failure, and annotated it in the failure list as reading-dependent. Under the
continuous reading it holds and the failure count drops from 6 to 5. The
authors should state whether `{0.03, 0.04, 0.05, 0.07}` is exact or rounded;
it changes a reported failure either way.

### 4.3 Section 3's reason-string grammar does not match the log

> Section 3: ```ocr_last_obs_iou iou=<A>```

No record uses that form. All 444 use `ocr_last_obs_iou=<A> gap=<B>` — the
value is suffixed to the keyword with `=` and no space, and there is a second
operand, `gap`, that Section 3 does not mention and Section 6 does not explain.
`gap` is `0` in all 444 records, so it carries no information in this corpus;
presumably it is the lost-frame gap, but the spec never says.

Separately, Section 3 states that "the `reason` field for the two refusals and
for observation-centric recovery carries its operands inline", implying the
other five reasons do not. But all 5097 `byte_second_assoc` records carry
`iou=<A> score=<B>` inline, and Section 3 documents no grammar for them.

**What I did.** Parsed the forms the data actually uses, and reported
`reason_format_undocumented` (444) and `reason_operands_undocumented` (5097)
structurally rather than as failures, since the operands are self-consistent
with their columns in every record. A verifier built strictly to Section 3's
grammar would have thrown away the `byte_second_assoc` operands and rejected
every `ocr` record as unparseable.

### 4.4 Section 2's definition of `iou_cost` is false for 94% of the records

> Section 2: "`iou_cost` | 1 minus the IoU overlap between `det_box` and
> `track_box`"

Recomputing that IoU from the two logged boxes disagrees with the logged
`iou_overlap` by more than 0.01 in **9537 of 10140** records that have both
boxes. This is far outside anything the 1-decimal box printing could explain;
the largest disagreement is 0.735. See section 5.1 below for the pattern.

**What I did.** Treated `iou_cost` and `iou_overlap` as the given operands,
exactly as Section 6 does for every IoU-based reason, and reported the box
disagreement as a separate structural finding. Had I instead recomputed the
cost from the boxes — which Section 2 licenses — the verifier would report
thousands of failures. **Which of these two readings the authors intend is the
single largest fork in the whole exercise:** 6 failures versus several
thousand, from the same spec.

### 4.5 Section 4.7 claims a decomposition the log cannot support

> Section 4.7: "where the minimum selected `c_emb` and no motion term was
> added, the appearance cost is recoverable exactly as `fused_cost`"

The condition is not decidable from the record. With `c1 = 1 - iou_cost *
det_score` computable and `c3 = min(c1, c_emb) + 0.2 * a_OCM` logged, the only
case the record settles is `c3 > c1`, which proves a motion term *was* added
(923 records). When `c3 < c1` the minimum did select `c_emb`, but `a_OCM` cannot
be shown to be zero, so `c_emb` is only a lower bound (2646 records). The spec
describes the recoverable case as if it were identifiable; it is not. All 3569
non-trivial records therefore land in the "not decomposable" category, which is
presumably more than Section 4.7 anticipates.

Worse, the corollary is internally inconsistent: Section 4.7 says the
similarity is recoverable as `1 - 2 * fused_cost`, yet **376** of the
appearance-selected records have `fused_cost > 0.5`, for which that expression
is negative — contradicting Section 5's invariant 2 that `cosine_sim` lies in
`[0,1]`. Either the `1 - 2 * fused_cost` relation is wrong, or `c_emb` is not
`(1 - sim)/2`, or those records had a motion term after all. The spec gives no
way to tell.

**What I did.** Reported all of it in the Section 4.7 category, passing and
failing nothing, as instructed.

### 4.6 Section 4.6's second-stage scope contradicts 814 records

> Section 4.6: "The low-score second association stage operates on detections
> below the `0.50` detection threshold and accepts a cost strictly below
> `0.50`."

**814 of 5097** `byte_second_assoc` records have `det_score >= 0.50`, up to
`0.5697` (812 strictly above; 2 print exactly `0.5000` and are within
rounding). They occur in 33 of the 35 files, so this is systematic, not a few
stragglers.

The ambiguity is which clause Section 6 makes the assertion. Section 6 lists
`det_score` as an operand but then says only "the assertion is that the cost was
acceptable at that stage (Section 4.6)". **What I did.** Tested the cost — which
holds for all 5097, with `iou_cost` never exceeding `0.4999` — and reported the
`det_score` clause as an advisory with all 814 records itemized. Reading
Section 4.6's scope as part of the assertion instead would turn 814 holds into
814 failures. My guess is that unmatched high-score detections fall through to
the second stage in the implementation and Section 4.6 describes only the
intended input set, but the spec as written does not permit that reading.

### 4.7 `unconfirmed_update` has no assertion at all

> Section 6: "**`unconfirmed_update`** asserts an unconfirmed or occluded track
> was updated from a detection on overlap. Operands: `iou_cost`,
> `iou_overlap`, `det_score`."

Every other reason gets either an explicit "the assertion is that …" sentence or
a stated threshold. This one gets neither, so there is nothing to check beyond
the operands being present. **What I assumed:** that this is a first-stage
update and so Section 4.6's "at or below `0.80`" applies, plus the weaker
property that a positive overlap exists. Both hold for all 944 records (max
`iou_cost` is `0.6739`), so the reading does not change any count — but that is
luck, not specification. A reader cannot tell from Section 6 whether a record
with `iou_cost = 0.95` should fail.

### 4.8 `ocr_last_obs_iou` has no acceptance threshold

Section 6 names the operands and asserts the recovery happened "by overlap
against its own last observation", but no threshold is given anywhere for this
stage, and Section 4 does not mention stage `5b` at all. **What I did.** Made
the primary assertion the mutual consistency of the three operands (reason
`iou=` against `iou_overlap`, and `iou_cost + iou_overlap = 1`), which holds
for all 444. The observed minimum is `0.302` and the distribution stops dead
there, which strongly suggests a `0.30` gate, so I added an advisory check
against `0.30` — explicitly labelled as my assumption, not the spec's — which
nothing violates. Section 4 should name this threshold.

### 4.9 Section 4.3's frame boundary is off by one

> Section 4.3: "For the first 30 frames, `delta = 0.05`. Thereafter: …"

`dancetrack0093` has `margin_failed` records at frames 21, 22 and 27 with
`delta = 0.05` and one at **frame 30 with `delta = 0.03`**. On a 1-based frame
index (Section 2) frame 30 is within "the first 30 frames", so it should still
be `0.05`. The data says the adaptive value takes over at frame 30 inclusive —
consistent with Section 4.1 measuring `t` *at* frame 30, but not with Section
4.3's wording. **What I did.** Implemented the spec's literal reading and
reported the one disagreeing record as an advisory (`delta_early_mismatch`)
rather than a failure, since the boundary is the spec's defect, not the log's.

### 4.10 "the appearance fields are absent" is not a defined set

> Section 6: "**`registry_empty_or_immature`** … the testable property is that
> the appearance fields are absent."

Which columns count? All 3 such records have `cosine_sim`, `cosine_theta`,
`cosine_margin` and `all_sims_top5` empty — but `feat_norm` populated (`1.0`),
and `feat_norm` is an appearance quantity by any natural reading of Section 2
("L2 norm of the query embedding"). **What I did.** Tested the four comparison
outputs, which is the reading that matches the stated rationale ("no comparison
was performed"), so all 3 hold; and reported the populated `feat_norm` as an
advisory. Under the broader reading all 3 would fail. The spec should enumerate
the set.

### 4.11 Minor: "cleared the gate" strictness, and `n_min = 5`

Section 6 makes `gate_failed` "strictly below" the gate, so I read `revived`'s
"cleared the gate" as the complement, `sim >= theta`. The spec never says so
and no record sits on the boundary, so nothing turns on it here. Separately,
Section 4.4's `n_min = 5` maturity rule is not checkable: the log has no
per-track clean-frame count, and `n_registry` is a registry size, not a
maturity count. Section 6 acknowledges this for `registry_empty_or_immature`
("there is no operand to check"), but it means the one rule with a concrete
numeric constant in Section 4.4 is unverifiable.

---

## 5. What surprised me in the data

### 5.1 The logged boxes do not reproduce the logged IoU — and the pattern says which boxes are stale

This is the strongest signal in the corpus and it splits cleanly by reason:

| reason | boxes agree (≤0.01) | geometric IoU higher | geometric IoU lower | median error |
|---|---:|---:|---:|---:|
| `gate_failed` | 27 | 0 | 0 | `+0.0000` |
| `margin_failed` | 35 | 0 | 0 | `+0.0000` |
| `revived` | 22 | 0 | 0 | `+0.0000` |
| `unconfirmed_update` | 180 | 745 | 19 | `+0.0452` |
| `byte_second_assoc` | 304 | 4786 | 7 | `+0.0672` |
| `ocr_last_obs_iou` | 7 | 344 | 93 | `+0.1775` |
| `re_activate_from_lost` | 28 | 3542 | 1 | `+0.1767` |

The three appearance-driven reasons agree to within `0.0002` — essentially
exactly. The four IoU-driven reasons disagree, almost always with the
geometric IoU *higher* than the logged overlap. That asymmetry is diagnostic:
where the IoU decided the record, the logged `track_box` is not the box the IoU
was computed against; where the IoU decided nothing, it is. The obvious
explanation is that `track_box` is serialized after the track has absorbed the
detection, so it has already moved toward `det_box` and overlaps it better than
the prediction did. If that is right, Section 2's `track_box` ("predicted track
box") is mislabelled for exactly the records where it matters, and the log
cannot be audited geometrically — only the scalar `iou_cost` can be trusted.
I would want this confirmed before the log is described as reproducible.

For `ocr_last_obs_iou` there is a second reading available: Section 6 says the
recovery is against the track's "own last observation rather than its predicted
position", so the reason's `iou=` and the column `track_box` would legitimately
refer to different boxes. But the reason's `iou=` matches `iou_overlap` to
within rounding in all 444 records, so `iou_overlap` is the last-observation
IoU and `track_box` is something else again. The 93 cases where the geometric
IoU is *lower* are unique to this reason and fit that story.

### 5.2 The decisions are perfectly separable per sequence, but three sequences contradict their own printed margin

Within every file, the `revived` margins and the `margin_failed` margins never
overlap — the gap is sometimes as narrow as `0.0299` against `0.0301`. That is
exactly what a deterministic single threshold produces, and it is good evidence
the association layer is behaving deterministically.

Which makes the three failures sharper, not softer. For `dancetrack0003`,
`0056` and `0070`, the interval the decisions require and the interval the
printed `delta` allows are **disjoint** (see the feasibility table). These
files cannot be explained by any single threshold consistent with what they
print. For `dancetrack0003` a `delta` of `0.04` would explain all its
decisions, yet its `margin_failed` records print `0.05`; the file is internally
inconsistent regardless of which value is "true". This is a per-sequence
conclusion that does not depend on my per-record tolerance at all, which is why
I trust these 5 failures more than the single `margin_failed` one.

### 5.3 376 records imply a negative similarity

If Section 4.7's `sim = 1 - 2 * fused_cost` is taken at face value for the 2646
records where appearance was selected, 376 of them yield a negative
similarity — impossible under Section 5's invariant 2. Something in Section 4.7
is wrong, and it is not a rounding artefact: `fused_cost` reaches `0.7999`,
which would imply a similarity of `-0.60`.

### 5.4 Everything Section 5 promised actually holds

All seven invariants pass on all 12016 records with zero violations: the
`iou_cost`/`iou_overlap` complement, every range bound, `cosine_sim` equal to
the top of `all_sims_top5`, `cosine_margin` equal to top-minus-runner-up,
descending sort order, non-decreasing frames, and `theta` inside `[0.65, 0.80]`
(155 distinct values, none out of range). After the problems above I expected
at least one to fray. The internal consistency of the *columns* is excellent;
the problems are all in the relationship between the columns and the prose.

### 5.5 Smaller oddities

- **Nothing was untestable for seven of the eight reasons.** Only `revived`
  produced untestable records (13), and only because of the missing `delta`.
  Operand coverage per reason is otherwise complete and perfectly regular —
  every record of a given reason populates exactly the same column set. That
  regularity is itself notable: `feat_mean_sim` is populated only for
  `revived`, and `n_enrolments` only for four of the eight reasons, with no
  exceptions anywhere in 12016 records.
- **`gap=0` in all 444 `ocr` records.** An operand that never varies.
- **`n_registry` peaks at 68 while `n_enrolments` reaches 1564**, and
  `registry_empty_or_immature` appears only 3 times in 12016 records despite
  Section 4.4's `n_min = 5` maturity rule. Either maturity is reached very
  quickly or that path is logged only in rare circumstances; the event-driven
  file (Section 1) makes this impossible to distinguish.
- **Two `all_sims_top5` entries have only four pairs**, which Section 2 permits
  ("up to five"), so invariant 4's "where at least two are present" guard is
  never actually exercised.

---

## Reproducing this

```
python3 verify_independent.py --report REPORT.md --max-items 25
```

Exit status is 1 when any record fails. `--max-items 0` is not special-cased;
raise it to itemize all 814 advisory records. The verifier reads only
`SPEC.md`'s companion logs under `logs/` and imports nothing outside the
standard library.

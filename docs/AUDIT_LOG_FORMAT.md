# DAAT Audit Log Format Specification

DAAT emits three structured evidence files per sequence–detector pair,
written as buffered CSV appends outside the association loop. Together they
allow an analyst to reconstruct, for any identity in the tracker output,
the complete basis of every association decision. The samples in
`sample_logs/` are the actual logs of sequence MOT17-12-DPM from the
official MOT17 test submission (Submission A).

All coordinates are pixel-space `[x1, y1, x2, y2]` (top-left / bottom-right).
Cosine similarities are computed between L2-normalized appearance
embeddings; higher is more similar. Empty cells mean the quantity was not
applicable to that event type (e.g., appearance fields on a motion-only
event).

---

## 1. `<SEQ>_ids_events.csv` — identity events (20 columns)

One row per identity-relevant event: new-track gating, revivals from the
lost buffer, registry re-associations, and refusals.

| Column | Type | Meaning |
|---|---|---|
| `frame` | int | 1-based frame index of the event. |
| `stage` | str | Pipeline stage that produced the event: `3` = lost-track re-activation during association; `5` = unconfirmed-track update; `6_new` = new-identity gate; `6_revive` = registry revival gate. |
| `old_id` | int | Identity hypothesis under test (the registry/lost track considered). `-1` if none (registry empty). |
| `new_id` | int | Identity actually assigned. Equal to `old_id` when the hypothesis was accepted; a fresh id when a new track was opened. |
| `reason` | str | Decision outcome. Values observed: `re_activate_from_lost`, `revived`, `unconfirmed_update`, `registry_empty_or_immature`, `gate_failed sim=<s><theta=<t>` (best similarity below the acceptance gate), `margin_failed margin=<m><<delta>` (winner–runner-up margin below the disambiguation threshold). Failure rows document *refused* identity claims. |
| `det_score` | float | Detector confidence of the candidate box. |
| `det_box` | list[4] | Candidate detection box. |
| `track_box` | list[4] | Predicted box of the hypothesis track (empty for `6_new` with no hypothesis). |
| `iou_cost` | float | 1 − IoU association cost between `det_box` and `track_box`. |
| `iou_overlap` | float | Raw IoU between `det_box` and `track_box`. |
| `fused_cost` | float | Blended motion+appearance assignment cost, when the fused stage handled the event. |
| `cosine_sim` | float | Best gallery cosine similarity of the candidate against the hypothesis identity. |
| `cosine_theta` | float | Acceptance gate in effect for this event (adaptive; see paper §3.3). |
| `cosine_margin` | float | `cosine_sim` minus the runner-up identity's similarity. The disambiguation test compares this against the margin threshold δ. |
| `feat_norm` | float | Norm of the candidate embedding before normalization (sanity/quality signal). |
| `feat_mean_sim` | float | Mean similarity of the candidate against the hypothesis identity's full gallery (vs. `cosine_sim`, the max). |
| `n_registry` | int | Registry size (number of enrolled identities) at event time. |
| `n_lost` | int | Lost-track buffer size at event time. |
| `n_enrolments` | int | Cumulative gallery enrolments for the hypothesis identity. |
| `all_sims_top5` | list[(id, sim)] | Top-5 registry identities by similarity, as `(track_id, cosine)` pairs — the full evidence behind gate and margin decisions. |

## 2. `<SEQ>_dlow_discards.csv` — low-confidence discard log (10 columns)

One row per stage-two (low-confidence) detection that was **not** used,
recording whether it could have rescued a lost track. This log is what
enables the gated-stage-two diagnosis in the paper (§4.5) without ground
truth.

| Column | Type | Meaning |
|---|---|---|
| `frame` | int | Frame index. |
| `det_score` | float | Detector confidence (below the stage-one threshold by construction). |
| `det_box` | list[4] | Discarded detection box. |
| `best_lost_track_id` | int | Lost track with the highest IoU against this detection (empty if none overlaps). |
| `best_lost_track_box` | list[4] | That lost track's predicted box. |
| `iou_overlap_with_lost` | float | IoU between the discarded detection and the best lost track. |
| `could_rescue_lost_track` | bool | Whether the overlap exceeded the rescue-eligibility threshold. |
| `rescued` | bool | Whether the rescue was actually performed. `could_rescue=True, rescued=False` rows quantify the recall cost of gating. |
| `n_dlow_this_frame` | int | Number of low-confidence detections in this frame. |
| `n_newly_lost_this_frame` | int | Number of tracks newly lost in this frame. |

## 3. `<SEQ>_registry.csv` — identity registry snapshot (6 columns)

One row per enrolled identity at end of sequence: the appearance state
underlying every gate/margin decision above.

| Column | Type | Meaning |
|---|---|---|
| `track_id` | int | Identity. |
| `n_clean_frames` | int | Number of confident, unoccluded observations enrolled into the gallery. |
| `template_mean_b64` | base64 | Mean gallery embedding, base64-encoded float32 vector (decode: `np.frombuffer(base64.b64decode(s), dtype=np.float32)`). |
| `s_min`, `s_max` | float | Min/max intra-gallery self-similarity — a spread measure of the identity's appearance variability. |
| `last_seen_frame` | int | Last frame the identity was observed. |

---

## Provenance guarantee

Every row is written at decision time from the same variables the
assignment code used; no field is recomputed afterwards. The association
affinity and assignment steps are deterministic given these inputs, so any
logged decision can be independently re-derived (paper §3.5).

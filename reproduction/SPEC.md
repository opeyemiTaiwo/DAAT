# Audit log specification, DAAT association layer

Version 1.1, frozen 7 November 2026. Version 1.1 incorporates eleven
clarifications identified by an independent reimplementation written from
version 1.0; the substantive ones concern `track_box` in Section 2, the
second-stage scope in Section 4.6, and the recoverability of `delta` in
Section 6.

This document specifies the evidence log emitted by the association layer and
the decision rules that produced each record, in sufficient detail to write an
independent verifier. It is derived from the frozen configuration and the log
schema. It does not describe any particular verifier implementation.

A verifier's job is to answer, for each logged record: **does the recorded
outcome follow from the operands recorded alongside it, under the rule the
record names?** That is a check of log fidelity and determinism. It is not a
check of tracking correctness, and a log that is internally consistent but
describes a different video would pass.

---

## 1. File layout

One CSV per sequence, named `<sequence>_ids_events.csv`. The file is
**event-driven**: a frame in which no identity-relevant decision occurred
writes no row. Absence of a row is therefore not evidence of absence of
activity, and per-frame track assignments cannot be reconstructed from this
file.

Some runs additionally serialize per-event cost matrices under
`iou_matrices/frame_<n>_iou.csv`. Coverage is partial and these are not
required to verify a record.

## 2. Column schema

Twenty columns, in this order:

| Column | Meaning |
|---|---|
| `frame` | 1-based frame index |
| `stage` | pipeline stage that produced the record (Section 3) |
| `old_id` | track identity before the decision |
| `new_id` | track identity after the decision |
| `reason` | rule name, optionally followed by the operands that decided it |
| `det_score` | detector confidence of the detection involved |
| `det_box` | detection box, `[x1, y1, x2, y2]` |
| `track_box` | the track's box, same format, where one exists. **Not the predicted box the association used.** On the four IoU-driven reasons it is serialized after the track has absorbed the detection, so IoU recomputed from `det_box` and `track_box` does not reproduce `iou_overlap`: it matches on 36 of 10,056 such records and reads higher almost always. On the three appearance-driven reasons, where no absorption occurs, it matches on 83 of 84. Do not recompute the cost from these boxes. |
| `iou_cost` | 1 minus the IoU overlap between `det_box` and `track_box` |
| `iou_overlap` | the IoU overlap itself |
| `fused_cost` | composite association cost (Section 4.7) |
| `cosine_sim` | cosine similarity to the best-matching template |
| `cosine_theta` | the appearance gate in force for this track at this frame |
| `cosine_margin` | best similarity minus runner-up similarity |
| `feat_norm` | L2 norm of the query embedding |
| `feat_mean_sim` | mean similarity across the registry |
| `n_registry` | number of enrolled identities |
| `n_lost` | number of tracks in the lost pool |
| `n_enrolments` | cumulative enrolments so far |
| `all_sims_top5` | up to five `(identity, similarity)` pairs, highest first |

Empty means not applicable to that decision, not zero. Numeric fields are
printed at **four decimal places**; operands embedded in the `reason` string
are printed at **three**. Both roundings are lossy and a verifier should
compare with a tolerance rather than for equality.

## 3. Stages and reasons

Each `stage` pairs with exactly one `reason` keyword.

| `stage` | `reason` keyword | Outcome |
|---|---|---|
| `3` | `re_activate_from_lost` | accepted |
| `4_dlow_second_assoc` | `byte_second_assoc` | accepted |
| `5` | `unconfirmed_update` | accepted |
| `5b_ocr_recovery` | `ocr_last_obs_iou` | accepted |
| `6_revive` | `revived` | accepted |
| `6_new` | `gate_failed` | refused, new identity opened |
| `6_new` | `margin_failed` | refused, new identity opened |
| `6_new` | `registry_empty_or_immature` | no comparison performed |

The `reason` field for the two refusals and for observation-centric recovery
carries its operands inline:

```
gate_failed sim=<A><theta=<B>
margin_failed margin=<A><<B>
ocr_last_obs_iou=<A> gap=<B>
byte_second_assoc iou=<A> score=<B>
```

Note the third form: the keyword and its operand are joined by `=`, so a
pattern matching `iou=` will also match inside the keyword itself. `gap` is
`0` on every record of the frozen submission and its meaning is not specified
here. `re_activate_from_lost`, `unconfirmed_update`, `revived` and
`registry_empty_or_immature` carry no inline operands.

## 4. The frozen configuration

These are the values and rules in force. Thresholds marked *adaptive* are
outputs of a rule and vary per track and per frame; a verifier must read the
value actually in force from the record rather than assuming a constant.

**4.1 Scene difficulty.** A scalar `t` measured once at frame 30 from mean
pairwise template similarity on unlabeled footage:

```
t = clip((s_pair - 0.45) / 0.15, 0, 1)
```

`t` is not logged per record. It is the input to the three adaptive rules
below, so their outputs are logged instead.

**4.2 Appearance gate (adaptive).** Per track, with `s_min` the track's own
minimum observed self-similarity:

```
theta = min(max(s_min, 0.65), 0.80)
```

Logged as `cosine_theta`. In force: values across `[0.65, 0.80]`.

**4.3 Disambiguation margin (adaptive).** During the warm-up window `delta =
0.05`; the window ends at frame 30 inclusive, so frame 30 itself already uses
the adaptive value. Thereafter:

```
delta = (1 - t) * 0.07 + t * 0.03
```

Not logged as its own column; the value in force appears as the right-hand
side of the `margin_failed` reason string. In force: `{0.03, 0.04, 0.05,
0.07}`.

**4.4 Template maturity.** A track may not be queried as a match candidate
until it has accumulated `n_min = 5` clean frames. A query made when no
template is mature yields `registry_empty_or_immature`.

**4.5 Templates.** One running L2-normalized mean embedding per track, 512-d,
OSNet x1_0. There is no fixed-size gallery and no blend weight.

**4.6 Assignment thresholds.** First-stage association accepts a cost at or
below `0.80`. The low-score second association stage accepts a cost strictly
below `0.50`.

The detections it operates on are those below the track threshold `tau*`, which
is the density rule's output and takes `{0.55, 0.57, 0.58}` by tertile, not the
`0.50` detection threshold. On the frozen DanceTrack submission `det_score` on
these records runs to `0.5697`, with 814 at or above `0.50` and 88 at or above
`0.55`. A verifier should test the cost, which Section 6 names as the
assertion, and should not treat `det_score >= 0.50` as a violation.

**4.7 Cost composition.** The first-stage cost is built in three steps, in
order:

```
c1 = 1 - (1 - IoU) * det_score          # detection-score fusion
c2 = min(c1, c_emb)                      # elementwise minimum against appearance
c3 = c2 + 0.2 * a_OCM                    # observation-centric motion term
```

`fused_cost` records `c3`. `c_emb` and `a_OCM` are **not** logged separately.
Consequently: where the minimum selected `c_emb` and no motion term was added,
the appearance cost is recoverable exactly as `fused_cost`, and the similarity
as `1 - 2 * fused_cost`. Where a motion term was added, the record does not
say what went into the composite. A verifier should report these cases
distinctly rather than passing or failing them silently.

**4.8 Appearance suppression.** Appearance is disabled where mutual IoU
exceeds `0.5`.

**4.9 Motion.** Observation-centric weight `0.2` over a 3-frame horizon.
Occlusion motion gate at 1.5 box diagonals, widening `+0.10` per lost frame.

## 5. Invariants a verifier may test

These hold by construction and are independent of any decision rule.

1. `iou_cost + iou_overlap = 1` wherever both are present.
2. `0 <= iou_cost <= 1`, `0 <= fused_cost <= 1`, `0 <= cosine_sim <= 1`.
3. `cosine_sim` equals the highest similarity in `all_sims_top5`.
4. `cosine_margin` equals the highest minus the second-highest similarity in
   `all_sims_top5`, where at least two are present.
5. `all_sims_top5` is sorted in descending similarity.
6. `frame` is non-decreasing within a file.
7. `cosine_theta` lies in `[0.65, 0.80]`.

## 6. What each record asserts, and how to check it

For each reason, the rule the record claims to have applied. A verifier should
state, per record, whether the assertion holds, fails, or cannot be tested
because an operand is absent. The third outcome is a property of the log and
should be reported, not counted as a pass.

**`gate_failed`** asserts the best available similarity fell below the gate in
force, so no re-association was made and a new identity was opened.
Operands: `cosine_sim`, `cosine_theta`, `all_sims_top5`, and the `sim=` and
`theta=` values in the reason string. The assertion is that the best
similarity is strictly below the gate.

**`margin_failed`** asserts the best and runner-up similarities were too close
to distinguish, so no re-association was made. Operands: `cosine_margin`,
`all_sims_top5`, and the `margin=` and right-hand values in the reason string.
The assertion is that the margin is strictly below the margin in force.

**`revived`** asserts a lost track was re-associated on appearance evidence:
the similarity cleared the gate and the margin was decisive. Operands:
`cosine_sim`, `cosine_theta`, `cosine_margin`, `all_sims_top5`. ``Decisive''
means at or above the `delta` in force, which this record does not carry: it
appears only in the reason string of a `margin_failed` record, so in a file
containing no refusal the threshold is not recoverable and the assertion is
untestable rather than satisfied. A verifier must not treat such a record as
passing. Where both appear, note that the accepted and refused margins in some
sequences bracket a value no printed `delta` takes, which Section 4.3's
rounding is not always wide enough to explain.

**`ocr_last_obs_iou`** asserts a track was recovered by overlap against its own
last observation rather than its predicted position. Operands: the value joined
to the keyword in the reason string, `iou_cost`, `iou_overlap`. The acceptance
floor is `0.30`; on the frozen submission the observed values stop at `0.302`,
which is consistent with it but does not by itself establish it.

**`byte_second_assoc`** asserts a low-confidence detection was matched to an
existing track in the second association stage. Operands: `iou_cost`,
`iou_overlap`, `det_score`. The assertion is that the cost was acceptable at
that stage (Section 4.6).

**`unconfirmed_update`** asserts an unconfirmed or occluded track was updated
from a detection on overlap. Operands: `iou_cost`, `iou_overlap`, `det_score`.
The assertion is that the cost was at or below the first-stage threshold of
Section 4.6.

**`re_activate_from_lost`** asserts a track was recovered from the lost pool at
a cost the assignment stage accepted. Operand: `fused_cost`, plus `iou_cost`
and `det_score`, which permit the decomposition in Section 4.7.

**`registry_empty_or_immature`** asserts no comparison was performed because no
template was mature. There is no operand to check. The testable property is
that the fields carrying a comparison are absent, specifically `cosine_sim`,
`cosine_theta`, `cosine_margin` and `all_sims_top5`; `feat_norm` is populated
on these records and its presence is not a violation. Section 4.4's `n_min` is
not verifiable from the schema, since the log does not carry a per-track clean
frame count.

## 7. Reporting

A verifier should report, per reason and in total:

- records examined
- assertions that hold
- assertions that fail, each with file, frame and the operands involved
- records that cannot be tested, with the operand that is missing

A record whose stated reason disagrees with its own logged operands is the
failure mode this log exists to make visible, and should be reported
individually rather than aggregated away.

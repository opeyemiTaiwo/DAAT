# Credits and licenses

This bundle contains third-party material. Each component keeps its own
license, listed below. Nothing here relicenses anything.

---

## DanceTrack validation ground truth

`data/dancetrack_gt/` holds the ground-truth annotations for 15 sequences of
the DanceTrack validation split, unmodified.

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Cite as:**

> P. Sun, J. Cao, Y. Jiang, Z. Yuan, S. Bai, K. Kitani, P. Luo.
> "DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion."
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.

**Source:** https://github.com/DanceTrack/DanceTrack

No changes were made to the annotation files. They are reorganized into the
dataset's own `<sequence>/gt/gt.txt` layout and nothing else.

---

## MOT17

**Not included in this bundle.**

MOT17 is licensed CC BY-NC-SA 3.0. The ShareAlike term requires adaptations to
carry the same license, which would propagate to material bundled with it, and
the NonCommercial term is awkward against a journal supplement. Rather than
resolve that, the scripts that need MOT17 take a `--gt` path and you supply
your own copy.

Measurements computed from MOT17 and reported in the paper are not
redistribution and are unaffected.

**Cite as:**

> A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler.
> "MOT16: A Benchmark for Multi-Object Tracking." arXiv:1603.00831, 2016.

**Source:** https://motchallenge.net/

---

## ByteTrack and OC-SORT

The instrumentation experiment in Section 4.5 attaches the audit layer to
these two trackers from outside their source, by wrapping their association
functions at the module boundary at run time. Neither tracker's source is
modified, and neither is redistributed here; `scripts/bytetrack_audit_port.py`
expects you to supply your own checkout.

**ByteTrack** is MIT licensed.

> Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang.
> "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
> *European Conference on Computer Vision (ECCV)*, 2022.
> https://github.com/ifzhang/ByteTrack

**OC-SORT**

> J. Cao, J. Pang, X. Weng, R. Khirodkar, K. Kitani.
> "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking."
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.
> https://github.com/noahcao/OC_SORT

---

## TrackEval

Used for HOTA and the accuracy comparisons. Not bundled; install it if you
want those rows.

> J. Luiten, A. Hoffhues. "TrackEval." https://github.com/JonathonLuiten/TrackEval

---

## Everything else

The event logs in `data/logs/`, the tracker output in
`data/dancetrack_tracks/`, the scripts in `scripts/`, `SPEC.md` and
`reproduce.py` are the authors' own work and are released with the paper.

`verify_independent.py` was written by an implementer with no access to
`verify.py`, from `SPEC.md` and the logs alone, as described in Section 4.6.

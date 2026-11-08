# Independent verification: what we are asking, and what it involves

**Paper:** Auditable Multi-Person Tracking for Forensic Video
**Journal:** MDPI *Information*, manuscript `information-4636481`

---

## Why you

A reviewer asked that our verification code be "independently implemented, or
at least run by someone outside the author team." You are not an author on
this paper, which is the whole qualification required. No familiarity with
multi-object tracking is needed, and you do not have to read the paper.

## What you are attesting to

That our released code, run on our released data, produces the numbers printed
in our paper.

That is deliberately narrow, and it is worth being clear about what it is not.
You are not attesting that our tracker is accurate, that our logs are
complete, that our conclusions follow, or that our method is sound. Only that
the arithmetic is as we reported it, witnessed by someone with no stake in the
answer.

If the numbers do not match, say so. A disagreement is a more useful result
than a confirmation, and we would rather hear it from you now than from a
reader later.

## Time

About fifteen minutes. The install is the slowest part; the run itself
takes a few minutes on an ordinary laptop.

---

## What to do

**1. Get the bundle.** `daat-reproduction-package.zip`, about 90 MB. Unzip it
anywhere.

**2. Check the environment.** Any machine with Python 3.9 or newer. No GPU, no
network access at run time, no dataset downloads.

```bash
cd daat-repro
pip install -r requirements.txt
python3 reproduce.py --smoke
```

This takes about 30 seconds and confirms the data is present and the code
runs. It should end with `Environment looks good.` If it does not, stop and
tell us what it printed; that is our problem to fix, not yours.

**3. Run it.**

```bash
python3 reproduce.py
```

A few minutes. It prints each claim as it goes, then a table of every number
with what the paper says and what it computed.

**4. Send us the last line.** It will read something like:

```
26 of 26 claims reproduced
```

If anything disagreed, the line says so and the table above it shows which.
Send the whole table in that case.

---

## What we need in writing

A few sentences is plenty. We will quote it in our response to the reviewers
and name you in the manuscript's acknowledgements if you are willing.

> I am [name], [position] at [institution]. I am not an author of this paper.
> On [date] I downloaded the reproduction package, ran `reproduce.py` on
> [machine / OS], and it reported [the verdict line]. [If anything differed:
> the following claims did not reproduce: ...]

Tell us if you would rather not be named. We can report that the verification
was performed by someone outside the author team without identifying you, and
that is still worth more than our running it ourselves.

---

## Things that might go wrong

**`pip install` fails.** Try `pip install -r requirements-lock.txt`, which
pins the exact versions we tested against.

**The smoke test reports missing files.** The bundle did not unzip completely.
Check that `data/` contains four subdirectories.

**A claim is marked `skip`.** That is expected and not a failure. Five groups of claims need MOT17, which we cannot redistribute for licensing
reasons, and the run lists them with the reason. The verdict line counts them separately.

**The run seems stuck.** The flag calibration is the slowest stage and prints
nothing while it works. Give it ten minutes before worrying.

---

## Questions

Contact the corresponding author, whose address is on the manuscript.

Thank you. This is a small favor that answers a reviewer comment we cannot
answer ourselves.

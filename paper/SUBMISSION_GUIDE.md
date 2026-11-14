# Submitting: arXiv now, TMLR in parallel

Two submissions, two different artefacts. Do arXiv first — it takes ten minutes and gives you
a citable link for the PhD applications. TMLR has no deadline, so it can follow the same week.

Posting to arXiv does **not** block TMLR. TMLR's dual-submission rule bars reuse of material
from work published, accepted, or submitted in parallel at another *archival, peer-reviewed*
venue. arXiv is neither archival-peer-reviewed nor a venue, and preprinting is standard
practice for TMLR submissions.

---

## 1. arXiv

**Upload:** `arxiv-submission.tar.gz` at <https://arxiv.org/submit>
**Do not upload the PDF** — arXiv compiles the source itself. The tarball was test-compiled
in a clean directory from its own contents: 9 pages, zero undefined references.

All the metadata fields are in `ARXIV_SUBMISSION.md`: title, plain-text abstract, `cs.LG`
primary, `stat.ML` cross-list, comments, licence.

**If you are asked for an endorsement**, that is arXiv's spam control, not a quality
judgement. Any recent `cs.LG` submitter can give it; they are attesting you work in the area,
not reviewing the paper. Nora Ouzir and Francesca Bugiotti are the natural asks — both
already know your work.

---

## 2. TMLR

**Upload:** `tmlr-submission-anonymous.pdf` at <https://openreview.net/group?id=TMLR>
(OpenReview takes the PDF, not the source.)

`tmlr-source.tar.gz` holds the LaTeX for later revisions and the camera-ready. One source,
three modes — switch the package option at the top of `main.tex`:

| option | use |
|---|---|
| `\usepackage{tmlr}` | anonymous submission *(current)* |
| `\usepackage[preprint]{tmlr}` | de-anonymised preprint |
| `\usepackage[accepted]{tmlr}` | camera-ready after acceptance |

### The anonymous mirror is live

<https://anonymous.4open.science/r/uncertainty-decomposition-review> — already created and
linked from the paper's Reproducibility section. It mirrors the `anon-review` branch, which
exists for exactly this purpose:

- **`paper/` is not on that branch.** It holds `residual-law-preprint.pdf`, which carries
  your name, email and the GitHub URL on page one — and anonymisation mirrors serve PDFs
  unchanged, because redaction only touches text files. Removing the directory is more robust
  than relying on a toggle. The PDF toggle is also off, as a second line of defence.
- **README, study README and LICENSE are de-identified on that branch**, so even the raw
  files carry no author.
- **The mirror is named `uncertainty-decomposition-review`, not `inr-uncertainty`** — the
  auto-generated id matched your public repository name, which a reviewer could have searched.

Verified after creation: everything under `paper/` returns 404, the code, grids and
pre-registrations return 200, and eight served files grep clean for every identity string.

**Keep `anon-review` in sync** if you change the code during review: `git checkout
anon-review && git merge main` then re-remove `paper/`. The mirror has auto-update switched
off, so it stays as the snapshot taken on creation unless you refresh it from the dashboard.

Do **not** link the public GitHub repo in the submission. Non-anonymous submissions are
rejected without review.

### Anonymity, verified

- No occurrence of your name, email, or the repo URL anywhere after `\begin{document}`
- The `\author` block is filled in but suppressed by `tmlr.sty` until you pass `[accepted]`
  or `[preprint]` — so nothing is lost and nothing leaks
- PDF metadata carries no Author or Title field (checked with `pdfinfo`)
- The rendered PDF greps clean for all identity strings

### What review will be like

Rolling submission, no deadline. Double-blind, open review on OpenReview: once an Action
Editor assigns reviewers, the paper becomes publicly visible. At least three reviewers.
Outcomes are Accept, Accept with minor revision, or Reject.

Acceptance turns on two questions only:

1. Are the claims supported by accurate, convincing and clear evidence?
2. Would at least some individuals in TMLR's audience be interested in the findings?

Novelty is explicitly **not** a criterion, and papers are to be accepted even if the
contribution is modest. This is why TMLR suits this paper: the part that would be attacked at
a conference — that Valdenegro-Toro and Seitzer got to "the decomposition misbehaves" first —
is not what TMLR scores.

---

## 3. Be ready to defend it

TMLR's submission volume has tripled in the past year, with single-author submissions up
thirteenfold. An editor interviewed the authors of ten desk-rejected papers and found three
single-author ones whose authors could not answer basic questions about their own problem
setup, notation, or where the abstract's claims were supported in the body. The stated
concern is that a paper its author cannot defend carries little signal about contribution.

Your submission is single-author and lands in that climate. Expect to be asked things like:

- **"Where is the 679× supported?"** — Table 1, column 2. It is the ratio of the largest to
  smallest `A` across the 97-setting grid.
- **"Why exponent 2 rather than a free parameter?"** — 2 is the prediction, not the fit.
  Minimising `½(log v + r²/v)` in `v` gives `v* = r²`. The fitted values (1.86–1.92) are
  the measurement; the agreement is the result.
- **"Isn't this just Seitzer et al.?"** — No. Seitzer is about the optimisation pathology:
  the mean fit is damaged en route because its gradient is scaled by the predicted variance.
  This paper is about the converged estimate: even when optimisation proceeds normally, what
  the head reports is the training residual. §2 says this explicitly.
- **"Why should I believe the causal claim?"** — I2. Same data, same model, same noise; only
  the number of training steps changes. `A` moves 12.6× and the true `σ²` is fixed at
  1.831e-4 by construction.
- **"Your falsifier failed."** — Yes, and §6 says so. H5 was the wrong test to write down,
  because I1 was never a clean probe of the aleatoric head. The I2-only split is reported as
  post-hoc.

Work through `analyse.py` and `run_interventions.py` line by line before submitting. Every
number in the paper is regenerated by those two scripts plus `make_figures.py`, so you can
check any claim against its source in a few minutes.

---

## Sources

- TMLR acceptance criteria — <https://www.jmlr.org/tmlr/acceptance-criteria.html>
- TMLR editorial policies and dual-submission rule — <https://jmlr.org/tmlr/editorial-policies.html>
- TMLR reviewer guide — <https://jmlr.csail.mit.edu/tmlr/reviewer-guide.html>
- TMLR style files — <https://github.com/JmlrOrg/tmlr-style-file>
- arXiv submission — <https://arxiv.org/submit>

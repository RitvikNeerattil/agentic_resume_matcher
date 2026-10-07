# Data

This folder holds the public dataset and the resume pools built from it. Everything here except this file is git-ignored, so each person downloads and builds their own copy.

```sh
pip install datasets
python3 -m shared.rjdfit_eval download   # data/rjdfit/rjdfit_train.csv and rjdfit_test.csv
python3 -m shared.rjdfit_eval build      # data/rjdfit_workload/pools.jsonl
```

## `data/rjdfit/`

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) on Hugging Face: 8,000 rows (6,241 train, 1,759 test) with the columns `resume_text`, `job_description_text` and `label` (No Fit, Potential Fit or Good Fit). The rows repeat a few hundred unique resumes and job descriptions, so each resume appears with many jobs. The dataset page has no card, so the license and the way the labels were made are unknown.

## `data/rjdfit_workload/pools.jsonl`

One line per resume, built by `shared.rjdfit_eval build` (seed 585, 30 resumes by default):

| field | notes |
|---|---|
| `resume_id` | `P01` to `P30` |
| `split` | `dev` for P01 and P02, `eval` for the rest |
| `resume` | Resume text, flattened and split into short lines |
| `pool` | The resume's candidate jobs, each with `job_id` (`D001`...), `description` and the dataset's `label` |

Build rules: at least 8 labeled jobs and at least 2 Potential or Good Fit per resume, up to 20 jobs per pool, and rows whose duplicates disagree on the label are dropped. Change them with `--resumes`, `--min-pool`, `--max-pool` and `--min-relevant`. The build prints pool sizes, the relevant share and the median number of resume lines.

## Older workload

The original hand-labeled workload (50 Greenhouse and Lever postings, 10 anonymized student resumes, 400 team labels, and the first benchmark reports) was removed from the repository. It is still in git history, and its results are summarized in the main README.

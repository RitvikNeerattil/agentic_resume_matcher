# Data

## Jobs (`data/jobs/`)

`jobs_50.jsonl` / `jobs_50.csv` hold the fixed 50-posting collection both matchers get: 25 from Greenhouse and 25 from Lever, 24 Software and 26 AI/ML/Data, from 46 employers. `job_id` values run J001 to J050. The 20-job small workload is the rows with `in_small_workload = true` (10 from each source).

| field | notes |
|---|---|
| `job_id` | Stable ID the matchers must return |
| `source`, `board`, `posting_id` | Greenhouse board token or Lever company slug, plus the posting ID |
| `company`, `title`, `location`, `category` | Category comes from Simplify |
| `sponsorship`, `degrees` | From Simplify, where available |
| `posted_or_updated` | Greenhouse `updated_at`, or Lever `createdAt` (epoch ms) |
| `source_url`, `simplify_id` | Provenance |
| `fetched_at` | When the raw pull ran (2026-09-24) |
| `description` | Full posting as plain text, with bullets kept as `- ` lines |

`selection_log.csv` records why each of the 223 candidate postings was selected or dropped.

### How it was built

1. **Index.** Active postings in [SimplifyJobs/New-Grad-Positions](https://github.com/SimplifyJobs/New-Grad-Positions) `listings.json` (commit `929e0b0`) with a Greenhouse or Lever URL, in the Software or AI/ML/Data categories. AI-training gig boards were left out (Innodata, TSMG, Prolific, Welo). At most 2 postings per employer. That gave 223 candidates, saved in `raw/simplify_index.json`.
2. **Fetch.** Each posting's full record came from the public GET endpoints on 2026-09-24 (`boards-api.greenhouse.io/v1/boards/{board}/jobs/{id}` and `api.lever.co/v0/postings/{company}/{id}`). 219 returned OK and 4 returned 404. The HTML was converted to plain text. Raw copy: `raw/job_snapshot_raw_2026-09-24.json`.
3. **Select.** From the repository root, run `python3 -m shared.build_job_snapshot`. It's deterministic (seed 585). The rules are in the script docstring: early-career only, a software or data title, US location, at least 1,000 characters of description, no duplicate titles, max 2 per employer, and 25 per source with the categories alternated.

Rerunning step 3 always gives the same 50. Rerunning step 2 won't, since postings close. The raw file is the frozen snapshot.

## Resumes (`data/resumes/`)

`R01.txt` to `R10.txt` are anonymized plain-text resumes (R01 and R02 for dev, R03 to R10 for eval), selected from 33 student resumes collected in a previous USC course. `resumes.csv` lists each one's split and preferences. The preferences were assigned by the team, since the resume owners didn't state any.

`python3 -m shared.prep_resumes` rebuilds the `.txt` files from the originals. It removes names, emails, phone numbers, links and street addresses, and keeps city/state. The original PDFs and `data/raw/resume_id_map.csv` (which maps IDs to source files) are git-ignored and stay local.

## Labels (`data/labels/`)

See [LABELING_RUBRIC.md](LABELING_RUBRIC.md). All 400 pairs in `data/labels/primary_labels.csv` are labeled; the team (Kevin, Ritvik, and Aidan) labeled them by applying the rubric, before any matcher predictions existed (`labeler` column says so). `reference_labels.csv` is the copy the experiment runner and summary tool read (`source = primary`). The 80-pair independent review in `second_review.csv` is still blank; filling it and running the agreement script would replace `reference_labels.csv` with reviewed labels. The agreement script refuses incomplete labels or reviews. Run commands from the repository root. To create sheets in a fresh checkout only:

```
python3 -m shared.make_label_sheets   # 400 primary pairs + 80 second-review pairs
python3 -m shared.label_agreement    # agreement, kappa, disagreements -> reference_labels.csv
```

## Not used

`rjdfit_train.csv` / `rjdfit_test.csv` ([cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)) have about 8k synthetic-style resume/JD pairs with fit labels. They aren't part of the planned workload. They're kept locally in case we want extra examples for calibrating the rubric, and are git-ignored because of their size.

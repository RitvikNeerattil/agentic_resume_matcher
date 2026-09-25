"""Create blank labeling sheets once resumes are in.

Expects data/resumes/resumes.csv with columns:
  resume_id, split (dev|eval), file (path to cleaned .txt, relative to data/resumes),
  pref_locations, pref_role_level, needs_sponsorship (yes/no), other_constraints

Writes:
  labels/primary_labels.csv   every eval resume x all 50 jobs (8 x 50 = 400 rows)
  labels/second_review.csv    stratified 20% overlap sample (same pairs, blank labels)
Labelers fill `relevant` (1/0) and `reason_code`; see LABELING_RUBRIC.md.
Do this BEFORE looking at any matcher output.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 585
OVERLAP = 0.20
FIELDS = ["pair_id", "resume_id", "job_id", "company", "title", "location", "relevant", "reason_code", "notes", "labeler"]


def main():
    jobs = [json.loads(l) for l in open(ROOT / "data/jobs/jobs_50.jsonl", encoding="utf-8")]
    resumes = list(csv.DictReader(open(ROOT / "data/resumes/resumes.csv", encoding="utf-8")))
    eval_ids = [r["resume_id"] for r in resumes if r["split"].strip().lower() == "eval"]
    rows = []
    for rid in eval_ids:
        for j in jobs:
            rows.append({"pair_id": f"{rid}__{j['job_id']}", "resume_id": rid, "job_id": j["job_id"],
                         "company": j["company"], "title": j["title"], "location": j["location"],
                         "relevant": "", "reason_code": "", "notes": "", "labeler": ""})
    out = ROOT / "labels"
    out.mkdir(exist_ok=True)
    for name, data in [("primary_labels.csv", rows)]:
        if (out / name).exists():
            raise SystemExit(f"{name} already exists; not overwriting labels")
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(data)
    rng = random.Random(SEED)
    sample = []
    for rid in eval_ids:  # stratify by resume so every resume gets second-reviewed
        mine = [r for r in rows if r["resume_id"] == rid]
        sample += rng.sample(mine, max(1, round(len(mine) * OVERLAP)))
    with open(out / "second_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sample)
    print(f"{len(rows)} primary pairs, {len(sample)} second-review pairs for {len(eval_ids)} eval resumes")


if __name__ == "__main__":
    main()

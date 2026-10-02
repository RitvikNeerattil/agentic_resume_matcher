"""Build the fixed ~50-job snapshot from the raw Greenhouse/Lever pull.

Input : data/raw/job_snapshot_raw_2026-09-24.json  (219 postings fetched 2026-09-24,
        employers/roles taken from SimplifyJobs/New-Grad-Positions listings.json)
        data/raw/simplify_index.json                (Simplify metadata for those postings)
Output: data/jobs/jobs_50.jsonl, data/jobs/jobs_50.csv, data/jobs/selection_log.csv

Selection rules (deterministic, seed 585):
  1. Posting fetched OK and description >= 1,000 characters.
  2. Early-career: drop titles with senior/staff/principal/lead/manager/director/intern/phd.
  2b. Software/data title (Simplify categories are occasionally wrong), no labeling,
      annotation, contract-student or school-restricted roles.
  3. US location (or US remote), since evaluation resumes are US new grads.
  4. Deduplicate on (company, normalized title); at most 2 postings per employer.
  5. 25 Greenhouse + 25 Lever, balanced between Software and AI/ML/Data where possible.
  6. The 20-job small workload = 10 Greenhouse + 10 Lever, flagged in `in_small_workload`.
"""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/job_snapshot_raw_2026-09-24.json"
IDX = ROOT / "data/raw/simplify_index.json"
OUT = ROOT / "data/jobs"
SEED = 585
PER_SOURCE = 25
SMALL_PER_SOURCE = 10
MIN_CHARS = 1000

SENIOR = re.compile(r"\b(senior|sr\.?|staff|principal|lead|manager|director|head of|intern(ship)?|co-?op|phd|ph\.d)\b", re.I)
US_STATE = re.compile(
    r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|SF|NYC|USA|US)\b"
)
US_NAME = re.compile(
    r"\b(united states|u\.s\.|california|new york|san francisco|seattle|boston|chicago|austin|"
    r"washington|los angeles|denver|atlanta|texas|virginia|massachusetts|colorado|illinois)\b", re.I
)


class _USHint:
    @staticmethod
    def search(s):
        return US_STATE.search(s) or US_NAME.search(s)


US_HINT = _USHint()
# Simplify's category is sometimes wrong (e.g. a PreK teacher tagged AI/ML/Data), so also check the title.
TECH_TITLE = re.compile(r"software|engineer|developer|programmer|data|machine learning|\bml\b|\bai\b|scientist|full stack|backend|frontend|sql", re.I)
OFF_TARGET = re.compile(r"teacher|annotation|label|contract student|evaluation specialist|intelligence analyst|only\)|pricing|client report|operator|data creator|\(sales\)|swing shift|collector|postdoc", re.I)
NON_US =re.compile(r"\b(canada|toronto|vancouver|london|uk|united kingdom|india|bangalore|singapore|germany|berlin|dublin|ireland|amsterdam|sydney|paris|tel aviv|mexico|brazil|china|shanghai|beijing|tokyo)\b", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def loc_ok(job, idx_locs):
    locs = " | ".join([job.get("location") or ""] + (idx_locs or []))
    if NON_US.search(locs) and not US_HINT.search(locs):
        return False
    return bool(US_HINT.search(locs)) or "remote" in locs.lower()


def main():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    idx = {(r["source"], r["board"], str(r["job_id"])): r for r in json.loads(IDX.read_text(encoding="utf-8"))}
    log, eligible = [], []
    for j in raw["jobs"]:
        key = (j["src"], j["b"], str(j["id"]))
        meta = idx.get(key, {})
        reason = None
        if j.get("status") != 200:
            reason = f"fetch_{j.get('status')}"
        elif len(j.get("text") or "") < MIN_CHARS:
            reason = "short_description"
        elif SENIOR.search(j.get("title") or ""):
            reason = "not_early_career"
        elif not TECH_TITLE.search(j.get("title") or "") or OFF_TARGET.search(j.get("title") or ""):
            reason = "not_swe_or_data_role"
        elif not loc_ok(j, meta.get("locations")):
            reason = "non_us"
        j["_meta"] = meta
        log.append({"source": j["src"], "board": j["b"], "posting_id": j["id"], "title": j.get("title"), "reason": reason or "eligible"})
        if not reason:
            eligible.append(j)

    rng = random.Random(SEED)
    chosen = []
    for src in ("greenhouse", "lever"):
        pool = [j for j in eligible if j["src"] == src]
        rng.shuffle(pool)
        # alternate categories so Software and AI/ML/Data are both represented
        by_cat = {}
        for j in pool:
            cat = "AI/ML/Data" if "AI" in (j["_meta"].get("category") or "") or "Data" in (j["_meta"].get("category") or "") else "Software"
            by_cat.setdefault(cat, []).append(j)
        order = []
        while any(by_cat.values()):
            for cat in sorted(by_cat):
                if by_cat[cat]:
                    order.append(by_cat[cat].pop(0))
        per_emp, seen, picked = Counter(), set(), []
        for j in order:
            emp = (j.get("company") or j["b"]).lower()
            tkey = (emp, norm_title(j.get("title")))
            if per_emp[emp] >= 2 or tkey in seen:
                continue
            per_emp[emp] += 1
            seen.add(tkey)
            picked.append(j)
            if len(picked) == PER_SOURCE:
                break
        for i, j in enumerate(picked):
            j["_small"] = i < SMALL_PER_SOURCE
        chosen.extend(picked)

    chosen_keys = {(j["src"], j["b"], str(j["id"])) for j in chosen}
    for row in log:
        if row["reason"] == "eligible":
            row["reason"] = "selected" if (row["source"], row["board"], str(row["posting_id"])) in chosen_keys else "eligible_not_sampled"

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for n, j in enumerate(chosen, 1):
        m = j["_meta"]
        records.append({
            "job_id": f"J{n:03d}",
            "source": j["src"],
            "board": j["b"],
            "posting_id": str(j["id"]),
            "company": m.get("company") or j.get("company"),
            "title": j.get("title"),
            "location": j.get("location"),
            "category": m.get("category"),
            "sponsorship": m.get("sponsorship"),
            "degrees": m.get("degrees"),
            "posted_or_updated": j.get("updated_at") or j.get("first_published") or j.get("created_at"),
            "source_url": j.get("url") or m.get("url"),
            "simplify_id": m.get("simplify_id"),
            "in_small_workload": j["_small"],
            "fetched_at": raw["fetched_at"],
            "description": j["text"],
        })
    with open(OUT / "jobs_50.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "jobs_50.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    with open(OUT / "selection_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print(Counter(r["reason"] for r in log))
    print(Counter(r["source"] for r in records), Counter(r["category"] for r in records),
          "small:", sum(r["in_small_workload"] for r in records))


if __name__ == "__main__":
    main()

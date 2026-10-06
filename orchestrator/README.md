# Kevin's orchestrator

| Component | Purpose |
|---|---|
| [`workflow.py`](workflow.py) | Controller and three workers: `parse_resume`, `extract_requirements`, and `rank_jobs`. |
| [`validation.py`](validation.py) | Resume evidence and complete job-extraction checks. |
| [`prompts/resume.txt`](prompts/resume.txt) | Resume parser instructions. |
| [`prompts/requirements.txt`](prompts/requirements.txt) | Batched requirement extractor instructions. |
| [`__init__.py`](__init__.py) | Exposes `match` and `extract_requirements` for the shared runner. |

The orchestrator imports [`shared/common.py`](../shared/common.py) for the API
client, retries, usage/cost logging, cleanup, and output contract, and uses
[`shared/rank.txt`](../shared/rank.txt) for ranking instructions. These components
are available for Aidan's later baseline integration. The baseline is Aidan's
responsibility and is not implemented here. Shared tools, settings, and tests live
in [`shared/`](../shared/); inputs and labels live in [`data/`](../data/).

Ritvik can call `orchestrator.match(client, resume, preferences, jobs)` from the
team's evaluation harness. It returns `{"matches": [{"job_id": "J001",
"explanation": "..."}, ...]}` with exactly five distinct valid job IDs.

Completed:

- Reviewed the shared job/resume interface and preserved the existing frozen data.
- Built resume parsing, batched job extraction, and ranking workers.
- Versioned worker prompts and validated evidence, coverage, and top-five IDs.
- Preserved the shared input/output contract for the team's comparison.
- Added usage, cache-aware cost, latency, raw response, and retry logs.
- Added the orchestrator's 48-run experiment and six-run reuse ablation.
- Added comparison tables, paired resume differences, and the project decision rule.
- Created 400 primary and 80 independent-review label rows.
- Prevented unfinished labels from becoming reference labels.
- Verified the pipeline offline with eight contract tests.

Still requires the team and API access:

- ~~Add ten anonymized resumes~~ Done: R01–R10 are in `data/resumes/` (preferences assigned by the team).
- Calibrate labels on two development resumes; finish human evaluation labels.
- Select a fixed model version, record dated provider prices, and run the pilot.
- Freeze the successful pilot, run evaluation and reuse, and produce real tables.

No API experiments, human relevance judgments, or quality claims have been fabricated.
The blank sheets are preparation, not completed labels. Offline fake responses only
verify control flow. Weeks 5–6 cannot be empirically completed without these inputs.

## Run

Run all commands from the repository root. Python 3.10 or newer is sufficient;
there are no third-party package dependencies.

```sh
python3 -m unittest shared.test_workflow -v
python3 -m shared.run_experiment --phase pilot
```

The default command prints the schedule and missing inputs without API calls.
Add `data/resumes/R01.txt` through `R10.txt` and update `resumes.csv` preferences.
Fill `shared/experiment.json`: a fixed model snapshot supporting Chat Completions,
JSON mode, temperature and the specified output limits; all prices in USD per
million tokens; price date and official source URL. The example config intentionally
has no guessed model or prices. Set `OPENAI_API_KEY` in your shell; do not commit it.
API usage and response formats follow the
[official API reference](https://developers.openai.com/api/reference/resources/chat).

```sh
python3 -m shared.run_experiment --phase pilot --output results/pilot --execute
```

The pilot is two dev resumes × two sizes = four orchestrator runs. Inspect
explanations, context/output limits, token usage, costs and failures. Estimate the
orchestrator main cost from pilot averages: 24 times the sum of the two size mean costs,
plus six reuse runs and one preparation. Confirm it fits the proposed $20 total
budget. If tuning is needed, adjust prompts/config and run a new pilot in a new
directory. Successful pilot prompts/config/jobs must match evaluation exactly.
Main runs recompute both intermediate extractions on every repetition.

Label sheets have already been created. Do not regenerate over human work.
Follow [`data/LABELING_RUBRIC.md`](../data/LABELING_RUBRIC.md) before looking at any eval predictions. Then:

```sh
python3 -m shared.label_agreement
# If disagreements exist, resolve them in data/labels/disagreements.csv and rerun.
python3 -m shared.run_experiment --phase main --pilot-dir results/pilot --output results/main --execute
python3 -m shared.summarize_results results/main
python3 -m shared.run_experiment --phase reuse --pilot-dir results/pilot --output results/reuse --execute
python3 -m shared.summarize_results results/reuse
```

`--execute` is the explicit paid execution switch. Each output directory is new;
existing logs are never overwritten. Runs execute one resume at a time. Job-order
seeds are saved for pairing with Aidan's baseline later. Every call gets at most
one retry. Invalid evidence,
incomplete extraction, truncated output and invalid rankings count as failures.
The HTTP timeout is 120 seconds per attempt. Any unknown billed usage is preserved
and stops further spending until reviewed. The $20 limit is per invocation and
checked between matching runs, so a run can cross it; monitor combined pilot,
main and ablation spending. This is a guard, not a guaranteed provider spending cap.

Each manifest records prompts, config/prices, workload/resume hashes, preferences,
labels hash, schedule, Python and OS. Raw response logs and resume-derived facts
are saved under git-ignored `results/`. Review privacy before sharing any output.
Completion tokens already include billed reasoning tokens; they are not added
twice. Cached input is billed separately using metadata. Missing usage is unknown,
never reported as free. Actual cached usage is logged; provider-cache disabling
is not assumed.

`comparison.csv` reports orchestrator Precision@5, repetition SD, latency median/range, mean
cost, known input/output tokens, failures and condition throughput. Failed runs
score zero. The summary tool can compute paired differences when the team provides
baseline logs. `decision.json` marks schedule completion; its comparison criterion
stays unknown without baseline results. Partial tables are marked incomplete.
Condition throughput uses summed matcher time; `batch.json` also records full batch
wall time. Reuse saves one prepared 50-job extraction, runs R03/R04 three times each,
and writes preparation and amortized costs separately. Ranking retains original
job text to preserve metadata and support checking extracted facts; the pilot will
measure the resulting overhead. Semantic explanation correctness still needs review.

AI assistance materially contributed to implementation, prompts, tests and these
instructions. Existing source records and collection scripts remain the data basis.

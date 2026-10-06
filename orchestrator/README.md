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
are also used by the one-call baseline in [`single_agent/`](../single_agent/),
added for the requested measured comparison. Shared tools, settings, and tests
live in [`shared/`](../shared/); inputs and labels live in [`data/`](../data/).

The requirement worker selects numbered lines from each complete posting. A JSON
schema requires every supplied job ID; local inference also constrains references
to that job’s source IDs. Each category selects up to four decisive lines. The
controller restores original quotes for validation, then ranks with source
references and the complete numbered postings, avoiding duplicate quote text.
Reuse accepts requirements saved by `extract_requirements`. Evidence checks ignore
whitespace differences while preserving the original words. Local requests
disable input truncation and context shifting.

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
- Verified the pipelines and runner offline with 16 contract tests.

Still requires the team:

- ~~Add ten anonymized resumes~~ Done: R01–R10 are in `data/resumes/` (preferences assigned by the team).
- Calibrate labels on two development resumes; finish human evaluation labels.
- Review the completed development measurements and run a successful pilot.
- Freeze the successful pilot, run evaluation and reuse, and produce real tables.

The local development benchmark completed all eight attempts: six succeeded.
Both R02 orchestrator runs failed resume-evidence validation after one retry.
The [measured PDF](../data/reports/pipeline_comparison.pdf) includes successful
end-to-end times, token usage, actual outputs, and failed-attempt accounting.
R01 completed both 20- and 50-job orchestrator runs. A fully successful pilot
still requires resolving the R02 parser failures. Quality claims remain pending. The 400 primary labels are done (by the team);
the 80-pair second review is still blank.
Offline fake responses only verify control flow. Full weeks 5–6 quality evaluation
requires the second label review and a successful pilot.

## Run

Run all commands from the repository root. Python 3.9 or newer is sufficient
for the matchers and runner. PDF/chart generation uses Matplotlib, already
available through `/opt/anaconda3/bin/python` in this environment.

```sh
python3 -m unittest shared.test_workflow -v
python3 -m shared.run_experiment --phase pilot
```

The default command prints the schedule and missing inputs without making model
requests. Resumes R01–R10 and assigned preferences are already present.

Choose the provider using `--config`:

- `shared/local_experiment.json` runs the downloaded
  `qwen3:4b-instruct-2507-q4_K_M` model through Ollama 0.35.0 on
  `http://127.0.0.1:11434`. No API key is required. The model supports a native
  262,144-token context; the config uses 131,072. API token charges are $0;
  hardware/electricity costs are unmeasured.
- `shared/experiment.json` selects OpenAI's fixed
  `gpt-4.1-mini-2025-04-14` snapshot and requires `OPENAI_API_KEY`. Its prices,
  dated October 6, 2026, are $0.40 input, $0.10 cached input, and $1.60 output
  per million tokens. [Official model/pricing documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

Measure development cost and end-to-end speed for both pipelines, then create
the PDF/charts:

```sh
python3 -m shared.run_experiment --phase benchmark --architectures both --repetitions 1 --config shared/local_experiment.json --output results/new_benchmark --execute
/opt/anaconda3/bin/python -m shared.make_pipeline_report results/new_benchmark --output data/reports
```

This makes eight matching runs: two development resumes × two job sizes × two
architectures. Omitting `--repetitions 1` gives three repetitions and 24 runs.
Both receive the same original inputs, shared cleanup, model, ranking prompt,
and final output contract. Each run's time includes retries and validation.
Human labels are not required for this development benchmark, and no
Precision@5 or accuracy conclusion follows from it.

For a paired pilot that can be frozen for quality evaluation:

```sh
python3 -m shared.run_experiment --phase pilot --architectures both --config shared/local_experiment.json --output results/paired_pilot --execute
```

The paired pilot has eight runs. All must succeed before the 96-run paired main
experiment can freeze them. Without `--architectures both`, the default pilot
contains four orchestrator runs and supports the default 48-run orchestrator
main experiment. Inspect explanations, context/output limits, tokens, costs,
and failures. Estimate cloud evaluation spending from the pilot before making
paid requests; the proposed combined budget is $20. If tuning is needed,
adjust prompts/config and run a new pilot in a new directory. Successful pilot
prompts/config/jobs must match evaluation exactly. Main runs recompute both
intermediate extractions on every repetition.

Label sheets have already been created. Do not regenerate over human work.
Follow [`data/LABELING_RUBRIC.md`](../data/LABELING_RUBRIC.md) before looking at any eval predictions. Then:

```sh
python3 -m shared.label_agreement
# If disagreements exist, resolve them in data/labels/disagreements.csv and rerun.
python3 -m shared.run_experiment --phase main --architectures both --config shared/local_experiment.json --pilot-dir results/paired_pilot --output results/main --execute
python3 -m shared.summarize_results results/main
python3 -m shared.run_experiment --phase reuse --config shared/local_experiment.json --pilot-dir results/paired_pilot --output results/reuse --execute
python3 -m shared.summarize_results results/reuse
```

Main/reuse require exactly 400 complete human reference labels. They retain the
pilot checks; the development benchmark bypasses those quality prerequisites.
`--execute` starts inference with the selected provider; cloud execution incurs
API charges. Each output directory is new;
existing logs are never overwritten. Runs execute one resume at a time. Job-order
seeds are paired across architectures, and method order alternates. Every call gets at most
one retry. Invalid evidence,
incomplete extraction, truncated output and invalid rankings count as failures.
The cloud HTTP timeout is 120 seconds per attempt; the local config allows 1,800
seconds per attempt. Any unknown token usage is preserved
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

`comparison.csv` reports each included architecture's Precision@5, repetition SD, latency median/range, mean
cost, known input/output tokens, failures and condition throughput. Failed runs
score zero. The summary tool computes paired differences when both architectures
are included. `decision.json` marks schedule completion; its quality criterion
stays unknown without baseline results and completed labels. Partial tables are
marked incomplete. Use `shared.make_pipeline_report` for development cost/speed
and output comparisons without requiring relevance labels.
Condition throughput uses summed matcher time; `batch.json` also records full batch
wall time. Reuse saves one prepared 50-job extraction, runs R03/R04 three times each,
and writes preparation and amortized costs separately. Ranking retains every original job line once, with source IDs linking extracted
requirements to the text. The pilot measures the resulting overhead. Semantic explanation correctness still needs review.

AI assistance materially contributed to implementation, prompts, tests and these
instructions. Existing source records and collection scripts remain the data basis.

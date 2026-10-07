# Kevin's orchestrator

| Component | Purpose |
|---|---|
| [`workflow.py`](workflow.py) | Controller and three workers: `parse_resume`, `extract_requirements` and `rank_jobs`. |
| [`validation.py`](validation.py) | Resume evidence and complete job-extraction checks. |
| [`prompts/resume.txt`](prompts/resume.txt) | Resume parser instructions. |
| [`prompts/requirements.txt`](prompts/requirements.txt) | Batched requirement extractor instructions. |
| [`__init__.py`](__init__.py) | Exposes `match` and `extract_requirements`. |

The orchestrator uses [`shared/common.py`](../shared/common.py) for the model client, retries, usage and cost logging, cleanup and output contract, and [`shared/rank.txt`](../shared/rank.txt) for ranking instructions. The one-call baseline in [`single_agent/`](../single_agent/) uses the same pieces, so the only difference between the two is the architecture.

## How it works

1. **Resume parser.** The resume is split into numbered lines. The model picks line IDs for skills, education and experience, and the controller restores the exact original text before checking evidence. (Asking the 4B model to retype quotes made it fail validation, so we use IDs.)
2. **Requirement extractor.** One call covers every job. Each posting is split into numbered lines, and the model picks up to four decisive line IDs each for required, preferred and constraint lines. A JSON schema requires every supplied job ID, and local inference also limits IDs to that job's own lines.
3. **Ranker.** Gets the parsed resume, the selected requirement lines and the complete numbered postings, and returns exactly five distinct valid job IDs with explanations.

Local requests turn off input truncation and context shifting. Every call gets at most one retry, and failed attempts keep their time and cost in the logs.

## Use

```python
from orchestrator import match
output = match(client, resume_text, preferences, jobs)
# {"matches": [{"job_id": "D001", "explanation": "..."}, ... five in total]}
```

`jobs` is a list of dicts with a `job_id` and a `description`. `preferences` can be `{}`. `match` also accepts `requirements=` to reuse extraction results saved from `extract_requirements` (a feature of the original design that the rjdfit evaluation does not use).

## Run and test

From the repository root:

```sh
python3 -m unittest shared.test_workflow shared.test_rjdfit
python3 -m shared.rjdfit_eval run --config shared/local_experiment.json --split dev --output results/dev --execute
```

The tests use fake responses to check control flow only. They are not experiment evidence. See the [main README](../README.md) for the full evaluation, configs and the Colab notebook.

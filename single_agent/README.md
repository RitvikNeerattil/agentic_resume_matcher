# Single-agent baseline

This is the one-call comparison pipeline. It sends the cleaned resume, preferences and every candidate job description to the model in one call. The shared ranking prompt asks for five ranked job IDs with grounded explanations, and the shared client validates the response and records tokens, billed cost, latency and any retry.

It uses the same input cleanup, model, output limit, ranking prompt and final validation as the orchestrator. Both supplied configs set `output_limits.baseline` equal to `output_limits.rank`. A successful run normally takes one call. An invalid response or request failure can add one logged retry.

```python
from single_agent import match
output = match(client, resume_text, preferences, jobs)
```

`jobs` is a list of dicts with a `job_id` and a `description`.

To compare it with the orchestrator on the rjdfit dataset, from the repository root:

```sh
python3 -m shared.rjdfit_eval run --config shared/local_experiment.json --architectures both --split dev --output results/dev --execute
```

Use a new output folder each time. For the cloud model, use `--config shared/experiment.json` and set `OPENAI_API_KEY`. See the [main README](../README.md) for the full evaluation and the Colab notebook, and [`orchestrator/README.md`](../orchestrator/README.md) for the other pipeline.

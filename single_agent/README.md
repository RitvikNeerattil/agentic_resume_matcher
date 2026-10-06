# Single-agent baseline

This folder contains the single-agent comparison pipeline. It sends the cleaned
resume, preferences, and complete candidate job descriptions to one LLM call.
The shared ranking prompt returns five ranked job IDs and grounded explanations.
The shared client validates the response and records tokens, billed cost,
latency, and any retry.

The baseline uses the same input cleanup, model, final output limit, ranking
prompt, and final validation as the orchestrator. Both supplied configs set
`output_limits.baseline` equal to `output_limits.rank`. It lives in its own
folder because it was added specifically for the requested measured comparison.

From the repository root, inspect the paired benchmark plan:

```sh
python3 -m shared.run_experiment --phase benchmark --architectures both --repetitions 1 --config shared/local_experiment.json
```

The downloaded Ollama model `qwen3:4b-instruct-2507-q4_K_M` runs locally at
`http://127.0.0.1:11434` without an API key. Run both pipelines on identical
development inputs, then generate the measured PDF/charts:

```sh
python3 -m shared.run_experiment --phase benchmark --architectures both --repetitions 1 --config shared/local_experiment.json --output results/new_benchmark --execute
/opt/anaconda3/bin/python -m shared.make_pipeline_report results/new_benchmark --output data/reports
```

This command measures eight matching runs: two development resumes × two job
sizes × two architectures. Omitting `--repetitions 1` uses three repetitions
and produces 24 runs. Development cost/speed benchmarks need no human labels;
they do not measure relevance accuracy. Local API charges are zero, while
hardware/electricity costs are unmeasured. The report requires Matplotlib.

For optional OpenAI execution, use `--config shared/experiment.json` and set
`OPENAI_API_KEY`; that config supplies the fixed cloud model and dated prices.
Use a new output directory for each benchmark. Actual comparisons come from
saved token counters and wall-clock logs. A successful baseline normally uses
one call; an invalid response or request failure can add one logged retry.

The full 96-run paired main experiment additionally requires completed human
labels and a successful eight-run pilot using `--architectures both`. See
[`orchestrator/README.md`](../orchestrator/README.md) for pilot/evaluation commands.

The completed local benchmark succeeded on all four single-agent runs. See the
[measured PDF](../data/reports/pipeline_comparison.pdf) for timing, tokens, and
actual output comparisons, including the orchestrator’s two recorded failures.

"""Build a PDF and charts from measured pipeline runs, without making API calls.

Usage: python -m shared.make_pipeline_report results/local_benchmark --output data/reports
Requires matplotlib. Missing usage or billing metadata stays unknown.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/resume_matcher_mpl')

METHODS = ('single', 'orchestrated')
NAMES = {'single': 'Single agent', 'orchestrated': 'Multi agent'}
COLORS = {'single': '#4A69BD', 'orchestrated': '#008C82'}
INK, MUTED, PALE = '#142C43', '#53677A', '#F1F5F9'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def key(run):
    return (run['resume_id'], run['size'], run['repetition'], run['architecture'])


def real_response(call, provider):
    """Reject the minimal transport fixtures used in the offline unit tests."""
    raw = call.get('raw_response')
    if not isinstance(raw, dict) or not raw.get('model'):
        return False
    choices = raw.get('choices')
    if not isinstance(choices, list) or not choices:
        return False
    content = choices[0].get('message', {}).get('content')
    if not isinstance(content, str) or not content.strip():
        return False
    if provider == 'ollama':
        original = raw.get('ollama_response', {})
        return (bool(original.get('created_at')) and original.get('done') is True
                and original.get('model') == raw['model']
                and original.get('message', {}).get('content') == content)
    return (isinstance(raw.get('id'), str) and raw['id'].startswith('chatcmpl-')
            and raw.get('object') == 'chat.completion' and number(raw.get('created')))


def usage(call):
    """Use the counters stored in the response, never a tokenizer estimate."""
    raw = call.get('raw_response', {})
    value = raw.get('usage')
    if not isinstance(value, dict):
        return None
    input_tokens, output_tokens = value.get('prompt_tokens'), value.get('completion_tokens')
    if not number(input_tokens) or not number(output_tokens):
        return None
    if input_tokens != int(input_tokens) or output_tokens != int(output_tokens):
        raise ValueError('Response token counts must be integers')
    if call.get('usage') != value:
        raise ValueError('Call usage differs from saved response metadata')
    original = raw.get('ollama_response')
    if original is not None and (original.get('prompt_eval_count') != input_tokens or original.get('eval_count') != output_tokens):
        raise ValueError('Normalized usage differs from original Ollama counters')
    cached = value.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    if not number(cached) or cached > input_tokens or cached != int(cached):
        raise ValueError('Invalid cached input token count')
    return int(input_tokens), int(output_tokens), int(cached)


def load_runs(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    runs = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines() if line.strip()]
    provider = manifest.get('config', {}).get('provider', 'openai')
    if provider not in ('openai', 'ollama'):
        raise ValueError('Only recorded OpenAI or Ollama responses are supported')
    if not runs or {r.get('architecture') for r in runs} != set(METHODS):
        raise ValueError('Supply actual runs for both single and orchestrated pipelines')
    schedule = manifest.get('schedule', [])
    expected = {key(row) for row in schedule}
    observed = [key(run) for run in runs]
    if not expected or len(expected) != len(schedule) or len(set(observed)) != len(observed):
        raise ValueError('Manifest schedule and run identifiers must be nonempty and unique')
    if not set(observed) <= expected:
        raise ValueError('Observed runs differ from the frozen manifest schedule')
    for size in {r['size'] for r in runs}:
        if {r['architecture'] for r in runs if r['size'] == size} != set(METHODS):
            raise ValueError('Each reported workload needs both pipelines')
    responses = defaultdict(int)
    for run in runs:
        if run.get('status') not in ('ok', 'failed') or not number(run.get('seconds')):
            raise ValueError('Each run needs a recorded status and measured duration')
        if not run.get('calls'):
            raise ValueError('A run without API call logs is not a measured run')
        for call in run['calls']:
            authentic = real_response(call, provider)
            if call.get('raw_response') is not None and not authentic:
                raise ValueError('Response is incomplete or an offline transport fixture')
            if call.get('status') == 'ok' and not authentic:
                raise ValueError('Successful calls require the saved provider response')
            if authentic:
                responses[run['architecture']] += 1
            counters = usage(call)
            if counters is not None and number(call.get('cost_usd')):
                prices = call.get('prices_per_million', manifest['config'].get('prices_per_million', {}))
                if not all(number(prices.get(k)) for k in ('input', 'cached_input', 'output')):
                    raise ValueError('Known costs need the frozen input/cache/output prices')
                inp, out, cached = counters
                calculated = ((inp - cached) * prices['input'] + cached * prices['cached_input'] + out * prices['output']) / 1e6
                if not math.isclose(calculated, call['cost_usd'], rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError('Logged call cost differs from its actual token accounting')
        known = all(usage(call) is not None and number(call.get('cost_usd')) for call in run['calls'])
        if known and (not number(run.get('cost_usd')) or not math.isclose(run['cost_usd'], sum(call['cost_usd'] for call in run['calls']), rel_tol=1e-8, abs_tol=1e-10)):
            raise ValueError('Run cost differs from the sum of its actual call costs')
        if run['status'] == 'ok':
            matches = run.get('output', {}).get('matches', [])
            ids = [match.get('job_id') for match in matches]
            if (len(ids) != 5 or len(set(ids)) != 5 or not set(ids) <= set(run.get('job_order', []))
                    or any(not isinstance(m.get('explanation'), str) or not m['explanation'].strip() for m in matches)):
                raise ValueError('Successful run has no validated five-job ranking')
    if not all(responses[method] for method in METHODS):
        raise ValueError('Each pipeline needs at least one nonempty actual provider response')
    pairs = defaultdict(dict)
    for run in runs:
        pairs[(run['resume_id'], run['size'], run['repetition'])][run['architecture']] = run
    for pair in pairs.values():
        if set(pair) == set(METHODS) and pair['single'].get('job_order') != pair['orchestrated'].get('job_order'):
            raise ValueError('Paired pipeline runs must receive identical job order')
    return manifest, runs


def summarize(manifest, runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run['size'], run['architecture'])].append(run)
    rows = []
    for (size, method), group in sorted(groups.items(), key=lambda item: (item[0][0], METHODS.index(item[0][1]))):
        durations = [r['seconds'] for r in group]
        succeeded = [r['seconds'] for r in group if r['status'] == 'ok']
        calls = [call for run in group for call in run['calls']]
        counted = [usage(call) for call in calls]
        unknown_usage = sum(value is None for value in counted)
        known_input = sum(value[0] for value in counted if value is not None)
        known_output = sum(value[1] for value in counted if value is not None)
        known_cached = sum(value[2] for value in counted if value is not None)
        unknown_cost = sum(value is None or not number(call.get('cost_usd')) for call, value in zip(calls, counted))
        known_cost = sum(call['cost_usd'] for call, value in zip(calls, counted) if value is not None and number(call.get('cost_usd')))
        n, total_seconds = len(group), sum(durations)
        rows.append({
            'size': size, 'architecture': method, 'attempted_runs': n,
            'expected_runs': sum(r['size'] == size and r['architecture'] == method for r in manifest['schedule']),
            'successful_runs': len(succeeded), 'failed_runs': n - len(succeeded), 'api_calls': len(calls),
            'retry_calls': sum(call.get('retry', 0) > 0 for call in calls),
            'median_seconds': statistics.median(durations), 'min_seconds': min(durations), 'max_seconds': max(durations),
            'median_success_seconds': statistics.median(succeeded) if succeeded else None,
            'total_matcher_seconds': total_seconds,
            'successful_runs_per_minute': len(succeeded) * 60 / total_seconds if total_seconds else None,
            'total_input_tokens': None if unknown_usage else known_input,
            'total_output_tokens': None if unknown_usage else known_output,
            'mean_input_tokens': None if unknown_usage else known_input / n,
            'mean_output_tokens': None if unknown_usage else known_output / n,
            'known_input_tokens': known_input, 'known_output_tokens': known_output, 'known_cached_tokens': known_cached,
            'unknown_usage_calls': unknown_usage, 'unknown_cost_calls': unknown_cost,
            'known_cost_usd': known_cost, 'total_cost_usd': None if unknown_cost else known_cost,
            'mean_cost_usd': None if unknown_cost else known_cost / n,
        })
    return rows


def fmt(value, digits=1):
    return 'Unknown' if value is None else f'{value:,.{digits}f}'


def token_fmt(value):
    return 'Unknown' if value is None else (f'{value / 1000:.1f}k' if value >= 1000 else f'{value:,.0f}')


def cost_fmt(value):
    if value is None:
        return 'Unknown'
    return '$0.00' if value == 0 else f'${value:.5f}'


def chart(ax, rows, metric):
    labels = [f"{r['size']} jobs\n{NAMES[r['architecture']]}" for r in rows]
    positions = list(range(len(rows)))
    colors = [COLORS[r['architecture']] for r in rows]
    if metric == 'latency':
        values = [r['median_seconds'] for r in rows]
        bars = ax.bar(positions, values, color=colors, width=.64)
        ax.errorbar(positions, values, yerr=[[v - r['min_seconds'] for r, v in zip(rows, values)],
                                            [r['max_seconds'] - v for r, v in zip(rows, values)]],
                    fmt='none', ecolor=INK, capsize=4, linewidth=1)
        ax.set_ylabel('Seconds per resume')
        ax.set_title('End-to-end time', loc='left', fontweight='bold', color=INK)
        maximum = max(r['max_seconds'] for r in rows)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + maximum * .04, f'{value:.1f}s', ha='center', fontsize=8, color=INK)
    elif metric == 'tokens':
        complete = [r['mean_input_tokens'] is not None for r in rows]
        inputs = [r['mean_input_tokens'] or 0 for r in rows]
        outputs = [r['mean_output_tokens'] or 0 for r in rows]
        ax.bar(positions, inputs, color=colors, width=.64, label='Input')
        ax.bar(positions, outputs, bottom=inputs, color=colors, width=.64, hatch='////', edgecolor='white', label='Output')
        ax.set_ylabel('Tokens per resume (all calls)')
        ax.set_title('Measured input + output', loc='left', fontweight='bold', color=INK)
        maximum = max([a + b for a, b in zip(inputs, outputs)] + [1])
        for x, a, b, known in zip(positions, inputs, outputs, complete):
            ax.text(x, a + b + maximum * .04, token_fmt(a + b) if known else 'Unknown', ha='center', fontsize=8, color=INK)
        ax.legend(loc='upper left', fontsize=7, frameon=False)
    else:
        values = [r['mean_cost_usd'] or 0 for r in rows]
        ax.bar(positions, values, color=colors, width=.64)
        maximum = max(values + [1e-5])
        for x, row, value in zip(positions, rows, values):
            ax.text(x, value + maximum * .05, cost_fmt(row['mean_cost_usd']), ha='center', fontsize=9, color=INK)
        ax.set_ylabel('USD per resume (API tokens)')
        ax.set_title('Measured token charges', loc='left', fontweight='bold', color=INK)
    ax.set_ylim(0, maximum * 1.28)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CBD5E1')
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)


def text(fig, x, y, value, size=9, color=INK, **kwargs):
    fig.text(x, y, value, fontsize=size, color=color, va='top', **kwargs)


def paragraph(fig, x, y, value, width=106, size=8.5, color=MUTED, line_height=.016):
    lines = textwrap.wrap(value, width=width)
    text(fig, x, y, '\n'.join(lines), size=size, color=color, linespacing=1.45)
    return y - len(lines) * line_height


def header(fig, title, subtitle, page):
    text(fig, .07, .955, title, size=21, weight='bold')
    text(fig, .07, .914, subtitle, size=9, color=MUTED)
    fig.add_artist(__import__('matplotlib').lines.Line2D([.07, .93], [.884, .884], color='#DCE5ED', linewidth=1))
    text(fig, .07, .035, 'Resume-to-job matching | CSCE 585 | Saved model responses and usage counters', size=7, color=MUTED)
    text(fig, .92, .035, str(page), size=7, color=MUTED, ha='right')


def paired_example(runs):
    pairs = defaultdict(dict)
    for run in runs:
        pairs[(run['resume_id'], run['size'], run['repetition'])][run['architecture']] = run
    identifiers = sorted(pairs, key=lambda item: (not all(r['status'] == 'ok' for r in pairs[item].values()), item))
    for identifier in identifiers:
        if set(pairs[identifier]) == set(METHODS):
            pair = pairs[identifier]
            if pair['single']['job_order'] != pair['orchestrated']['job_order']:
                raise ValueError('Paired output example has different job orders')
            return identifier, pair
    return None, None


def render(directory, output, manifest, runs, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white'})
    for metric, name in (('latency', 'latency_comparison.png'), ('tokens', 'token_usage.png'), ('cost', 'token_cost.png')):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        chart(ax, rows, metric)
        fig.tight_layout()
        fig.savefig(output / name, dpi=180)
        plt.close(fig)
    config = manifest['config']
    local = config.get('provider') == 'ollama'
    successes = sum(r['status'] == 'ok' for r in runs)
    total_seconds = sum(r['seconds'] for r in runs)
    total_known = all(r['total_cost_usd'] is not None for r in rows)
    total_cost = sum(r['known_cost_usd'] for r in rows) if total_known else None
    provider = 'Local Ollama' if local else 'OpenAI API'
    repetitions = len({r['repetition'] for r in runs})
    context = config.get('context_tokens')
    native_context = manifest.get('provider_metadata', {}).get('model_context_tokens')
    server = manifest.get('provider_metadata', {}).get('server', {}).get('version')
    local_context = (f" Ollama {server or 'version unrecorded'}; configured context {context or 'unrecorded'} tokens"
                     + (f' (model limit {native_context}).' if native_context else '.')) if local else ''
    measurement_notes = manifest.get('measurement_notes', [])
    if isinstance(measurement_notes, str):
        measurement_notes = [measurement_notes]
    cache_note = ' '.join(measurement_notes) or 'Model loading and server cache state can affect local timings; no forced cold-cache claim is made.'
    batch = json.loads((directory / 'batch.json').read_text()) if (directory / 'batch.json').exists() else None
    expected = len(manifest['schedule'])
    with PdfPages(output / 'pipeline_comparison.pdf', metadata={'Title': 'Measured single-agent vs multi-agent pipeline comparison', 'Author': 'CSCE 585'}) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        header(fig, 'Single agent vs multi agent', f"Measured comparison  /  {provider}  /  {config['model']}", 1)
        text(fig, .07, .865, 'ACTUAL EXECUTIONS', size=8, color=MUTED, weight='bold')
        text(fig, .07, .842, f'{successes}/{len(runs)}', size=27, weight='bold')
        text(fig, .07, .8, f'successful runs; {len(runs)}/{expected} scheduled attempts', size=8, color=MUTED)
        text(fig, .4, .842, f'{total_seconds / 60:.1f} min', size=27, weight='bold')
        text(fig, .4, .8, 'sum of measured matcher durations', size=8, color=MUTED)
        text(fig, .77, .842, cost_fmt(total_cost), size=27, weight='bold')
        text(fig, .77, .8, 'API token charges', size=8, color=MUTED)
        if local:
            paragraph(fig, .07, .768, 'Local inference has no API token billing. Electricity and hardware costs were not measured.', size=8.5)
        else:
            paragraph(fig, .07, .768, f"Charges use measured response token counts and frozen prices dated {config['price_date']}.", size=8.5)
        columns = [.07, .18, .355, .47, .60, .715, .845]
        headings = ['Jobs', 'Pipeline', 'OK / runs', 'Median time', 'Input / run', 'Output / run', 'USD / run']
        top, height = .724, .032
        fig.add_artist(Rectangle((.065, top - height), .87, height, transform=fig.transFigure, facecolor=INK, edgecolor='none'))
        for x, label in zip(columns, headings):
            text(fig, x, top - .01, label, size=7.5, color='white', weight='bold')
        for i, row in enumerate(rows):
            y = top - (i + 1) * height
            fig.add_artist(Rectangle((.065, y - height), .87, height, transform=fig.transFigure, facecolor=PALE if i % 2 == 0 else 'white', edgecolor='none'))
            cells = [str(row['size']), NAMES[row['architecture']], f"{row['successful_runs']} / {row['attempted_runs']}",
                     f"{row['median_seconds']:.1f}s", token_fmt(row['mean_input_tokens']), token_fmt(row['mean_output_tokens']), cost_fmt(row['mean_cost_usd'])]
            for x, value in zip(columns, cells):
                text(fig, x, y - .01, value, size=8, color=COLORS[row['architecture']] if x == columns[1] else INK)
        left = fig.add_axes([.1, .27, .36, .23])
        right = fig.add_axes([.57, .27, .36, .23])
        chart(left, rows, 'latency')
        chart(right, rows, 'tokens')
        paragraph(fig, .07, .215, 'Time bars show the median across attempted runs; whiskers show the measured minimum and maximum. Token bars show mean input + output per attempt, including every worker and retry. Missing metadata is unknown.', size=8)
        details = [f"{r['size']} jobs, {NAMES[r['architecture']]}: {r['api_calls']} calls, {r['retry_calls']} retries, {r['min_seconds']:.1f}-{r['max_seconds']:.1f}s range, {fmt(r['successful_runs_per_minute'], 2)} successes/min."
                   for r in rows]
        text(fig, .07, .147, '\n'.join(details), size=7.4, color=MUTED, linespacing=1.5)
        if batch and number(batch.get('seconds')):
            text(fig, .07, .073, f"Recorded batch wall time: {batch['seconds'] / 60:.2f} min (includes loop/logging overhead).", size=7.5, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        identifier, pair = paired_example(runs)
        subtitle = f'{identifier[0]} / {identifier[1]} candidate jobs / repetition {identifier[2] + 1}' if pair else 'No same-input pair in the supplied logs'
        header(fig, 'Actual ranked outputs', subtitle, 2)
        text(fig, .07, .866, 'SAME RESUME, PREFERENCES, JOBS AND JOB ORDER', size=8, color=MUTED, weight='bold')
        if pair:
            for method, x in zip(METHODS, (.07, .53)):
                run = pair[method]
                text(fig, x, .835, NAMES[method], size=16, color=COLORS[method], weight='bold')
                text(fig, x, .804, f"{run['seconds']:.1f}s  /  {len(run['calls'])} calls  /  {cost_fmt(run.get('cost_usd'))}", size=8, color=MUTED)
                if run['status'] != 'ok':
                    text(fig, x, .767, 'No validated ranking', size=10, color=COLORS[method], weight='bold')
                    last = next((call for call in reversed(run['calls']) if call.get('status') == 'failed'), {})
                    detail = last.get('error_detail') or last.get('error') or run.get('error', 'Failure recorded')
                    paragraph(fig, x, .737, f"The pipeline failed after its allowed retry. Last worker: {last.get('worker', 'unknown')}. {textwrap.shorten(detail, width=180, placeholder='...')}", width=48, size=8, color=INK)
                    paragraph(fig, x, .647, 'Its measured time, response tokens, and known charges are included on page 1. Raw failed responses are saved in runs.jsonl.', width=48, size=8, color=MUTED)
                for rank, match in enumerate(run['output'].get('matches', []) if run['status'] == 'ok' else [], 1):
                    y = .767 - (rank - 1) * .082
                    text(fig, x, y, f"{rank}. {match['job_id']}", size=10, color=COLORS[method], weight='bold')
                    # Keep the actual words and mark truncation. Full outputs remain in runs.jsonl.
                    lines = textwrap.wrap(' '.join(match['explanation'].split()), width=48)
                    if len(lines) > 3:
                        lines = lines[:3]
                        lines[-1] = textwrap.shorten(lines[-1] + ' [continued]', width=48, placeholder='...')
                    text(fig, x, y - .025, '\n'.join(lines), size=8, color=INK, linespacing=1.45)
            if all(run['status'] == 'ok' for run in pair.values()):
                overlap = len({m['job_id'] for m in pair['single']['output']['matches']} & {m['job_id'] for m in pair['orchestrated']['output']['matches']})
                caption = f'Top-five overlap in this pair: {overlap}/5 jobs. Explanations are shortened saved excerpts; full explanations are in runs.jsonl. Overlap measures agreement, not relevance.'
            else:
                caption = 'This same-input pair includes a failed pipeline. Only validated rankings are shown. Explanations are shortened saved excerpts; complete outputs and failure details are in runs.jsonl.'
            paragraph(fig, .07, .338, caption, size=8.5)
        else:
            paragraph(fig, .07, .8, 'Both pipelines were attempted, but no same-input pair finished successfully. Inspect the recorded failures before comparing rankings.', size=10)
        text(fig, .07, .29, 'WHAT THESE MEASUREMENTS COVER', size=9, weight='bold')
        notes = [
            'Single agent makes one ranking request. Multi agent parses the resume, extracts all job requirements in one batch, and ranks using those structured outputs. Workers run sequentially; the controller adds no LLM call.',
            f"This {manifest.get('phase', 'recorded')} uses {len({r['resume_id'] for r in runs})} resumes, {repetitions} repetition(s) per condition, and {len(runs)} attempts. Failures and retries count toward time and tokens. Throughput is successes / total matcher minutes, at one resume at a time.",
            'Human relevance labels are not used in this timing/cost report. Precision@5 and a winner on recommendation quality are not inferred from these sample rankings.',
            f"Recorded platform: {manifest.get('platform', 'not recorded')}; Python {manifest.get('python', 'not recorded')}.{local_context}",
            cache_note,
            f"Price source ({config.get('price_date', 'date unknown')}): {config.get('price_source', 'not recorded')}",
        ]
        y = .266
        for note in notes:
            y = paragraph(fig, .07, y, note, width=123, size=7.2, line_height=.013) - .008
        text(fig, .07, .068, f"Source: {directory.as_posix()}/runs.jsonl + manifest.json", size=7, color=MUTED)
        text(fig, .07, .052, f"{len(runs)}/{expected} scheduled attempts present. No extrapolated or hypothetical results.", size=7, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing actual runs.jsonl and manifest.json')
    parser.add_argument('--output', type=Path, default=Path('data/reports'))
    args = parser.parse_args()
    try:
        manifest, runs = load_runs(args.directory)
        rows = summarize(manifest, runs)
        # Verify a same-order output pair before writing any report artifacts.
        paired_example(runs)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        raise SystemExit(f'Cannot report measured comparison: {exc}') from exc
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'source_directory': args.directory.as_posix(), 'source_runs_sha256': hashlib.sha256((args.directory / 'runs.jsonl').read_bytes()).hexdigest(),
               'model': manifest['config']['model'], 'provider': manifest['config'].get('provider', 'openai'),
               'observed_runs': len(runs), 'expected_runs': len(manifest['schedule']),
               'schedule_complete': len(runs) == len(manifest['schedule']), 'results': rows,
               'local_cost_note': 'API token charges only; electricity and hardware costs were not measured.' if manifest['config'].get('provider') == 'ollama' else None}
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    render(args.directory, args.output, manifest, runs, rows)
    print(f"Saved measured comparison to {args.output / 'pipeline_comparison.pdf'}")
    print(f"{len(runs)}/{len(manifest['schedule'])} scheduled attempts; {sum(r['status'] == 'ok' for r in runs)} successful.")


if __name__ == '__main__':
    main()

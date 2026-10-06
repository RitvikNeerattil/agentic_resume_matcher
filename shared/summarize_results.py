"""Regenerate comparison and paired-resume tables from saved runs (no API calls)."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from shared.common import ROOT, digest


def precision(run, labels):
    if run['status'] != 'ok':
        return 0.0
    seen = set()
    credit = 0
    allowed = set(run['job_order'])
    for match in run['output'].get('matches', [])[:5]:
        jid = match.get('job_id')
        if jid in allowed and jid not in seen:
            credit += labels[(run['resume_id'], jid)]
        seen.add(jid)
    return credit / 5


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, labels):
    groups = defaultdict(list)
    for run in runs:
        groups[(run['architecture'], run['size'])].append(run)
    rows = []
    for (architecture, size), group in sorted(groups.items()):
        scores = [precision(r, labels) for r in group]
        seconds = [r['seconds'] for r in group]
        unknown = sum(r['cost_usd'] is None for r in group)
        rows.append({'architecture': architecture, 'size': size, 'runs': len(group),
                     'precision_at_5': statistics.mean(scores), 'precision_sd': statistics.pstdev(scores),
                     'median_seconds': statistics.median(seconds), 'min_seconds': min(seconds), 'max_seconds': max(seconds),
                     'mean_cost_usd': '' if unknown else statistics.mean(r['cost_usd'] for r in group),
                     'unknown_cost_runs': unknown, 'input_tokens_known': sum(c['usage']['prompt_tokens'] for r in group for c in r['calls'] if c['usage']),
                     'output_tokens_known': sum(c['usage']['completion_tokens'] for r in group for c in r['calls'] if c['usage']),
                     'failure_rate': sum(r['status'] != 'ok' for r in group) / len(group),
                     'throughput_per_minute': sum(r['status'] == 'ok' for r in group) * 60 / sum(seconds) if sum(seconds) else 0})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/labels/reference_labels.csv')
    args = parser.parse_args()
    runs = [json.loads(line) for line in (args.directory / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    if not runs or any(r['phase'] == 'pilot' for r in runs):
        raise SystemExit('Use completed eval runs; pilot does not have held-out labels')
    raw = list(csv.DictReader(args.labels.open(encoding='utf-8')))
    if len(raw) != 400 or any(r['relevant'] not in ('0', '1') for r in raw):
        raise SystemExit('Need 400 completed reference labels')
    labels = {(r['resume_id'], r['job_id']): int(r['relevant']) for r in raw}
    expected = {(f'R{i:02}', f'J{j:03}') for i in range(3, 11) for j in range(1, 51)}
    if set(labels) != expected:
        raise SystemExit('Invalid or duplicate reference pairs')
    manifest = json.loads((args.directory / 'manifest.json').read_text(encoding='utf-8'))
    if digest(args.labels.read_text(encoding='utf-8')) != manifest['labels_sha256']:
        raise SystemExit('Labels changed since execution; use the frozen reference labels')
    rows = summarize(runs, labels)
    write_csv(args.directory / 'comparison.csv', rows)
    paired = []
    for rid in sorted({r['resume_id'] for r in runs}):
        for size in (20, 50):
            methods = {a: [precision(r, labels) for r in runs if r['resume_id'] == rid and r['size'] == size and r['architecture'] == a] for a in ('single', 'orchestrated')}
            if all(methods.values()):
                paired.append({'resume_id': rid, 'size': size,
                               'single_p5': statistics.mean(methods['single']),
                               'orchestrated_p5': statistics.mean(methods['orchestrated']),
                               'difference': statistics.mean(methods['orchestrated']) - statistics.mean(methods['single'])})
    write_csv(args.directory / 'paired_resumes.csv', paired)
    preparation_path = args.directory / 'preparation.json'
    if preparation_path.exists():
        prep = json.loads(preparation_path.read_text(encoding='utf-8'))
        known = all(c['cost_usd'] is not None for c in prep['calls']) and all(r['cost_usd'] is not None for r in runs)
        (args.directory / 'reuse_cost.json').write_text(json.dumps({
            'preparation_seconds': prep['seconds'],
            'preparation_cost_usd': sum(c['cost_usd'] for c in prep['calls']) if known else None,
            'amortized_cost_usd': (sum(c['cost_usd'] for c in prep['calls']) + sum(r['cost_usd'] for r in runs)) / len(runs) if known else None}, indent=2))
    key = lambda r: (r['resume_id'], r['size'], r['repetition'], r['architecture'])
    expected = {key(r) for r in manifest['schedule']}
    expected_runs = len(expected)
    complete = len(runs) == expected_runs and {key(r) for r in runs} == expected
    decision = {'complete': complete, 'expected_runs': expected_runs, 'observed_runs': len(runs), 'criterion_met': None}
    by_method = {r['architecture']: r for r in rows if r['size'] == 50}
    if complete and 'single' in by_method and 'orchestrated' in by_method:
        a, b = by_method['single'], by_method['orchestrated']
        if a['mean_cost_usd'] != '' and b['mean_cost_usd'] != '':
            decision['criterion_met'] = (b['precision_at_5'] - a['precision_at_5'] >= .05 - 1e-12 and b['mean_cost_usd'] <= 2 * a['mean_cost_usd'] and b['median_seconds'] <= 2 * a['median_seconds'])
    (args.directory / 'decision.json').write_text(json.dumps(decision, indent=2))
    print(json.dumps(rows, indent=2))
    print('Throughput uses total matcher time per condition, including failed runs and retries.')


if __name__ == '__main__':
    main()

"""Orchestrator pilot (4 runs), main (48 runs), or reuse (6 runs + preparation)."""
import argparse
import csv
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from shared.common import Client, ROOT, digest
from orchestrator import extract_requirements, match as orchestrated_match


def prompt_snapshot():
    return {str(p.relative_to(ROOT)): p.read_text()
            for directory in ('shared', 'orchestrator/prompts')
            for p in sorted((ROOT / directory).glob('*.txt'))}


def plan(phase, seed):
    ids = ['R01', 'R02'] if phase == 'pilot' else [f'R{i:02}' for i in range(3, 11)]
    if phase == 'reuse':
        ids = ids[:2]
    rows = []
    for rid in ids:
        for size in ([50] if phase == 'reuse' else [20, 50]):
            for repetition in range(1 if phase == 'pilot' else 3):
                rows.append({'resume_id': rid, 'size': size, 'repetition': repetition,
                             'architecture': 'reuse' if phase == 'reuse' else 'orchestrated',
                             'order_seed': seed + int(rid[1:]) * 100 + size + repetition})
    return rows


def preflight(config, resumes, phase):
    errors = []
    if not config['model'] or not config['price_date'] or not config['price_source']:
        errors.append('Set a fixed model version, price_date and price_source in config')
    if any(not isinstance(v, (int, float)) or v < 0 for v in config['prices_per_million'].values()):
        errors.append('Set all three dated prices (USD per million tokens)')
    if not os.environ.get('OPENAI_API_KEY'):
        errors.append('Set OPENAI_API_KEY in your shell')
    ids = {r['resume_id'] for r in plan(phase, config['seed'])}
    for rid in sorted(ids):
        path = ROOT / 'data/resumes' / resumes[rid]['file']
        if not path.is_file() or not path.read_text().strip():
            errors.append('Missing cleaned resume: ' + rid)
    if phase != 'pilot':
        path = ROOT / 'data/labels/reference_labels.csv'
        if not path.exists():
            errors.append('Finish independent human labels before evaluating predictions')
        else:
            rows = list(csv.DictReader(path.open()))
            pairs = {(r['resume_id'], r['job_id']) for r in rows if r['relevant'] in ('0', '1')}
            expected = {(f'R{i:02}', f'J{j:03}') for i in range(3, 11) for j in range(1, 51)}
            if pairs != expected or len(rows) != 400:
                errors.append('Reference labels must contain exactly 400 complete unique eval pairs')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'main', 'reuse'], default='pilot')
    parser.add_argument('--config', type=Path, default=ROOT / 'shared/experiment.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pilot')
    parser.add_argument('--pilot-dir', type=Path, help='Completed pilot directory that freezes model, prompts and workload')
    parser.add_argument('--execute', action='store_true', help='Make paid API requests; default only prints plan')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    resumes = {r['resume_id']: r for r in csv.DictReader((ROOT / 'data/resumes/resumes.csv').open())}
    jobs = [json.loads(line) for line in (ROOT / 'data/jobs/jobs_50.jsonl').read_text().splitlines()]
    assert len(jobs) == 50 and len({j['job_id'] for j in jobs}) == 50
    assert sum(j['in_small_workload'] for j in jobs) == 20
    schedule = plan(args.phase, config['seed'])
    errors = preflight(config, resumes, args.phase)
    print(json.dumps({'phase': args.phase, 'matching_runs': len(schedule),
                      'preparation_calls': int(args.phase == 'reuse'), 'blockers': errors, 'plan': schedule}, indent=2))
    if not args.execute:
        return
    if errors:
        raise SystemExit('Preflight failed; no API requests made')
    if args.phase != 'pilot':
        if args.pilot_dir is None:
            raise SystemExit('Pass --pilot-dir with a completed pilot before evaluation')
        pilot = json.loads((args.pilot_dir / 'manifest.json').read_text())
        pilot_runs = [json.loads(line) for line in (args.pilot_dir / 'runs.jsonl').read_text().splitlines()]
        current_prompts = prompt_snapshot()
        if len(pilot_runs) != 4 or any(r['status'] != 'ok' for r in pilot_runs):
            raise SystemExit('Complete all four orchestrator pilot runs successfully before freezing')
        if pilot['config'] != config or pilot['prompts'] != current_prompts or pilot['jobs_sha256'] != digest(jobs):
            raise SystemExit('Config, prompts or workload changed since pilot; run a new pilot')
    if args.output.exists():
        raise SystemExit('Choose a new output directory; existing results are never overwritten')
    args.output.mkdir(parents=True)
    prompts = prompt_snapshot()
    freeze = {'config': config, 'prompts': prompts, 'jobs_sha256': digest(jobs),
              'schedule': schedule, 'platform': platform.platform(), 'python': platform.python_version(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'phase': args.phase,
              'preferences': {rid: resumes[rid] for rid in {r['resume_id'] for r in schedule}},
              'resumes_sha256': {rid: digest((ROOT / 'data/resumes' / resumes[rid]['file']).read_text())
                                 for rid in {r['resume_id'] for r in schedule}}}
    if args.phase != 'pilot':
        freeze['labels_sha256'] = digest((ROOT / 'data/labels/reference_labels.csv').read_text())
    (args.output / 'manifest.json').write_text(json.dumps(freeze, indent=2))
    spent = 0.0
    requirements = None
    if args.phase == 'reuse':
        client = Client(config)
        start = time.perf_counter()
        try:
            requirements = extract_requirements(client, jobs)
        finally:
            preparation = {'calls': client.calls, 'seconds': time.perf_counter() - start}
            (args.output / 'preparation.json').write_text(json.dumps(preparation, indent=2))
        (args.output / 'requirements.json').write_text(json.dumps(requirements, indent=2))
        if any(c['cost_usd'] is None for c in client.calls):
            raise SystemExit('Unknown preparation cost; inspect logs before continuing')
        spent = sum(c['cost_usd'] for c in client.calls)
    batch_start = time.perf_counter()
    for row in schedule:
        if spent >= config['budget_usd']:
            raise SystemExit('Budget reached; completed runs preserved. A request may cross the budget.')
        subset = [j for j in jobs if row['size'] == 50 or j['in_small_workload']]
        random.Random(row['order_seed']).shuffle(subset)
        resume = resumes[row['resume_id']]
        client = Client(config)
        start = time.perf_counter()
        record = {**row, 'phase': args.phase, 'job_order': [j['job_id'] for j in subset]}
        try:
            record['output'] = orchestrated_match(client, (ROOT / 'data/resumes' / resume['file']).read_text(),
                {k: v for k, v in resume.items() if k.startswith('pref_') or k in ('needs_sponsorship', 'other_constraints')},
                subset, requirements=requirements)
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='failed', error=type(exc).__name__, output={'matches': []})
        record.update(seconds=time.perf_counter() - start, calls=client.calls)
        record['cost_usd'] = sum(c['cost_usd'] for c in client.calls) if all(c['cost_usd'] is not None for c in client.calls) else None
        spent += sum(c['cost_usd'] or 0 for c in client.calls)
        with (args.output / 'runs.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(f"{row['resume_id']} {row['size']} {row['architecture']}: {record['status']}", flush=True)
        if record['cost_usd'] is None:
            raise SystemExit('Unknown billed usage; logs saved. Check provider usage before continuing.')
    (args.output / 'batch.json').write_text(json.dumps({'seconds': time.perf_counter() - batch_start, 'known_cost_usd': spent}))


if __name__ == '__main__':
    main()

"""Measure pipelines on paired inputs; benchmark uses dev resumes, without labels."""
import argparse
import csv
import json
import os
import platform
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from shared.common import Client, ROOT, digest, load_api_key
from orchestrator import extract_requirements, match as orchestrated_match
from single_agent import match as single_match


def prompt_snapshot():
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8')
            for directory in ('shared', 'orchestrator/prompts')
            for p in sorted((ROOT / directory).glob('*.txt'))}


def local_metadata(config):
    """Check context capacity and record the exact local model before inference."""
    base = config['base_url'].rstrip('/')
    def fetch(path, body=None):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    models = fetch('/api/tags')['models']
    model = next((m for m in models if m['name'] == config['model']), None)
    if model is None:
        raise SystemExit('Install the configured Ollama model before running')
    info = fetch('/api/show', {'model': config['model']})
    capacity = [v for k, v in info['model_info'].items() if k.endswith('.context_length')]
    if not capacity or config['context_tokens'] > min(capacity):
        raise SystemExit('Configured context exceeds the model capacity')
    return {'server': fetch('/api/version'), 'model': model, 'model_context_tokens': min(capacity)}


def plan(phase, seed, architectures='orchestrated', repetitions=None):
    ids = ['R01', 'R02'] if phase in ('pilot', 'benchmark') else [f'R{i:02}' for i in range(3, 11)]
    if phase == 'reuse':
        ids = ids[:2]
    rows = []
    for rid in ids:
        for size in ([50] if phase == 'reuse' else [20, 50]):
            for repetition in range(repetitions if repetitions is not None else (1 if phase == 'pilot' else 3)):
                methods = ['reuse'] if phase == 'reuse' else (['single', 'orchestrated'] if architectures == 'both' else [architectures])
                if (int(rid[1:]) + size + repetition) % 2:
                    methods.reverse()
                for method in methods:
                    rows.append({'resume_id': rid, 'size': size, 'repetition': repetition,
                                 'architecture': method,
                                 'order_seed': seed + int(rid[1:]) * 100 + size + repetition})
    return rows


def preflight(config, resumes, phase):
    load_api_key()
    errors = []
    if not config['model'] or not config['price_date'] or not config['price_source']:
        errors.append('Set a fixed model version, price_date and price_source in config')
    if any(not isinstance(v, (int, float)) or v < 0 for v in config['prices_per_million'].values()):
        errors.append('Set all three dated prices (USD per million tokens)')
    if config.get('provider') != 'ollama' and not os.environ.get('OPENAI_API_KEY'):
        errors.append('Set OPENAI_API_KEY in your shell')
    ids = {r['resume_id'] for r in plan(phase, config['seed'])}
    for rid in sorted(ids):
        path = ROOT / 'data/resumes' / resumes[rid]['file']
        if not path.is_file() or not path.read_text(encoding='utf-8').strip():
            errors.append('Missing cleaned resume: ' + rid)
    if phase not in ('pilot', 'benchmark'):
        path = ROOT / 'data/labels/reference_labels.csv'
        if not path.exists():
            errors.append('Finish independent human labels before evaluating predictions')
        else:
            rows = list(csv.DictReader(path.open(encoding='utf-8')))
            pairs = {(r['resume_id'], r['job_id']) for r in rows if r['relevant'] in ('0', '1')}
            expected = {(f'R{i:02}', f'J{j:03}') for i in range(3, 11) for j in range(1, 51)}
            if pairs != expected or len(rows) != 400:
                errors.append('Reference labels must contain exactly 400 complete unique eval pairs')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'benchmark', 'main', 'reuse'], default='pilot')
    parser.add_argument('--architectures', choices=['single', 'orchestrated', 'both'], default='orchestrated')
    parser.add_argument('--repetitions', type=int, help='Benchmark repetitions per dev resume/size (default three)')
    parser.add_argument('--config', type=Path, default=ROOT / 'shared/experiment.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pilot')
    parser.add_argument('--pilot-dir', type=Path, help='Completed pilot directory that freezes model, prompts and workload')
    parser.add_argument('--execute', action='store_true', help='Run inference with the selected provider; default only prints plan')
    args = parser.parse_args()
    if args.repetitions is not None and (args.phase != 'benchmark' or args.repetitions < 1):
        parser.error('--repetitions requires benchmark phase and a positive count')
    if args.phase == 'reuse' and args.architectures != 'orchestrated':
        parser.error('Reuse is an orchestrator ablation')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    resumes = {r['resume_id']: r for r in csv.DictReader((ROOT / 'data/resumes/resumes.csv').open(encoding='utf-8'))}
    jobs = [json.loads(line) for line in (ROOT / 'data/jobs/jobs_50.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(jobs) == 50 and len({j['job_id'] for j in jobs}) == 50
    assert sum(j['in_small_workload'] for j in jobs) == 20
    schedule = plan(args.phase, config['seed'], args.architectures, args.repetitions)
    errors = preflight(config, resumes, args.phase)
    print(json.dumps({'phase': args.phase, 'matching_runs': len(schedule),
                      'preparation_calls': int(args.phase == 'reuse'), 'blockers': errors, 'plan': schedule}, indent=2))
    if not args.execute:
        return
    if errors:
        raise SystemExit('Preflight failed; no API requests made')
    provider_metadata = local_metadata(config) if config.get('provider') == 'ollama' else {}
    if args.phase not in ('pilot', 'benchmark'):
        if args.pilot_dir is None:
            raise SystemExit('Pass --pilot-dir with a completed pilot before evaluation')
        pilot = json.loads((args.pilot_dir / 'manifest.json').read_text(encoding='utf-8'))
        pilot_runs = [json.loads(line) for line in (args.pilot_dir / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
        current_prompts = prompt_snapshot()
        pilot_methods = {r['architecture'] for r in pilot_runs}
        needed_methods = {r['architecture'] for r in schedule} - {'reuse'}
        if args.phase == 'reuse':
            needed_methods = {'orchestrated'}
        if len(pilot_runs) != len(pilot['schedule']) or any(r['status'] != 'ok' for r in pilot_runs) or not needed_methods <= pilot_methods:
            raise SystemExit('Complete a successful pilot covering the requested architectures before freezing')
        if pilot['config'] != config or pilot['prompts'] != current_prompts or pilot['jobs_sha256'] != digest(jobs):
            raise SystemExit('Config, prompts or workload changed since pilot; run a new pilot')
    if args.output.exists():
        raise SystemExit('Choose a new output directory; existing results are never overwritten')
    args.output.mkdir(parents=True)
    prompts = prompt_snapshot()
    freeze = {'config': config, 'prompts': prompts, 'jobs_sha256': digest(jobs),
              'provider_metadata': provider_metadata,
              'measurement_notes': ['Timings use one continuous local server session; model loading and cache state were not reset or standardized. The server had already been used for development before this benchmark.'] if config.get('provider') == 'ollama' else [],
              'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p.read_text(encoding='utf-8'))
                              for directory in ('shared', 'orchestrator', 'single_agent')
                              for p in sorted((ROOT / directory).glob('*.py'))},
              'schedule': schedule, 'platform': platform.platform(), 'python': platform.python_version(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'phase': args.phase,
              'preferences': {rid: resumes[rid] for rid in {r['resume_id'] for r in schedule}},
              'resumes_sha256': {rid: digest((ROOT / 'data/resumes' / resumes[rid]['file']).read_text(encoding='utf-8'))
                                 for rid in {r['resume_id'] for r in schedule}}}
    if args.phase not in ('pilot', 'benchmark'):
        freeze['labels_sha256'] = digest((ROOT / 'data/labels/reference_labels.csv').read_text(encoding='utf-8'))
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
            matcher = single_match if row['architecture'] == 'single' else orchestrated_match
            extras = {} if row['architecture'] == 'single' else {'requirements': requirements}
            record['output'] = matcher(client, (ROOT / 'data/resumes' / resume['file']).read_text(encoding='utf-8'),
                {k: v for k, v in resume.items() if k.startswith('pref_') or k in ('needs_sponsorship', 'other_constraints')},
                subset, **extras)
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='failed', error=type(exc).__name__, output={'matches': []})
        record.update(seconds=time.perf_counter() - start, calls=client.calls)
        record['cost_usd'] = sum(c['cost_usd'] for c in client.calls) if all(c['cost_usd'] is not None for c in client.calls) else None
        spent += sum(c['cost_usd'] or 0 for c in client.calls)
        with (args.output / 'runs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(f"{row['resume_id']} {row['size']} {row['architecture']}: {record['status']}", flush=True)
        if record['cost_usd'] is None:
            raise SystemExit('Unknown billed usage; logs saved. Check provider usage before continuing.')
    (args.output / 'batch.json').write_text(json.dumps({'seconds': time.perf_counter() - batch_start, 'known_cost_usd': spent}))


if __name__ == '__main__':
    main()

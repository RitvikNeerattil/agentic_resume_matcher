"""Single vs orchestrated matching on the public rjdfit dataset (labeled resume/job pairs).

Each resume in rjdfit appears with many job descriptions, each labeled
"No Fit" / "Potential Fit" / "Good Fit". For every selected resume we build a
pool from its labeled postings, ask both matchers for the top five, and score
the rankings against those labels. No hand labeling is needed.

  python -m shared.rjdfit_eval download
  python -m shared.rjdfit_eval build data/rjdfit/rjdfit_train.csv data/rjdfit/rjdfit_test.csv
  python -m shared.rjdfit_eval run --config shared/local_experiment.json --output results/rjdfit_local --execute
  python -m shared.rjdfit_eval summarize results/rjdfit_local
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from shared.common import Client, ROOT, digest
from orchestrator import match as orchestrated_match
from single_agent import match as single_match

POOLS = ROOT / 'data/rjdfit_workload/pools.jsonl'
DATA = ROOT / 'data/rjdfit'
GAIN = {'no fit': 0, 'potential fit': 1, 'good fit': 2}


def prompt_snapshot():
    """Every prompt either pipeline uses, so a run records exactly what the models were told."""
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8')
            for directory in ('shared', 'orchestrator/prompts')
            for p in sorted((ROOT / directory).glob('*.txt'))}


def cmd_download(args):
    """Fetch the public dataset from Hugging Face into data/rjdfit/ (needs `pip install datasets`)."""
    from datasets import load_dataset
    data = load_dataset('cnamuangtoun/resume-job-description-fit')
    DATA.mkdir(parents=True, exist_ok=True)
    for split, name in (('train', 'rjdfit_train.csv'), ('test', 'rjdfit_test.csv')):
        data[split].to_csv(DATA / name)
    print(f'Saved {len(data["train"])} + {len(data["test"])} rows to {DATA}')


def structure(text, limit=300):
    """Turn messy PDF-style text into short lines so line-ID evidence selection works.

    rjdfit text has hard wraps and run-on sentences ("...analytics.Adept in..."),
    so we flatten whitespace, split at sentence ends (with or without a space),
    and break any piece longer than `limit` characters at a comma or space.
    """
    flat = re.sub(r'\s+', ' ', text).strip()
    pieces = re.split(r'(?<=[.;!?])\s*(?=[A-Z0-9])', flat)
    lines = []
    for piece in pieces:
        piece = piece.strip()
        while len(piece) > limit:
            cut = piece.rfind(', ', 0, limit)
            if cut < limit // 2:
                cut = piece.rfind(' ', 0, limit)
            if cut <= 0:
                cut = limit
            lines.append(piece[:cut + 1].strip())
            piece = piece[cut + 1:].strip()
        if piece:
            lines.append(piece)
    return '\n'.join(lines)


def read_rows(paths):
    csv.field_size_limit(sys.maxsize)
    rows = []
    for path in paths:
        with open(path, encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            missing = {'resume_text', 'job_description_text', 'label'} - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f'{path}: missing columns {sorted(missing)}')
            rows.extend(reader)
    return rows


def build_pools(rows, resumes=30, min_pool=8, max_pool=20, min_relevant=2, seed=585):
    """Group labeled rows by resume and sample a fixed-size pool for each chosen resume."""
    labels = {}
    for r in rows:
        label = r['label'].strip().lower()
        if label not in GAIN:
            raise SystemExit('Unexpected label: ' + r['label'])
        labels.setdefault((r['resume_text'], r['job_description_text']), set()).add(label)
    by_resume = defaultdict(dict)
    for (resume, job), seen in labels.items():
        if len(seen) == 1:  # drop pairs whose duplicate rows disagree
            by_resume[resume][job] = next(iter(seen))
    eligible = [(r, jobs) for r, jobs in by_resume.items()
                if len(jobs) >= min_pool and sum(GAIN[v] > 0 for v in jobs.values()) >= min_relevant]
    eligible.sort(key=lambda item: hashlib.sha256(item[0].encode()).hexdigest())
    rng = random.Random(seed)
    rng.shuffle(eligible)
    job_ids = {}
    pools = []
    for n, (resume, jobs) in enumerate(eligible[:resumes], 1):
        chosen = sorted(jobs, key=lambda j: hashlib.sha256(j.encode()).hexdigest())
        rng.shuffle(chosen)
        chosen = chosen[:max_pool]
        # Keep at least min_relevant positives after sampling.
        if sum(GAIN[jobs[j]] > 0 for j in chosen) < min_relevant:
            positives = [j for j in jobs if GAIN[jobs[j]] > 0][:min_relevant]
            chosen = positives + [j for j in chosen if j not in positives][:max_pool - len(positives)]
        pool = []
        for job in chosen:
            jid = job_ids.setdefault(job, f'D{len(job_ids) + 1:03}')
            pool.append({'job_id': jid, 'description': structure(job), 'label': jobs[job]})
        pools.append({'resume_id': f'P{n:02}', 'split': 'dev' if n <= 2 else 'eval',
                      'resume': structure(resume), 'pool': pool})
    return pools


def cmd_build(args):
    rows = read_rows(args.inputs)
    pools = build_pools(rows, args.resumes, args.min_pool, args.max_pool, args.min_relevant, args.seed)
    if len(pools) < args.resumes:
        print(f'Only {len(pools)} resumes met the pool rules (wanted {args.resumes}).')
    POOLS.parent.mkdir(parents=True, exist_ok=True)
    POOLS.write_text(''.join(json.dumps(p) + '\n' for p in pools), encoding='utf-8')
    sizes = [len(p['pool']) for p in pools]
    rel = [sum(GAIN[j['label'].lower()] > 0 for j in p['pool']) / len(p['pool']) for p in pools]
    chars = [len(p['resume']) for p in pools]
    jchars = [len(j['description']) for p in pools for j in p['pool']]
    print(f'{len(rows)} rows -> {len(pools)} pools ({sum(p["split"] == "dev" for p in pools)} dev) written to {POOLS}')
    print(f'pool size {min(sizes)}-{max(sizes)} | relevant share {statistics.mean(rel):.0%} | '
          f'resume chars median {statistics.median(chars):.0f} | job chars median {statistics.median(jchars):.0f}')
    lines = [p['resume'].count('\n') + 1 for p in pools]
    print(f'resume lines median {statistics.median(lines):.0f} (line IDs need several; check this is not 1)')


def load_pools():
    if not POOLS.exists():
        raise SystemExit('Run the build command first')
    return [json.loads(line) for line in POOLS.read_text(encoding='utf-8').splitlines()]


def score(run, pool):
    """nDCG@5 (graded), precision@5 and top-1 hit; failed or invalid output scores 0."""
    gains = {j['job_id']: GAIN[j['label'].lower()] for j in pool}
    ids = []
    if run['status'] == 'ok':
        for m in run['output'].get('matches', [])[:5]:
            if m.get('job_id') in gains and m['job_id'] not in ids:
                ids.append(m['job_id'])
    discount = lambda i: 1 / math.log2(i + 2)
    dcg = sum((2 ** gains[j] - 1) * discount(i) for i, j in enumerate(ids))
    ideal = sum((2 ** g - 1) * discount(i) for i, g in enumerate(sorted(gains.values(), reverse=True)[:5]))
    return {'ndcg5': dcg / ideal if ideal else 0.0,
            'precision5': sum(gains[j] > 0 for j in ids) / 5,
            'top1_relevant': float(bool(ids) and gains[ids[0]] > 0),
            'random_precision5': sum(g > 0 for g in gains.values()) / len(gains)}


def run_pools(pools, config, architectures, repetitions, output, transport=None, execute=True, resume_run=False):
    runs_path = output / 'runs.jsonl'
    done = set()
    if runs_path.exists():
        if not resume_run:
            raise SystemExit('Output exists; pass --resume to continue or choose a new directory')
        done = {(r['resume_id'], r['repetition'], r['architecture']) for r in
                map(json.loads, runs_path.read_text(encoding='utf-8').splitlines())}
    methods = ['single', 'orchestrated'] if architectures == 'both' else [architectures]
    schedule = []
    for n, p in enumerate(pools):
        for rep in range(repetitions):
            order = methods[::-1] if (n + rep) % 2 else methods  # balance which method runs first
            schedule += [(p, rep, a) for a in order]
    print(f'{len(schedule)} runs over {len(pools)} resumes ({len(done)} already done)')
    if not execute:
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps({
        'config': config, 'workload_sha256': digest(pools), 'prompts': prompt_snapshot(),
        'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p.read_text(encoding='utf-8'))
                        for d in ('shared', 'orchestrator', 'single_agent') for p in sorted((ROOT / d).glob('*.py'))}, 'architectures': methods,
        'repetitions': repetitions, 'started_at': datetime.now(timezone.utc).isoformat()}, indent=2))
    spent = 0.0
    for p, rep, arch in schedule:
        if (p['resume_id'], rep, arch) in done:
            continue
        if spent >= config['budget_usd']:
            raise SystemExit('Budget reached; completed runs preserved')
        jobs = [{'job_id': j['job_id'], 'description': j['description']} for j in p['pool']]
        random.Random(config['seed'] + int(p['resume_id'][1:]) * 10 + rep).shuffle(jobs)
        client = Client(config, transport)
        record = {'resume_id': p['resume_id'], 'split': p['split'], 'repetition': rep, 'architecture': arch,
                  'size': len(jobs), 'job_order': [j['job_id'] for j in jobs]}
        start = time.perf_counter()
        try:
            matcher = single_match if arch == 'single' else orchestrated_match
            record['output'] = matcher(client, p['resume'], {}, jobs)
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='failed', error=type(exc).__name__, output={'matches': []})
        record.update(seconds=time.perf_counter() - start, calls=client.calls)
        known = all(c['cost_usd'] is not None for c in client.calls)
        record['cost_usd'] = sum(c['cost_usd'] for c in client.calls) if known else None
        record['scores'] = score(record, p['pool'])
        spent += sum(c['cost_usd'] or 0 for c in client.calls)
        with runs_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(p['resume_id'], arch, record['status'], f"ndcg5={record['scores']['ndcg5']:.2f}", flush=True)


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        if r['split'] == 'eval':
            groups[r['architecture']].append(r)
    rows = []
    for arch, g in sorted(groups.items()):
        tokens = [sum((c['usage'] or {}).get('prompt_tokens', 0) + (c['usage'] or {}).get('completion_tokens', 0)
                      for c in r['calls']) for r in g]
        costs = [r['cost_usd'] for r in g if r['cost_usd'] is not None]
        rows.append({'architecture': arch, 'runs': len(g),
                     'ndcg5': statistics.mean(r['scores']['ndcg5'] for r in g),
                     'precision5': statistics.mean(r['scores']['precision5'] for r in g),
                     'top1_relevant': statistics.mean(r['scores']['top1_relevant'] for r in g),
                     'random_precision5': statistics.mean(r['scores']['random_precision5'] for r in g),
                     'median_seconds': statistics.median(r['seconds'] for r in g),
                     'median_tokens': statistics.median(tokens),
                     'mean_cost_usd': statistics.mean(costs) if len(costs) == len(g) else '',
                     'failure_rate': sum(r['status'] != 'ok' for r in g) / len(g)})
    paired = []
    by = defaultdict(dict)
    for r in runs:
        if r['split'] == 'eval':
            by[r['resume_id']].setdefault(r['architecture'], []).append(r['scores']['ndcg5'])
    for rid, d in sorted(by.items()):
        if 'single' in d and 'orchestrated' in d:
            paired.append({'resume_id': rid, 'single_ndcg5': statistics.mean(d['single']),
                           'orchestrated_ndcg5': statistics.mean(d['orchestrated']),
                           'difference': statistics.mean(d['orchestrated']) - statistics.mean(d['single'])})
    return rows, paired


def by_pool_size(runs, edges=(12, 16)):
    """nDCG@5 and time per architecture for small, medium and large pools (a stand-in for workload size)."""
    def bucket(n):
        return f'<= {edges[0]} jobs' if n <= edges[0] else (f'{edges[0] + 1}-{edges[1]} jobs' if n <= edges[1] else f'> {edges[1]} jobs')
    groups = defaultdict(list)
    for r in runs:
        if r['split'] == 'eval':
            groups[(bucket(r['size']), r['architecture'])].append(r)
    return [{'pool_size': b, 'architecture': a, 'runs': len(g),
             'ndcg5': statistics.mean(r['scores']['ndcg5'] for r in g),
             'median_seconds': statistics.median(r['seconds'] for r in g)}
            for (b, a), g in sorted(groups.items())]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def cmd_summarize(args):
    runs = [json.loads(line) for line in (args.directory / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    rows, paired = summarize(runs)
    write_csv(args.directory / 'comparison.csv', rows)
    write_csv(args.directory / 'paired_resumes.csv', paired)
    sizes = by_pool_size(runs)
    write_csv(args.directory / 'by_pool_size.csv', sizes)
    print(json.dumps(rows, indent=2))
    for row in sizes:
        print(row)
    if paired:
        diffs = [p['difference'] for p in paired]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        print(f'paired nDCG@5 (orchestrated minus single) over {len(diffs)} resumes: mean {statistics.mean(diffs):+.3f}, '
              f'orchestrated better on {wins}, worse on {losses}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('download')
    b = sub.add_parser('build')
    b.add_argument('inputs', nargs='*', type=Path, default=[DATA / 'rjdfit_train.csv', DATA / 'rjdfit_test.csv'])
    b.add_argument('--resumes', type=int, default=30)
    b.add_argument('--min-pool', type=int, default=8)
    b.add_argument('--max-pool', type=int, default=20)
    b.add_argument('--min-relevant', type=int, default=2)
    b.add_argument('--seed', type=int, default=585)
    r = sub.add_parser('run')
    r.add_argument('--config', type=Path, default=ROOT / 'shared/experiment.json')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--architectures', choices=['single', 'orchestrated', 'both'], default='both')
    r.add_argument('--repetitions', type=int, default=1)
    r.add_argument('--split', choices=['dev', 'eval', 'all'], default='all')
    r.add_argument('--execute', action='store_true')
    r.add_argument('--resume', action='store_true')
    s = sub.add_parser('summarize')
    s.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'build':
        cmd_build(args)
    elif args.command == 'summarize':
        cmd_summarize(args)
    else:
        pools = [p for p in load_pools() if args.split in ('all', p['split'])]
        config = json.loads(args.config.read_text(encoding='utf-8'))
        run_pools(pools, config, args.architectures, args.repetitions, args.output,
                  execute=args.execute, resume_run=args.resume)


if __name__ == '__main__':
    main()

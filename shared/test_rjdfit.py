"""Offline tests for the rjdfit evaluation. Fake model output is never experiment evidence."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from shared.common import ROOT
from shared.rjdfit_eval import GAIN, build_pools, by_pool_size, run_pools, score, structure, summarize


def fake_rows():
    rows = []
    labels = ['No Fit', 'Potential Fit', 'Good Fit']
    for r in range(4):
        for j in range(12):
            rows.append({'resume_text': f'Resume {r}. BS in CS. Built Python tools. Worked on ETL.',
                         'job_description_text': f'Job {j}. Needs Python. Requires a degree.',
                         'label': labels[(r + j) % 3]})
    rows.append(dict(rows[0]))  # exact duplicate row is fine
    return rows


def transport(payload):
    """Answer each worker by the schema name the matcher sends."""
    name = payload['response_format']['json_schema']['name']
    data = json.loads(payload['messages'][1]['content'])
    if name == 'resume':
        value = {'skills': [{'name': 'Python', 'evidence_id': 'L001'}], 'education': [], 'experience': []}
    elif name == 'requirements':
        value = {'jobs': {j['job_id']: {'required': ['E001'], 'preferred': [], 'constraints': []} for j in data['jobs']}}
    else:
        value = {'matches': [{'job_id': j['job_id'], 'explanation': 'Python fit.'} for j in data['jobs'][:5]]}
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 0}}}


class RjdfitTests(unittest.TestCase):
    def test_pools_and_labels(self):
        pools = build_pools(fake_rows(), resumes=3, min_pool=8, max_pool=10)
        self.assertEqual(len(pools), 3)
        self.assertEqual([p['split'] for p in pools], ['dev', 'dev', 'eval'])
        for p in pools:
            self.assertEqual(len(p['pool']), 10)
            self.assertTrue(all(j['label'] in GAIN for j in p['pool']))
        # build is deterministic
        self.assertEqual(pools, build_pools(fake_rows(), resumes=3, min_pool=8, max_pool=10))

    def test_conflicting_duplicates_dropped(self):
        rows = fake_rows()
        rows.append({**rows[0], 'label': 'Good Fit' if rows[0]['label'] != 'Good Fit' else 'No Fit'})
        pools = build_pools(rows, resumes=4, min_pool=8, max_pool=20)
        self.assertTrue(all(len(p['pool']) <= 12 for p in pools))

    def test_structure_splits_run_on_text(self):
        self.assertEqual(structure('Did SQL.Adept in Python; Led a team'), 'Did SQL.\nAdept in Python;\nLed a team')
        long = ', '.join(f'skill{i}' for i in range(200))
        lines = structure(long).split('\n')
        self.assertTrue(all(len(x) <= 300 for x in lines) and len(lines) > 3)
        self.assertEqual(' '.join(structure('a\n  b c').split()), 'a b c')

    def test_score(self):
        pool = [{'job_id': f'D{i}', 'label': l} for i, l in
                enumerate(['Good Fit', 'Potential Fit', 'No Fit', 'No Fit', 'No Fit', 'No Fit'])]
        ok = lambda ids: {'status': 'ok', 'output': {'matches': [{'job_id': i} for i in ids]}}
        best = score(ok(['D0', 'D1', 'D2', 'D3', 'D4']), pool)
        self.assertAlmostEqual(best['ndcg5'], 1.0)
        self.assertEqual(best['top1_relevant'], 1.0)
        worst = score(ok(['D5', 'D4', 'D3', 'D2', 'D0']), pool)
        self.assertLess(worst['ndcg5'], .5)
        self.assertEqual(score(ok(['D0', 'D0', 'D0', 'D0', 'D0']), pool)['precision5'], .2)  # no duplicate credit
        self.assertEqual(score(ok(['BAD'] * 5), pool)['ndcg5'], 0)
        self.assertEqual(score({'status': 'failed', 'output': {'matches': []}}, pool)['ndcg5'], 0)

    def test_end_to_end_with_fake_model(self):
        config = json.loads((ROOT / 'shared/experiment.json').read_text(encoding='utf-8'))
        pools = build_pools(fake_rows(), resumes=3, min_pool=8, max_pool=10)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            run_pools(pools, config, 'both', 1, out, transport=transport)
            runs = [json.loads(line) for line in (out / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(runs), 6)
            self.assertTrue(all(r['status'] == 'ok' for r in runs))
            run_pools(pools, config, 'both', 1, out, transport=transport, resume_run=True)  # nothing left to do
            self.assertEqual(len((out / 'runs.jsonl').read_text(encoding='utf-8').splitlines()), 6)
            rows, paired = summarize(runs)
            self.assertEqual({r['architecture'] for r in rows}, {'single', 'orchestrated'})
            self.assertEqual(len(paired), 1)  # one eval resume
            sizes = by_pool_size(runs)
            self.assertEqual({r['architecture'] for r in sizes}, {'single', 'orchestrated'})


if __name__ == '__main__':
    unittest.main()

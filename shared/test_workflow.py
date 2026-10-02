"""Offline contract tests. Fake API responses are never experiment evidence."""
import json
import unittest

from shared.common import Client, ROOT, validate_rank
from orchestrator import match
from orchestrator.validation import validate
from shared.run_experiment import plan, prompt_snapshot
from shared.summarize_results import precision


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'shared/experiment.json').read_text())
        self.config['prices_per_million'] = {'input': 1, 'cached_input': .5, 'output': 2}
        self.jobs = [{'job_id': f'J{i:03}', 'description': 'Python required.'} for i in range(1, 6)]
        self.ranking = {'matches': [{'job_id': j['job_id'], 'explanation': 'Python experience; degree unknown.'} for j in self.jobs]}

    def response(self, value):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 40}, 'completion_tokens_details': {'reasoning_tokens': 5}}}

    def client(self, values):
        responses = iter(values)
        return Client(self.config, lambda payload: self.response(next(responses)))

    def test_three_calls_and_reuse(self):
        parsed = {'skills': [{'name': 'Python', 'evidence': 'Python'}], 'education': [], 'experience': []}
        extracted = {'jobs': [{**j, 'required': ['Python'], 'preferred': [], 'constraints': []} for j in self.jobs]}
        client = self.client([parsed, extracted, self.ranking])
        self.assertEqual(match(client, 'Python', {}, self.jobs), self.ranking)
        self.assertEqual([c['worker'] for c in client.calls], ['resume', 'requirements', 'rank'])
        self.assertAlmostEqual(client.calls[0]['cost_usd'], .00012)
        client = self.client([parsed, self.ranking])
        match(client, 'Python', {}, self.jobs, requirements=extracted)
        self.assertEqual(len(client.calls), 2)

    def test_retry_keeps_billed_usage(self):
        client = self.client([{'matches': []}, self.ranking])
        client.call('rank', {'jobs': self.jobs})
        self.assertEqual([c['status'] for c in client.calls], ['failed', 'ok'])
        self.assertAlmostEqual(sum(c['cost_usd'] for c in client.calls), .00024)

    def test_failure_after_one_retry(self):
        client = self.client([{'matches': []}, {'matches': []}])
        with self.assertRaises(ValueError):
            client.call('rank', {'jobs': self.jobs})
        self.assertEqual(len(client.calls), 2)

    def test_evidence_and_coverage(self):
        with self.assertRaises(ValueError):
            validate('resume', {'skills': [{'name': 'Java', 'evidence': 'Java'}], 'education': [], 'experience': []}, {'resume': 'Python'})
        with self.assertRaises(ValueError):
            validate('requirements', {'jobs': []}, {'jobs': self.jobs})
        duplicate = {'matches': [self.ranking['matches'][0]] * 5}
        with self.assertRaises(ValueError):
            validate_rank(duplicate, {'jobs': self.jobs})

    def test_prompt_snapshot_covers_shared_and_orchestrator_prompts(self):
        self.assertEqual(set(prompt_snapshot()), {'shared/rank.txt',
                         'orchestrator/prompts/resume.txt', 'orchestrator/prompts/requirements.txt'})

    def test_orchestrator_schedule(self):
        self.assertEqual(len(plan('pilot', 585)), 4)
        self.assertEqual(len(plan('main', 585)), 48)
        self.assertEqual(len(plan('reuse', 585)), 6)
        schedule = plan('main', 585)
        self.assertTrue(all(r['architecture'] == 'orchestrated' for r in schedule))
        self.assertEqual(schedule, plan('main', 585))
        self.assertEqual(len({(r['resume_id'], r['size'], r['repetition']) for r in schedule}), 48)

    def test_precision_penalties(self):
        labels = {('R03', j['job_id']): 1 for j in self.jobs}
        run = {'resume_id': 'R03', 'status': 'ok', 'job_order': [j['job_id'] for j in self.jobs],
               'output': {'matches': [{'job_id': 'J001'}, {'job_id': 'J001'}, {'job_id': 'INVALID'}]}}
        self.assertEqual(precision(run, labels), .2)
        run['status'] = 'failed'
        self.assertEqual(precision(run, labels), 0)

    def test_frozen_data(self):
        jobs = [json.loads(line) for line in (ROOT / 'data/jobs/jobs_50.jsonl').read_text().splitlines()]
        self.assertEqual(len(jobs), 50)
        self.assertEqual(sum(j['in_small_workload'] for j in jobs), 20)
        self.assertEqual(sum(j['source'] == 'lever' for j in jobs), 25)
        self.assertEqual(sum(j['source'] == 'lever' and j['in_small_workload'] for j in jobs), 10)


if __name__ == '__main__':
    unittest.main()

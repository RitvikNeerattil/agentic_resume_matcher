"""Offline contract tests. Fake API responses are never experiment evidence."""
import json
import unittest
from unittest.mock import patch
from io import BytesIO

from shared.common import Client, ROOT, validate_rank
from orchestrator import match
from orchestrator.validation import validate
from orchestrator.workflow import decode_requirements, requirements_schema
from shared.run_experiment import plan, prompt_snapshot
from shared.summarize_results import precision
from single_agent import match as single_match


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
        extracted = {'jobs': [{**j, 'required': ['Python required.'], 'preferred': [], 'constraints': []} for j in self.jobs]}
        ids = {'jobs': {j['job_id']: {'required': ['E001'], 'preferred': [], 'constraints': []} for j in self.jobs}}
        client = self.client([parsed, ids, self.ranking])
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
        validate('resume', {'skills': [{'name': 'Python', 'evidence': 'Python and SQL'}],
                           'education': [], 'experience': []}, {'resume': 'Python and\nSQL'})
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

    def test_paired_benchmark_schedule(self):
        schedule = plan('benchmark', 585, 'both', 1)
        self.assertEqual(len(schedule), 8)
        self.assertTrue(all(r['resume_id'] in ('R01', 'R02') for r in schedule))
        for a, b in zip(schedule[::2], schedule[1::2]):
            self.assertEqual(a['order_seed'], b['order_seed'])
            self.assertEqual(a['size'], b['size'])
            self.assertNotEqual(a['architecture'], b['architecture'])
        self.assertEqual(len(plan('main', 585, 'both')), 96)

    def test_single_agent_one_call(self):
        client = self.client([self.ranking])
        self.assertEqual(single_match(client, 'Python', {}, self.jobs), self.ranking)
        self.assertEqual([c['worker'] for c in client.calls], ['baseline'])

    def test_source_evidence_ids_cannot_invent_requirements(self):
        job = self.jobs[0]
        data = {'jobs': [{'job_id': job['job_id'], 'description_lines': [
            {'evidence_id': 'E001', 'text': 'Python required.'}]}]}
        output = {'jobs': {job['job_id']: {'required': ['E001'], 'preferred': [], 'constraints': []}}}
        decoded = decode_requirements(output, data, [job])
        self.assertEqual(decoded['jobs'][0]['required'], ['Python required.'])
        output['jobs'][job['job_id']]['required'] = ['E999']
        with self.assertRaises(ValueError):
            decode_requirements(output, data, [job])

    def test_local_response_usage_and_zero_api_billing(self):
        config = {**self.config, 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434',
                  'context_tokens': 131072, 'prices_per_million': {'input': 0, 'cached_input': 0, 'output': 0}}
        raw = {'model': config['model'], 'message': {'content': json.dumps(self.ranking)}, 'done_reason': 'stop',
               'prompt_eval_count': 123, 'eval_count': 45}
        with patch('shared.common.urllib.request.urlopen', return_value=BytesIO(json.dumps(raw).encode())) as opener:
            client = Client(config)
            self.assertEqual(single_match(client, 'Python', {}, self.jobs), self.ranking)
        request = json.loads(opener.call_args[0][0].data)
        self.assertFalse(request['truncate'])
        self.assertFalse(request['shift'])
        self.assertEqual(client.calls[0]['usage']['prompt_tokens'], 123)
        self.assertEqual(client.calls[0]['usage']['completion_tokens'], 45)
        self.assertEqual(client.calls[0]['cost_usd'], 0)

    def test_requirements_schema_and_decoder_require_all_fifty_jobs(self):
        jobs = [{'job_id': f'J{i:03}', 'description': 'Python required.'} for i in range(1, 51)]
        numbered = [{**job, 'description_lines': [{'evidence_id': 'E001', 'text': job['description']}]} for job in jobs]
        schema = requirements_schema(numbered)
        self.assertEqual(set(schema['properties']['jobs']['required']), {job['job_id'] for job in jobs})
        self.assertFalse(schema['properties']['jobs']['additionalProperties'])
        hosted = requirements_schema(numbered, provider='openai')
        self.assertEqual(hosted['properties']['jobs']['required'], schema['properties']['jobs']['required'])
        items = hosted['properties']['jobs']['properties']['J001']['properties']['required']['items']
        self.assertNotIn('enum', items)
        self.assertEqual(items['pattern'], '^E[0-9]{3,}$')
        values = {job['job_id']: {'required': ['E001'], 'preferred': [], 'constraints': []} for job in jobs}
        decoded = decode_requirements({'jobs': values}, {'jobs': numbered}, jobs)
        self.assertEqual(len(decoded['jobs']), 50)
        del values['J050']
        with self.assertRaises(ValueError):
            decode_requirements({'jobs': values}, {'jobs': numbered}, jobs)

    def test_structured_schema_reaches_native_and_hosted_requests(self):
        schema = {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}
        seen = []
        def hosted(payload):
            seen.append(payload['response_format'])
            return self.response(self.ranking)
        Client(self.config, hosted).call('rank', {'jobs': self.jobs}, schema=schema)
        self.assertEqual(seen[0], {'type': 'json_schema', 'json_schema': {'name': 'rank', 'strict': True, 'schema': schema}})
        config = {**self.config, 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434', 'context_tokens': 131072}
        raw = {'model': config['model'], 'message': {'content': json.dumps(self.ranking)}, 'done_reason': 'stop',
               'prompt_eval_count': 123, 'eval_count': 45}
        with patch('shared.common.urllib.request.urlopen', return_value=BytesIO(json.dumps(raw).encode())) as opener:
            Client(config).call('rank', {'jobs': self.jobs}, schema=schema)
        self.assertEqual(json.loads(opener.call_args[0][0].data)['format'], schema)

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

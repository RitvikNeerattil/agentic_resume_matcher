"""Shared API client, accounting, input cleanup and top-five output contract."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clean(text):
    return '\n'.join(line.rstrip() for line in text.replace('\r\n', '\n').replace('\r', '\n').splitlines()).strip()


def prepare_input(resume, preferences, jobs):
    return {'resume': clean(resume), 'preferences': preferences,
            'jobs': [{**j, 'description': clean(j['description'])} for j in jobs]}


def validate_rank(value, data):
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    matches = value.get('matches', [])
    ids = [m['job_id'] for m in matches]
    allowed = {j['job_id'] for j in data['jobs']}
    if len(ids) != 5 or len(set(ids)) != 5 or not set(ids) <= allowed:
        raise ValueError('Expected five distinct valid job IDs')
    if any(not isinstance(m.get('explanation'), str) or not m['explanation'].strip() for m in matches):
        raise ValueError('Missing explanation')
    return value


class Client:
    def __init__(self, config, transport=None):
        self.config = config
        self.transport = transport or self.request
        self.calls = []

    def request(self, payload):
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('Set OPENAI_API_KEY before live execution')
        request = urllib.request.Request('https://api.openai.com/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def call(self, worker, data, *, prompt=None, validator=validate_rank):
        if prompt is None:
            prompt = (ROOT / 'shared/rank.txt').read_text()
        for attempt in range(2):
            start = time.perf_counter()
            log = {'worker': worker, 'retry': attempt, 'model': self.config['model'],
                   'prices_per_million': self.config['prices_per_million'], 'usage': None, 'cost_usd': None}
            try:
                response = self.transport({'model': self.config['model'], 'temperature': self.config['temperature'],
                    'max_completion_tokens': self.config['output_limits'][worker], 'store': False,
                    'response_format': {'type': 'json_object'},
                    'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(data)}]})
                log['raw_response'] = response
                usage = response.get('usage')
                log['usage'] = usage
                if usage:
                    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                    prices = self.config['prices_per_million']
                    log['cost_usd'] = ((usage['prompt_tokens'] - cached) * prices['input'] + cached * prices['cached_input'] + usage['completion_tokens'] * prices['output']) / 1e6
                choice = response['choices'][0]
                if choice['finish_reason'] != 'stop':
                    raise ValueError('Incomplete model output')
                value = validator(json.loads(choice['message']['content']), data)
                log['status'] = 'ok'
                return value
            except Exception as exc:
                # HTTP error bodies can contain sensitive request data; retain only type.
                log['status'] = 'failed'
                log['error'] = type(exc).__name__
                if attempt == 1:
                    raise
            finally:
                log['seconds'] = time.perf_counter() - start
                self.calls.append(log)


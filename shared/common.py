"""Shared API client, accounting, input cleanup and top-five output contract."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_api_key():
    """Read only OPENAI_API_KEY from the git-ignored .env, without executing it."""
    if os.environ.get('OPENAI_API_KEY'):
        return
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            name, separator, value = line.partition('=')
            if separator and name.strip() == 'OPENAI_API_KEY':
                value = value.strip().strip('\"\'')
                if value:
                    os.environ['OPENAI_API_KEY'] = value


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
        if self.config.get('provider') == 'ollama':
            body = {'model': payload['model'], 'messages': payload['messages'],
                    'format': payload['response_format'].get('json_schema', {}).get('schema', 'json'),
                    'stream': False, 'keep_alive': '30m',
                    'truncate': False, 'shift': False,
                    'options': {'temperature': payload['temperature'],
                                'num_ctx': self.config['context_tokens'],
                                'num_predict': payload['max_completion_tokens'],
                                'seed': self.config['seed']}}
            request = urllib.request.Request(self.config['base_url'].rstrip('/') + '/api/chat',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 1800)) as response:
                raw = json.load(response)
            # Normalize the response while preserving the original server counters.
            usage = None
            if 'prompt_eval_count' in raw and 'eval_count' in raw:
                usage = {'prompt_tokens': raw['prompt_eval_count'], 'completion_tokens': raw['eval_count'],
                         'prompt_tokens_details': {'cached_tokens': raw.get('prompt_eval_cached_count', 0)}}
            return {'model': raw['model'], 'usage': usage,
                    'choices': [{'finish_reason': raw.get('done_reason'), 'message': raw['message']}],
                    'ollama_response': raw}
        load_api_key()
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('Set OPENAI_API_KEY before live execution')
        request = urllib.request.Request('https://api.openai.com/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def call(self, worker, data, *, prompt=None, validator=validate_rank, schema=None):
        if prompt is None:
            prompt = (ROOT / 'shared/rank.txt').read_text()
        retry_hint = ''
        for attempt in range(2):
            start = time.perf_counter()
            log = {'worker': worker, 'retry': attempt, 'model': self.config['model'],
                   'prices_per_million': self.config['prices_per_million'], 'usage': None, 'cost_usd': None}
            try:
                response = self.transport({'model': self.config['model'], 'temperature': self.config['temperature'],
                    'max_completion_tokens': self.config['output_limits'][worker], 'store': False,
                    'service_tier': 'default',
                    'response_format': ({'type': 'json_schema', 'json_schema': {
                        'name': worker, 'strict': True, 'schema': schema}}
                        if schema is not None else {'type': 'json_object'}),
                    'messages': [{'role': 'system', 'content': prompt + retry_hint}, {'role': 'user', 'content': json.dumps(data)}]})
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
                if isinstance(exc, ValueError):
                    log['error_detail'] = str(exc)
                    retry_hint = '\nPrevious output failed validation: ' + str(exc) + '. Correct the JSON using only the supplied input.'
                if hasattr(exc, 'code'):
                    log['http_status'] = exc.code
                if attempt == 1:
                    raise
            finally:
                log['seconds'] = time.perf_counter() - start
                self.calls.append(log)

"""Resume parsing, batched requirement extraction, then scoring and ranking."""
from pathlib import Path

from shared.common import clean, prepare_input
from .validation import validate

PROMPTS = Path(__file__).resolve().parent / 'prompts'


def parse_resume(client, resume):
    return client.call('resume', {'resume': clean(resume)},
                       prompt=(PROMPTS / 'resume.txt').read_text(),
                       validator=lambda value, data: validate('resume', value, data))


def requirements_schema(numbered_jobs, provider='ollama'):
    """Require every job key and constrain selections to that job's source IDs."""
    fields = ('required', 'preferred', 'constraints')
    properties = {}
    for job in numbered_jobs:
        evidence_ids = [line['evidence_id'] for line in job['description_lines']]
        # OpenAI limits schemas to 1,000 enum values; membership is still
        # checked against this job's source lines by decode_requirements.
        item = ({'type': 'string', 'enum': evidence_ids} if provider == 'ollama'
                else {'type': 'string', 'pattern': '^E[0-9]{3,}$'})
        selection = {'type': 'array', 'items': item}
        properties[job['job_id']] = {
            'type': 'object',
            'properties': {field: selection for field in fields},
            'required': list(fields),
            'additionalProperties': False,
        }
    return {
        'type': 'object',
        'properties': {
            'jobs': {
                'type': 'object',
                'properties': properties,
                'required': list(properties),
                'additionalProperties': False,
            },
        },
        'required': ['jobs'],
        'additionalProperties': False,
    }


def decode_requirements(value, data, original_jobs):
    """Resolve selected source IDs to quotes before checking evidence/coverage."""
    if not isinstance(value, dict) or set(value) != {'jobs'} or not isinstance(value['jobs'], dict):
        raise ValueError('Return a JSON object with jobs keyed by supplied job IDs')
    sources = {j['job_id']: {line['evidence_id']: line['text'] for line in j['description_lines']}
               for j in data['jobs']}
    rows = value['jobs']
    if set(rows) != set(sources):
        raise ValueError('Extraction must cover every supplied job exactly once')
    decoded = []
    for job in data['jobs']:
        job_id = job['job_id']
        row = rows[job_id]
        if not isinstance(row, dict) or set(row) != {'required', 'preferred', 'constraints'}:
            raise ValueError('Each job must contain only required, preferred, and constraints arrays')
        result = {'job_id': job_id}
        for field in ('required', 'preferred', 'constraints'):
            selected = row.get(field)
            if not isinstance(selected, list) or any(not isinstance(i, str) for i in selected):
                raise ValueError(field + ' must be a list of evidence IDs; use [] when absent')
            if any(i not in sources[job_id] for i in selected):
                raise ValueError('Unknown evidence ID; use only IDs listed for that job')
            result[field] = [sources[job_id][i] for i in selected]
        decoded.append(result)
    return validate('requirements', {'jobs': decoded}, {'jobs': original_jobs})


def extract_requirements(client, jobs):
    original_jobs = [{**j, 'description': clean(j['description'])} for j in jobs]
    numbered = []
    for job in original_jobs:
        lines = [line.strip() for line in job['description'].splitlines() if line.strip()]
        numbered.append({**{k: v for k, v in job.items() if k != 'description'},
                         'description_lines': [{'evidence_id': f'E{i:03}', 'text': line}
                                               for i, line in enumerate(lines, 1)]})
    data = {'jobs': numbered}
    return client.call('requirements', data,
                       prompt=(PROMPTS / 'requirements.txt').read_text(),
                       schema=requirements_schema(numbered, client.config.get('provider', 'openai')),
                       validator=lambda value, data: decode_requirements(value, data, original_jobs))


def rank_jobs(client, parsed, preferences, jobs, requirements):
    # Shared ranking instructions and output contract are available for team integration.
    return client.call('rank', {'resume_facts': parsed, 'preferences': preferences,
                               'jobs': jobs, 'requirements': requirements})


def match(client, resume, preferences, jobs, requirements=None):
    data = prepare_input(resume, preferences, jobs)
    parsed = parse_resume(client, data['resume'])
    extracted = requirements if requirements is not None else extract_requirements(client, data['jobs'])
    validate('requirements', extracted, {'jobs': data['jobs']})
    return rank_jobs(client, parsed, preferences, data['jobs'], extracted)

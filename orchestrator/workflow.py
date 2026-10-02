"""Resume parsing, batched requirement extraction, then scoring and ranking."""
from pathlib import Path

from shared.common import clean, prepare_input
from .validation import validate

PROMPTS = Path(__file__).resolve().parent / 'prompts'


def parse_resume(client, resume):
    return client.call('resume', {'resume': clean(resume)},
                       prompt=(PROMPTS / 'resume.txt').read_text(),
                       validator=lambda value, data: validate('resume', value, data))


def extract_requirements(client, jobs):
    data = {'jobs': [{**j, 'description': clean(j['description'])} for j in jobs]}
    return client.call('requirements', data,
                       prompt=(PROMPTS / 'requirements.txt').read_text(),
                       validator=lambda value, data: validate('requirements', value, data))


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

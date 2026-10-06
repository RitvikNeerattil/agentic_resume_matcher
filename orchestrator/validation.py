"""Evidence and complete-job coverage checks for extraction workers."""


def supported(quote, source):
    """Keep exact words, allowing only line-break and spacing differences."""
    return isinstance(quote, str) and bool(quote.strip()) and ' '.join(quote.split()) in ' '.join(source.split())


def validate(worker, value, data):
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    if worker == 'resume':
        for key in ('skills', 'education', 'experience'):
            if not isinstance(value.get(key), list):
                raise ValueError('Missing resume field: ' + key)
        for skill in value['skills']:
            if not isinstance(skill, dict) or not isinstance(skill.get('name'), str) or not skill['name']:
                raise ValueError('Invalid skill')
            quote = skill.get('evidence')
            if not supported(quote, data['resume']):
                raise ValueError('Unsupported resume evidence')
        for quote in value['education'] + value['experience']:
            if not supported(quote, data['resume']):
                raise ValueError('Unsupported resume evidence')
    elif worker == 'requirements':
        jobs = {j['job_id']: j for j in data['jobs']}
        rows = value.get('jobs', [])
        ids = [r['job_id'] for r in rows]
        if len(ids) != len(jobs) or set(ids) != set(jobs):
            raise ValueError('Extraction must cover every job exactly once')
        for row in rows:
            for field in ('required', 'preferred', 'constraints'):
                if not isinstance(row.get(field), list):
                    raise ValueError('Missing requirements field: ' + field)
                for quote in row[field]:
                    if not supported(quote, jobs[row['job_id']]['description']):
                        raise ValueError('Unsupported job evidence')
    else:
        raise ValueError("Unknown extraction worker: " + worker)
    return value

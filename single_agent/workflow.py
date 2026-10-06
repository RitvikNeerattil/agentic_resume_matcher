"""Rank a cleaned resume and all candidate jobs in one LLM call."""

from shared.common import prepare_input


def match(client, resume, preferences, jobs):
    return client.call('baseline', prepare_input(resume, preferences, jobs))

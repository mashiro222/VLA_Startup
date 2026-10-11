"""Crash-safe episode journal. Infrastructure errors never become task failures."""
import json
import os
from pathlib import Path


def read_records(path, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    valid_end = 0
    with path.open('rb') as source:
        while True:
            line = source.readline()
            if not line:
                break
            # Every committed record ends with a newline. A killed write can leave
            # an incomplete final record, which is safe to discard and replay.
            if not line.endswith(b'\n'):
                if repair_tail:
                    with path.open('r+b') as output:
                        output.truncate(valid_end)
                break
            records.append(json.loads(line))
            valid_end = source.tell()
    return records


def append(path, record):
    with Path(path).open('a') as output:
        output.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
        output.flush()
        os.fsync(output.fileno())


def episode_keys(manifest):
    return [f"{r['suite']}:{r['task_id']}:{episode}"
            for r in manifest['tasks'] for episode in range(manifest['trials_per_task'])]


def validate_prefix(records, manifest, run_fingerprint=None):
    expected = episode_keys(manifest)
    if len(records) > len(expected):
        raise ValueError('Journal has more episodes than the manifest')
    for index, record in enumerate(records):
        if record['episode_key'] != expected[index]:
            raise ValueError('Journal is not an ordered, unique prefix of the manifest')
        if record['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Journal belongs to a different manifest')
        if run_fingerprint is not None and record['run_fingerprint'] != run_fingerprint:
            raise ValueError('Cannot resume with changed model or evaluation settings')
        if type(record['success']) is not bool:
            raise ValueError('Success must be a boolean, not a missing/error value')
    return expected

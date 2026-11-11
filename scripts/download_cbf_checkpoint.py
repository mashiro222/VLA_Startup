#!/usr/bin/env python3
"""Download pinned inference files only; verify every SHA-256. No GPU or job."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(manifest, item):
    path = ROOT / '.runtime/checkpoints/pi05_vla_arena' / item['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == item['bytes'] and checksum(path) == item['sha256']:
        return item['path'] + ' verified'
    partial = path.with_name(path.name + '.partial')
    url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['path']}"
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < item['bytes']:
                request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=120) as response:
                    append = offset and response.status == 206
                    with partial.open('ab' if append else 'wb') as out:
                        for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                            out.write(block)
            if partial.stat().st_size != item['bytes'] or checksum(partial) != item['sha256']:
                partial.unlink()
                raise ValueError(f'Checksum failed: {item["path"]}')
            partial.replace(path)
            return item['path'] + ' downloaded and verified'
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, manifest, x) for x in manifest['files']]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
    (ROOT / '.runtime/checkpoints/pi05_vla_arena/verified.json').write_text(json.dumps(manifest, indent=2) + '\n')

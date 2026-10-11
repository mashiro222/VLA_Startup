#!/usr/bin/env python3
"""Fetch pinned inference assets only. No training datasets or model execution."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'


def fetch(url, path, digest, algorithm='sha256', size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid(p):
        if not p.is_file() or (size is not None and p.stat().st_size != size):
            return False
        if algorithm == 'crc32c':
            import google_crc32c
            h = google_crc32c.Checksum()
        else:
            h = hashlib.new(algorithm)
        with p.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                h.update(block)
        actual = h.hexdigest()
        if isinstance(actual, bytes):
            actual = actual.decode('ascii')
        return actual == digest
    if valid(path):
        print('Verified:', path.name, flush=True)
        return
    temporary = path.with_name(path.name + '.partial')
    for attempt in range(4):
        try:
            if valid(temporary):
                temporary.replace(path)
                print('Verified resumed download:', path.name, flush=True)
                return
            if size is not None and temporary.exists() and temporary.stat().st_size >= size:
                temporary.unlink()
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = 'ab' if offset and response.status == 206 else 'wb'
                with temporary.open(mode) as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if not valid(temporary):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f'Checksum mismatch: {path.name}')
            temporary.replace(path)
            print('Downloaded:', path.name, flush=True)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def checkpoint():
    manifest = ROOT / 'config/checkpoint_files.json'
    rows = json.loads(manifest.read_text())
    def one(row):
        encoded = urllib.parse.quote(row['name'], safe='/')
        url = f"https://storage.googleapis.com/openpi-assets/{encoded}?generation={row['generation']}"
        suffix = row['name'].removeprefix('checkpoints/pi05_libero/')
        crc = base64.b64decode(row['crc32c']).hex()
        fetch(url, RUNTIME / 'checkpoints/pi05_libero' / suffix, crc, 'crc32c', int(row['size']))
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(one, rows))
    (RUNTIME / 'checkpoints/pi05_libero/.verified.json').write_text(json.dumps(rows, indent=2))
    token = json.loads((ROOT / 'config/tokenizer.json').read_text())
    fetch(token['url'], RUNTIME / 'cache/openpi/big_vision/paligemma_tokenizer.model',
          token['sha256'], size=token['size'])


def assets():
    spec = json.loads((ROOT / 'config/sources.json').read_text())['assets']
    archive_path = RUNTIME / 'downloads/assets.zip'
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/assets.zip"
    fetch(url, archive_path, spec['sha256'], size=spec['bytes'])
    extract_assets(archive_path, spec)


def extract_assets(archive_path, spec):
    """Extract an already SHA256-verified archive, preserving all runtime files.

    The upstream ZIP has ~450k small files. Cache parent-directory creation and
    limit pending work to 256 entries; four I/O workers avoid serial metadata
    round trips without submitting hundreds of thousands of Future objects.
    Existing files are reused only after size and ZIP CRC checks.
    """
    destination = RUNTIME / 'repos/libero_plus/libero/libero/assets'
    marker = destination / '.vla-assets-sha256'
    if marker.exists() and marker.read_text().strip() == spec['sha256']:
        return
    # The official ZIP can contain an absolute build-machine prefix before assets/.
    # Extract only the subtree, rejecting traversal and symlinks.
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if 'assets' not in parts or member.is_dir():
                continue
            suffix = parts[parts.index('assets') + 1:]
            if not suffix or any(x in ('.', '..') for x in suffix):
                raise ValueError(f'Unsafe archive member: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not supported')
            target = destination.joinpath(*suffix)
            members[target] = member
        for parent in sorted({p.parent for p in members}):
            parent.mkdir(parents=True, exist_ok=True)

        def one(pair):
            target, member = pair
            if target.exists() and target.stat().st_size == member.file_size:
                crc = 0
                with target.open('rb') as existing:
                    for block in iter(lambda: existing.read(1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                if crc == member.CRC:
                    return
            with archive.open(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)

        pairs = list(members.items())
        print('Extracting verified archive files:', len(pairs), flush=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for offset in range(0, len(pairs), 256):
                list(executor.map(one, pairs[offset:offset + 256]))
                if offset % (256 * 40) == 0:
                    print('Asset extraction:', min(offset + 256, len(pairs)), '/', len(pairs), flush=True)
    for required in ('scenes', 'new_objects', 'textures'):
        if not (destination / required).is_dir():
            raise RuntimeError(f'Expected assets subdirectory is missing: {required}')
    marker.write_text(spec['sha256'] + '\n')
    print('Extracted and CRC-checked asset files:', len(pairs), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['checkpoint', 'assets', 'all'], default='all')
    args = parser.parse_args()
    if args.only in ('checkpoint', 'all'):
        checkpoint()
    if args.only in ('assets', 'all'):
        assets()

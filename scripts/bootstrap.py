#!/usr/bin/env python3
"""Download a checksum-verified uv binary and exact upstream Git revisions."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / 'config/sources.json').read_text())


def main():
    tools = ROOT / '.tools'
    tools.mkdir(exist_ok=True)
    if not (tools / 'uv').exists():
        version = SOURCES['uv_version']
        with urllib.request.urlopen(f'https://pypi.org/pypi/uv/{version}/json') as response:
            data = json.load(response)
        wheel = next(x for x in data['urls'] if x['filename'].endswith('manylinux_2_17_x86_64.manylinux2014_x86_64.whl'))
        blob = urllib.request.urlopen(wheel['url']).read()
        if hashlib.sha256(blob).hexdigest() != wheel['digests']['sha256']:
            raise RuntimeError('uv wheel checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            name = next(n for n in archive.namelist() if n.endswith('/scripts/uv'))
            (tools / 'uv').write_bytes(archive.read(name))
        (tools / 'uv').chmod(0o755)
    for key in ('openpi', 'libero_plus', 'libero_clean'):
        spec = SOURCES[key]
        target = ROOT / '.runtime/repos' / key
        if not (target / '.git').exists():
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'init', str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', spec['url']], check=True)
        actual = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if actual.returncode != 0 or actual.stdout.strip() != spec['commit']:
            status = subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True)
            if status.strip():
                raise RuntimeError(f'Refusing to change dirty checkout: {target}')
            subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', spec['commit']], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', spec['commit']], check=True)
        print(key, spec['commit'], flush=True)
    spec = json.loads((ROOT / 'config/imagemagick.json').read_text())
    native = ROOT / '.runtime/native-7.1.1'
    if not (native / 'squashfs-root/usr/lib/libMagickWand-7.Q16HDRI.so').exists():
        blob = urllib.request.urlopen(spec['url']).read()
        if hashlib.sha256(blob).hexdigest() != spec['sha256']:
            raise RuntimeError('ImageMagick checksum mismatch')
        app = ROOT / '.runtime/imagemagick-7.1.1-28.AppImage'
        app.write_bytes(blob)
        app.chmod(0o755)
        native.mkdir(parents=True, exist_ok=True)
        with (native / 'extract.log').open('w') as output:
            subprocess.run([str(app), '--appimage-extract'], cwd=native, stdout=output, check=True)


if __name__ == '__main__':
    main()

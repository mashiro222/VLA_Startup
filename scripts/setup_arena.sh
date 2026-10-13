#!/usr/bin/env bash
# No training, policy inference, or Slurm submission.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
arena_repo="$VLA_RUNTIME/repos/vla_arena"
arena_commit="$(python3 -c 'import json; print(json.load(open("config/arena.json"))["upstream"]["commit"])')"
if [[ ! -d "$arena_repo/.git" ]]; then
    git clone --no-checkout --depth 1 https://github.com/PKU-Alignment/VLA-Arena.git "$arena_repo"
fi
if ! git -C "$arena_repo" cat-file -e "$arena_commit^{commit}"; then
    git -C "$arena_repo" fetch --depth 1 origin "$arena_commit"
fi
# Preserve local modifications; never reset an edited upstream checkout.
if [[ -f "$arena_repo/README.md" ]]; then
    git -C "$arena_repo" diff --exit-code HEAD -- >/dev/null
fi
git -C "$arena_repo" checkout --detach "$arena_commit"
arena_env="$VLA_RUNTIME/arena"
if [[ ! -x "$arena_env/bin/python" ]]; then "$UV" venv --python 3.11 "$arena_env"; fi
if [[ ! -f requirements/arena.lock ]]; then
    "$UV" pip compile requirements/arena.in --python "$arena_env/bin/python" \
        --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        -o requirements/arena.lock
fi
"$UV" pip sync --python "$arena_env/bin/python" requirements/arena.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$arena_env/bin/python" scripts/patch_arena.py
python3 - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
base = root / '.runtime/repos/vla_arena/vla_arena/vla_arena'
config = {'benchmark_root': str(base), 'bddl_files': str(base / 'bddl_files'),
          'init_states': str(base / 'init_files'), 'assets': str(base / 'assets'),
          'datasets': str(root / '.runtime/unused_datasets')}
target = root / '.runtime/config/arena/config.yaml'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(config, indent=2) + '\n')
PY
mkdir -p "$VLA_RUNTIME/locks"
"$UV" pip freeze --python "$arena_env/bin/python" > "$VLA_RUNTIME/locks/arena.freeze.txt"
bash scripts/arena.sh preflight --imports

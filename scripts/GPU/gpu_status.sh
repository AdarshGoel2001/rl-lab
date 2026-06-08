#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

NVIDIA_SMI="${RL_LAB_GPU_NVIDIA_SMI:-/usr/lib/wsl/lib/nvidia-smi}"
REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
NVIDIA_SMI_Q="$(gpu_quote "$NVIDIA_SMI")"

gpu_ssh "
  set +e
  cd $REMOTE_REPO_Q
  printf '== host ==\n'
  hostname
  whoami
  printf '\n== gpu ==\n'
  if [ -x $NVIDIA_SMI_Q ]; then
    $NVIDIA_SMI_Q
  else
    echo 'nvidia-smi not found at $NVIDIA_SMI'
  fi
  printf '\n== tmux ==\n'
  tmux ls 2>/dev/null || true
  printf '\n== latest experiments ==\n'
  latest=\$(find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -10 | cut -d' ' -f2-
  printf '\n== latest run status ==\n'
  if [ -n \"\$latest\" ] && [ -f \"\$latest/run_status.json\" ]; then
    python3 - \"\$latest/run_status.json\" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    print(f'failed to read run_status.json: {exc}')
    raise SystemExit(0)

metrics = payload.get('last_metrics') or {}
rows = [
    ('status', payload.get('status')),
    ('run_id', payload.get('run_id')),
    ('workflow', payload.get('workflow_name')),
    ('global_step', payload.get('global_step')),
    ('phase', payload.get('phase')),
    ('action', payload.get('action')),
    ('hook_state', payload.get('hook_state')),
    ('eval_return_mean', metrics.get('eval/return_mean')),
    ('updated_at', payload.get('updated_at')),
]
for key, value in rows:
    if value is not None:
        print(f'{key}: {value}')
PY
  else
    echo 'no run_status.json found'
  fi
  printf '\n== git ==\n'
  if [ \"\${RL_LAB_GPU_STATUS_GIT:-0}\" = \"1\" ]; then
    git status --short --branch || echo 'git status failed'
  else
    echo 'skipped; set RL_LAB_GPU_STATUS_GIT=1 to run git status on WSL'
  fi
"

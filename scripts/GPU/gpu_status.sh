#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

NVIDIA_SMI="${RL_LAB_GPU_NVIDIA_SMI:-/usr/lib/wsl/lib/nvidia-smi}"
REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
NVIDIA_SMI_Q="$(gpu_quote "$NVIDIA_SMI")"

gpu_ssh "
  set -e
  cd $REMOTE_REPO_Q
  printf '== host ==\n'
  hostname
  whoami
  printf '\n== git ==\n'
  git status --short --branch
  printf '\n== gpu ==\n'
  $NVIDIA_SMI_Q || true
  printf '\n== tmux ==\n'
  tmux ls 2>/dev/null || true
  printf '\n== latest experiments ==\n'
  latest=\$(find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -10 | cut -d' ' -f2-
  printf '\n== latest run status ==\n'
  if [ -n \"\$latest\" ] && [ -f \"\$latest/run_status.json\" ]; then
    python -m json.tool \"\$latest/run_status.json\" 2>/dev/null || cat \"\$latest/run_status.json\"
  else
    echo 'no run_status.json found'
  fi
"

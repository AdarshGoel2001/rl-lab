#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

RUN_PATH=""

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_metrics.sh [--run <remote-experiment-path>]

Print the selected WSL experiment status, checkpoints, TensorBoard event files,
and recent metric-like log lines. Defaults to the latest experiment.

Options:
  --run PATH   Inspect this remote experiment path instead of the latest one.

Environment:
  RL_LAB_GPU_SSH_HOST                  SSH host alias. Default: wsl
  RL_LAB_GPU_REPO                      Remote repo path. Default: /home/omkar/adarsh/rl-lab
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run)
      if [ "$#" -lt 2 ]; then
        echo "--run requires a remote experiment path" >&2
        usage >&2
        exit 2
      fi
      RUN_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RUN_SELECTED=0
REMOTE_RUN_PATH=""
if [ -n "$RUN_PATH" ]; then
  RUN_SELECTED=1
  REMOTE_RUN_PATH="$(printf "%q" "$RUN_PATH")"
fi

REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"

gpu_ssh "
  set -e
  cd $REMOTE_REPO_Q
  if [ '$RUN_SELECTED' = '1' ]; then
    latest=$REMOTE_RUN_PATH
  else
    latest=\$(find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  fi
  if [ -z \"\$latest\" ]; then
    echo 'no experiments found'
    exit 1
  fi
  if [ ! -d \"\$latest\" ]; then
    echo \"run not found: \$latest\" >&2
    exit 1
  fi
  echo \"== experiment ==\"
  echo \"\$latest\"
  echo
  echo '== run status =='
  if [ -f \"\$latest/run_status.json\" ]; then
    python -m json.tool \"\$latest/run_status.json\" 2>/dev/null || cat \"\$latest/run_status.json\"
  else
    echo 'no run_status.json found'
  fi
  echo
  echo '== checkpoints =='
  find \"\$latest/checkpoints\" -maxdepth 1 -type f 2>/dev/null | sort || true
  echo
  echo '== tensorboard events =='
  find \"\$latest\" -type f -name 'events.out.tfevents*' 2>/dev/null | sort | tail -10 || true
  echo
  echo '== recent log metrics =='
  if [ -f \"\$latest/logs/bash_output.log\" ]; then
    tail -40 \"\$latest/logs/bash_output.log\"
  elif [ -f \"\$latest/train.log\" ]; then
    grep -E 'Training complete|return_|world_model|controller|eval' \"\$latest/train.log\" | tail -40 || true
  else
    echo 'no metric log found'
  fi
"

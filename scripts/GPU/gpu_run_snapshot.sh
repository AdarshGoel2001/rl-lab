#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

RUN_PATH=""
TAIL_BYTES="${RL_LAB_GPU_SNAPSHOT_TAIL_BYTES:-262144}"
MAX_EVALS="${RL_LAB_GPU_SNAPSHOT_MAX_EVALS:-20}"

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_run_snapshot.sh [options]

Print a tiny JSON snapshot for a remote run without pulling logs, TensorBoard
events, checkpoints, or replay data.

Options:
  --run PATH      Remote experiment path. Defaults to latest experiment.
  --tail-bytes N  Bytes to read from the end of train.log. Default: 262144.
  --max-evals N   Number of recent eval returns to include. Default: 20.

Environment:
  RL_LAB_GPU_SSH_HOST             SSH host alias. Default: wsl
  RL_LAB_GPU_REPO                 Remote repo path. Default: /home/omkar/adarsh/rl-lab
  RL_LAB_GPU_SNAPSHOT_TAIL_BYTES  Default train.log tail bytes.
  RL_LAB_GPU_SNAPSHOT_MAX_EVALS   Default recent eval count.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run)
      RUN_PATH="${2:?--run requires a remote experiment path}"
      shift 2
      ;;
    --tail-bytes)
      TAIL_BYTES="${2:?--tail-bytes requires a value}"
      shift 2
      ;;
    --max-evals)
      MAX_EVALS="${2:?--max-evals requires a value}"
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

repo_root="$(gpu_repo_root)"
cd "$repo_root"

REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
if [ -n "$RUN_PATH" ]; then
  RUN_PATH_Q="$(gpu_quote "$RUN_PATH")"
  latest="$(
    gpu_ssh "
      set -e
      cd $REMOTE_REPO_Q
      if [ ! -d $RUN_PATH_Q ]; then
        echo 'remote run not found: $RUN_PATH' >&2
        exit 1
      fi
      printf '%s\n' $RUN_PATH_Q
    "
  )"
else
  latest="$(
    gpu_ssh "
      set -e
      cd $REMOTE_REPO_Q
      find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
    "
  )"
fi

if [ -z "$latest" ]; then
  echo "no remote experiments found" >&2
  exit 1
fi

if [[ "$latest" = /* ]]; then
  remote_source="$latest"
else
  remote_source="$GPU_REMOTE_REPO/$latest"
fi

REMOTE_SOURCE_Q="$(gpu_quote "$remote_source")"
gpu_ssh "
  cd $REMOTE_REPO_Q
  python3 scripts/GPU/gpu_run_snapshot_remote.py --run $REMOTE_SOURCE_Q --tail-bytes $TAIL_BYTES --max-evals $MAX_EVALS
"

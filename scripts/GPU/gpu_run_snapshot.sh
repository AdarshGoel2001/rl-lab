#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

RUN_PATH=""
TAIL_BYTES="${RL_LAB_GPU_SNAPSHOT_TAIL_BYTES:-262144}"
MAX_EVALS="${RL_LAB_GPU_SNAPSHOT_MAX_EVALS:-20}"
SNAPSHOT_CHUNK_SIZE="${RL_LAB_GPU_SNAPSHOT_CHUNK_SIZE:-512}"
LOCAL_TMP=""
LOCAL_B64_TMP=""
REMOTE_TMP=""

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
  RL_LAB_GPU_SNAPSHOT_CHUNK_SIZE  Remote base64 chunk size. Default: 512.
USAGE
}

cleanup() {
  rm -f "$LOCAL_TMP" "$LOCAL_B64_TMP"
  if [ -n "${REMOTE_TMP:-}" ]; then
    local remote_tmp_q
    local remote_repo_q
    remote_tmp_q="$(gpu_quote "$REMOTE_TMP")"
    remote_repo_q="$(gpu_quote "$GPU_REMOTE_REPO")"
    gpu_ssh "cd $remote_repo_q && rm -f $remote_tmp_q" >/dev/null 2>&1 || true
  fi
}

append_base64_file_to_file() {
  local b64_file="$1"
  local out_file="$2"
  python3 -c '
import base64
import sys
from pathlib import Path

payload_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
payload = payload_path.read_bytes()
with out_path.open("ab") as handle:
    handle.write(base64.b64decode(payload))
' "$b64_file" "$out_file"
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
trap cleanup EXIT

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
REMOTE_TMP="$(
  gpu_ssh "
    set -e
    cd $REMOTE_REPO_Q
    mkdir -p .agent_runs
    tmp=\$(mktemp .agent_runs/run_snapshot.XXXXXX.json)
    python3 scripts/GPU/gpu_run_snapshot_remote.py --run $REMOTE_SOURCE_Q --tail-bytes $TAIL_BYTES --max-evals $MAX_EVALS > \"\$tmp\"
    printf '%s\n' \"\$tmp\"
  "
)"

REMOTE_TMP_Q="$(gpu_quote "$REMOTE_TMP")"
size_bytes="$(gpu_ssh "cd $REMOTE_REPO_Q && stat -c %s $REMOTE_TMP_Q" </dev/null)"
chunks=$(( (size_bytes + SNAPSHOT_CHUNK_SIZE - 1) / SNAPSHOT_CHUNK_SIZE ))

LOCAL_TMP="$(mktemp "${TMPDIR:-/tmp}/rl_lab_gpu_snapshot.XXXXXX")"
LOCAL_B64_TMP="$LOCAL_TMP.b64"
: > "$LOCAL_TMP"

for ((chunk_index = 0; chunk_index < chunks; chunk_index++)); do
  gpu_ssh "cd $REMOTE_REPO_Q && dd if=$REMOTE_TMP_Q bs=$SNAPSHOT_CHUNK_SIZE skip=$chunk_index count=1 status=none | base64 -w0" > "$LOCAL_B64_TMP" </dev/null
  append_base64_file_to_file "$LOCAL_B64_TMP" "$LOCAL_TMP"
done

cat "$LOCAL_TMP"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

DEST_ROOT="${RL_LAB_GPU_ARTIFACTS_DIR:-remote_artifacts/wsl}"
INCLUDE_CHECKPOINT=0
RUN_PATH=""
ANALYZE=0
RSYNC_SSH="$(gpu_rsync_ssh)"

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_pull_latest.sh [options]

Copy WSL experiment evidence back to the local repo under
remote_artifacts/wsl/<experiment-name>/.

Default files:
  train.log
  logs/bash_output.log
  run_status.json
  run_summary.json
  .hydra/config.yaml
  .hydra/overrides.yaml
  .hydra/hydra.yaml
  pre_run/
  diagnostics/
  checkpoints/final.json
  metrics.csv
  loss_curves.png
  reconstruction_grid.png

Options:
  --run PATH     Pull this remote experiment path instead of the latest one.
  --checkpoint  Also copy checkpoint .pt/.json files and TensorBoard event files.
  --analyze     After pull, export TensorBoard scalar summaries and plots locally.

Environment:
  RL_LAB_GPU_SSH_HOST       SSH host alias. Default: wsl
  RL_LAB_GPU_REPO           Remote repo path. Default: /home/omkar/adarsh/rl-lab
  RL_LAB_GPU_ARTIFACTS_DIR  Local destination root. Default: remote_artifacts/wsl
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run)
      RUN_PATH="${2:?--run requires a remote experiment path}"
      shift 2
      ;;
    --checkpoint)
      INCLUDE_CHECKPOINT=1
      shift
      ;;
    --analyze)
      ANALYZE=1
      INCLUDE_CHECKPOINT=1
      shift
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

experiment_name="$(basename "$latest")"
dest="$DEST_ROOT/$experiment_name"
mkdir -p "$dest"

if [[ "$latest" = /* ]]; then
  remote_source="$latest"
else
  remote_source="$GPU_REMOTE_REPO/$latest"
fi

include_args=(
  --include='/train.log'
  --include='/run_status.json'
  --include='/run_summary.json'
  --include='/metrics.csv'
  --include='/loss_curves.png'
  --include='/reconstruction_grid.png'
  --include='/.hydra/'
  --include='/.hydra/config.yaml'
  --include='/.hydra/overrides.yaml'
  --include='/.hydra/hydra.yaml'
  --include='/pre_run/'
  --include='/pre_run/***'
  --include='/diagnostics/'
  --include='/diagnostics/***'
  --include='/logs/'
  --include='/logs/bash_output.log'
  --include='/checkpoints/'
  --include='/checkpoints/final.json'
)

if [ "$INCLUDE_CHECKPOINT" = "1" ]; then
  include_args+=(--include='/checkpoints/*.pt')
  include_args+=(--include='/checkpoints/*.json')
  include_args+=(--include='/checkpoints/final.pt')
  include_args+=(--include='/checkpoints/latest.pt')
  include_args+=(--include='/checkpoints/best.pt')
  include_args+=(--include='/checkpoints/step_*.pt')
  include_args+=(--include='/checkpoints/step_*.json')
  include_args+=(--include='/checkpoints/best_step_*.pt')
  include_args+=(--include='/checkpoints/best_step_*.json')
  include_args+=(--include='/checkpoints/final.json')
  include_args+=(--include='/tensorboard/')
  include_args+=(--include='/tensorboard/***')
  include_args+=(--include='/runs/')
  include_args+=(--include='/runs/**/')
  include_args+=(--include='/runs/**/events.out.tfevents*')
fi

include_args+=(--exclude='*')

rsync -az --prune-empty-dirs -e "$RSYNC_SSH" \
  "${include_args[@]}" \
  "$GPU_SSH_HOST:$remote_source/" \
  "$dest/"

if [ "$INCLUDE_CHECKPOINT" = "1" ]; then
  mkdir -p "$dest/runs"
  rsync -az --prune-empty-dirs -e "$RSYNC_SSH" \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    "$GPU_SSH_HOST:$GPU_REMOTE_REPO/runs/$experiment_name/" \
    "$dest/runs/" 2>/dev/null || true
fi

if [ "$ANALYZE" = "1" ]; then
  python scripts/research/export_tensorboard_run.py "$dest" --out "$dest/analysis"
fi

echo "pulled latest remote experiment:"
echo "$latest"
echo "local copy:"
echo "$dest"

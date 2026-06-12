#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

DEST_ROOT="${RL_LAB_GPU_ARTIFACTS_DIR:-remote_artifacts/wsl}"
INCLUDE_CHECKPOINT=0
RUN_PATH=""
ANALYZE=0
RSYNC_SSH="$(gpu_rsync_ssh)"
FALLBACK_MAX_BYTES="${RL_LAB_GPU_FALLBACK_MAX_BYTES:-50000}"
SKIPPED_FILE_LOG=""

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

pull_remote_file_base64() {
  local remote_file="$1"
  local local_file="$2"
  local remote_file_q
  local tmp_file
  local tmp_b64_file
  local size_bytes
  local chunk_size=512
  local chunks
  local chunk_index

  remote_file_q="$(gpu_quote "$remote_file")"
  tmp_file="$local_file.tmp"
  tmp_b64_file="$local_file.b64.tmp"
  mkdir -p "$(dirname "$local_file")"
  rm -f "$tmp_file" "$tmp_b64_file"
  size_bytes="$(gpu_ssh "stat -c %s $remote_file_q" </dev/null)"
  if [ "$size_bytes" -gt "$FALLBACK_MAX_BYTES" ]; then
    echo "skipping large fallback file: $remote_file ($size_bytes bytes)" >&2
    if [ -n "$SKIPPED_FILE_LOG" ]; then
      printf '%s\t%s\t%s\n' "$size_bytes" "$remote_file" "$local_file" >> "$SKIPPED_FILE_LOG"
    fi
    return 0
  fi
  chunks=$(( (size_bytes + chunk_size - 1) / chunk_size ))
  : > "$tmp_file"
  for ((chunk_index = 0; chunk_index < chunks; chunk_index++)); do
    gpu_ssh "dd if=$remote_file_q bs=$chunk_size skip=$chunk_index count=1 status=none | base64 -w0" > "$tmp_b64_file" </dev/null
    append_base64_file_to_file "$tmp_b64_file" "$tmp_file"
  done
  rm -f "$tmp_b64_file"
  mv "$tmp_file" "$local_file"
}

list_run_evidence_files() {
  local root="$1"
  local include_checkpoint="$2"
  local root_q
  root_q="$(gpu_quote "$root")"
  gpu_ssh "
    cd $root_q
    {
      for rel in train.log run_status.json run_summary.json metrics.csv loss_curves.png reconstruction_grid.png logs/bash_output.log checkpoints/final.json; do
        [ -f \"\$rel\" ] && printf '%s\n' \"\$rel\"
      done
      for dir in .hydra pre_run diagnostics; do
        [ -d \"\$dir\" ] && find \"\$dir\" -type f -printf '%p\n'
      done
      [ -d runs ] && find runs -type f -name 'events.out.tfevents*' -printf '%p\n'
      if [ '$include_checkpoint' = '1' ] && [ -d checkpoints ]; then
        find checkpoints -type f \\( -name '*.pt' -o -name '*.json' \\) -printf '%p\n'
      fi
    } | sed 's#^\./##' | sort -u
  "
}

pull_run_evidence_base64() {
  local root="$1"
  local destination="$2"
  local include_checkpoint="$3"
  local rel

  echo "rsync failed; falling back to base64 file copy" >&2
  SKIPPED_FILE_LOG="$destination/pull_skipped_files.tsv"
  : > "$SKIPPED_FILE_LOG"
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    pull_remote_file_base64 "$root/$rel" "$destination/$rel"
  done < <(list_run_evidence_files "$root" "$include_checkpoint")
}

pull_external_runs_events_base64() {
  local destination="$1"
  local remote_runs_root="$GPU_REMOTE_REPO/runs/$experiment_name"
  local remote_runs_root_q
  local rel

  remote_runs_root_q="$(gpu_quote "$remote_runs_root")"
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    pull_remote_file_base64 "$remote_runs_root/$rel" "$destination/runs/$rel"
  done < <(
    gpu_ssh "REMOTE_RUNS_ROOT=$remote_runs_root_q python3 -" <<'PY'
import os
from pathlib import Path

root = Path(os.environ["REMOTE_RUNS_ROOT"])
if root.exists():
    for path in sorted(root.rglob("events.out.tfevents*")):
        if path.is_file():
            print(path.relative_to(root))
PY
  )
}

write_checkpoint_manifest() {
  local root="$1"
  local manifest_path="$2"
  local root_q
  local tmp_manifest

  root_q="$(gpu_quote "$root")"
  tmp_manifest="$manifest_path.tmp"
  mkdir -p "$(dirname "$manifest_path")"

  if ! gpu_ssh "RL_LAB_CHECKPOINT_ROOT=$root_q python3 -" > "$tmp_manifest" <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RL_LAB_CHECKPOINT_ROOT"])
checkpoint_dir = root / "checkpoints"
entries = []

if checkpoint_dir.exists():
    for path in sorted(checkpoint_dir.iterdir(), key=lambda item: item.name):
        if not (path.is_file() or path.is_symlink()):
            continue

        entry = {
            "path": path.relative_to(root).as_posix(),
            "kind": "symlink" if path.is_symlink() else "file",
        }

        if path.is_symlink():
            entry["link_target"] = os.readlink(path)

        try:
            stat_result = path.stat()
            entry["size_bytes"] = int(stat_result.st_size)
        except FileNotFoundError:
            entry["size_bytes"] = int(path.lstat().st_size)

        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()

        entries.append(entry)

print(
    json.dumps(
        {
            "schema_version": 1,
            "remote_run": str(root),
            "checkpoint_dir": str(checkpoint_dir),
            "entries": entries,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  then
    rm -f "$tmp_manifest"
    return 1
  fi

  if ! python3 -m json.tool "$tmp_manifest" >/dev/null; then
    rm -f "$tmp_manifest"
    return 1
  fi

  mv "$tmp_manifest" "$manifest_path"
}

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_pull_latest.sh [options]

Copy WSL experiment evidence back to the local repo under
remote_artifacts/wsl/<experiment-name>/.

Default files:
  train.log
  logs/bash_output.log
  TensorBoard event files under runs/
  run_status.json
  run_summary.json
  .hydra/config.yaml
  .hydra/overrides.yaml
  .hydra/hydra.yaml
  pre_run/
  diagnostics/
  checkpoints/final.json
  checkpoint_manifest.json with remote checkpoint paths, sizes, and hashes
  metrics.csv
  loss_curves.png
  reconstruction_grid.png
  pull_skipped_files.tsv when fallback transfer skips large files

Options:
  --run PATH     Pull this remote experiment path instead of the latest one.
  --checkpoint  Also copy heavyweight checkpoint .pt/.json files.
  --analyze     After pull, export TensorBoard scalar summaries and plots locally.

Environment:
  RL_LAB_GPU_SSH_HOST       SSH host alias. Default: wsl
  RL_LAB_GPU_REPO           Remote repo path. Default: /home/omkar/adarsh/rl-lab
  RL_LAB_GPU_ARTIFACTS_DIR  Local destination root. Default: remote_artifacts/wsl
  RL_LAB_GPU_FALLBACK_MAX_BYTES
                            Max bytes copied by chunked fallback. Default: 50000
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
  --include='/runs/'
  --include='/runs/**/'
  --include='/runs/**/events.out.tfevents*'
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
fi

include_args+=(--exclude='*')

if ! rsync -az --prune-empty-dirs -e "$RSYNC_SSH" \
  "${include_args[@]}" \
  "$GPU_SSH_HOST:$remote_source/" \
  "$dest/"; then
  pull_run_evidence_base64 "$remote_source" "$dest" "$INCLUDE_CHECKPOINT"
fi

mkdir -p "$dest/runs"
rsync -az --prune-empty-dirs -e "$RSYNC_SSH" \
  --include='*/' \
  --include='events.out.tfevents*' \
  --exclude='*' \
  "$GPU_SSH_HOST:$GPU_REMOTE_REPO/runs/$experiment_name/" \
  "$dest/runs/" 2>/dev/null || pull_external_runs_events_base64 "$dest"

if ! write_checkpoint_manifest "$remote_source" "$dest/checkpoint_manifest.json"; then
  printf '{"schema_version":1,"remote_run":%s,"entries":[],"error":"checkpoint manifest collection failed"}\n' \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$remote_source")" \
    > "$dest/checkpoint_manifest.json"
fi

if [ "$ANALYZE" = "1" ]; then
  python scripts/research/export_tensorboard_run.py "$dest" --out "$dest/analysis"
fi

echo "pulled latest remote experiment:"
echo "$latest"
echo "local copy:"
echo "$dest"

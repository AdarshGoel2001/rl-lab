#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

PATCH_PATH="${RL_LAB_GPU_PATCH_PATH:-/tmp/rl-lab-gpu-sync.patch}"
RESET_REMOTE=0
ALLOW_DIRTY=0
DRY_RUN=0
PATH_FILTERS=()
EXCLUDE_FILTERS=("remote_artifacts" "experiments" "runs" "datasets" ".agent_runs")

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_sync_patch.sh [options]

Create a patch from the local dirty worktree, copy it to the WSL GPU repo,
check it with git apply --check, then apply it.

Options:
  --reset-remote   Reset tracked remote code and clean untracked non-artifact files first.
  --allow-dirty    Allow applying onto a dirty remote tree; still requires git apply --check.
  --paths LIST     Space-separated path allowlist, for example: --paths "src scripts tests".
  --path PATH      Add one path to the allowlist. May be repeated.
  --exclude PATH   Exclude a path from the patch. May be repeated.
  --dry-run        Print patch summary without copying or applying.

Environment:
  RL_LAB_GPU_SSH_HOST    SSH host alias. Default: wsl
  RL_LAB_GPU_REPO        Remote repo path. Default: /home/omkar/adarsh/rl-lab
  RL_LAB_GPU_PATCH_PATH  Remote/local patch path. Default: /tmp/rl-lab-gpu-sync.patch
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --reset-remote)
      RESET_REMOTE=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --paths)
      read -r -a parsed_paths <<< "${2:?--paths requires a value}"
      PATH_FILTERS+=("${parsed_paths[@]}")
      shift 2
      ;;
    --path)
      PATH_FILTERS+=("${2:?--path requires a value}")
      shift 2
      ;;
    --exclude)
      EXCLUDE_FILTERS+=("${2:?--exclude requires a value}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

tmp_patch="${PATCH_PATH}"
: > "$tmp_patch"

if [ "${#PATH_FILTERS[@]}" -eq 0 ]; then
  PATH_FILTERS=(".")
fi

PATHSPECS=("${PATH_FILTERS[@]}")
for excluded in "${EXCLUDE_FILTERS[@]}"; do
  PATHSPECS+=(":(exclude)$excluded")
done

git diff --binary HEAD -- "${PATHSPECS[@]}" > "$tmp_patch"

while IFS= read -r -d '' path; do
  # git diff --no-index exits 1 when it successfully finds differences.
  git diff --binary --no-index -- /dev/null "$path" >> "$tmp_patch" || {
    diff_status=$?
    if [ "$diff_status" -ne 1 ]; then
      exit "$diff_status"
    fi
  }
done < <(git ls-files --others --exclude-standard -z -- "${PATHSPECS[@]}")

if [ ! -s "$tmp_patch" ]; then
  echo "no local tracked or untracked changes to sync"
  exit 0
fi

echo "patch file: $tmp_patch"
echo "patch bytes: $(wc -c < "$tmp_patch")"
echo "path filters: ${PATH_FILTERS[*]}"
echo "exclude filters: ${EXCLUDE_FILTERS[*]}"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

gpu_scp "$tmp_patch" "$GPU_SSH_HOST:$PATCH_PATH"

REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
PATCH_PATH_Q="$(gpu_quote "$PATCH_PATH")"

gpu_ssh "
  set -e
  cd $REMOTE_REPO_Q

  if [ '$RESET_REMOTE' = '1' ]; then
    git reset --hard
    git clean -fd -- . ':(exclude)experiments' ':(exclude)runs' ':(exclude)datasets' ':(exclude).agent_runs'
  elif [ '$ALLOW_DIRTY' != '1' ]; then
    repo_status=\$(git status --porcelain -- . ':(exclude)experiments' ':(exclude)runs' ':(exclude)datasets' ':(exclude).agent_runs')
    if [ -n \"\$repo_status\" ]; then
      echo 'remote repo has source/config/test/doc changes; rerun with --reset-remote if Mac is source of truth, or --allow-dirty for an independent patch' >&2
      echo \"\$repo_status\" >&2
      exit 1
    fi
  fi

  git apply --check $PATCH_PATH_Q
  git apply $PATCH_PATH_Q
  echo 'applied patch on remote: $GPU_REMOTE_REPO'
"

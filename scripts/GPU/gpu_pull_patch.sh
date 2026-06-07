#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

APPLY_PATCH=0
REMOTE_PATCH_PATH="${RL_LAB_GPU_PULL_PATCH_PATH:-/tmp/rl-lab-gpu-pull.patch}"
LOCAL_PATCH_PATH=""
PATH_FILTERS=()
EXCLUDE_FILTERS=("remote_artifacts" "experiments" "runs" "datasets")

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_pull_patch.sh [options]

Create a patch from the remote GPU repo dirty worktree and copy it back to the
local repo. By default it only checks whether the patch applies locally. Use
--apply when the remote code changes are intentionally coming back.

Options:
  --apply       Apply the pulled patch locally after git apply --check passes.
  --out PATH    Local patch path. Default: remote_code_patches/<timestamp>.patch
  --paths LIST  Space-separated path allowlist, for example: --paths "src scripts tests".
  --path PATH   Add one path to the allowlist. May be repeated.
  --exclude PATH
                Exclude a path from the patch. May be repeated.

Environment:
  RL_LAB_GPU_SSH_HOST          SSH host alias. Default: wsl
  RL_LAB_GPU_REPO              Remote repo path. Default: /home/omkar/adarsh/rl-lab
  RL_LAB_GPU_PULL_PATCH_PATH   Remote temp patch path. Default: /tmp/rl-lab-gpu-pull.patch
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply)
      APPLY_PATCH=1
      shift
      ;;
    --out)
      LOCAL_PATCH_PATH="${2:?--out requires a value}"
      shift 2
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

if [ "${#PATH_FILTERS[@]}" -eq 0 ]; then
  PATH_FILTERS=(".")
fi

PATHSPECS=("${PATH_FILTERS[@]}")
for excluded in "${EXCLUDE_FILTERS[@]}"; do
  PATHSPECS+=(":(exclude)$excluded")
done

REMOTE_PATHSPECS="$(gpu_join_quoted "${PATHSPECS[@]}")"
REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
REMOTE_PATCH_Q="$(gpu_quote "$REMOTE_PATCH_PATH")"

remote_result="$(
  gpu_ssh "
  set -e
  cd $REMOTE_REPO_Q
  : > $REMOTE_PATCH_Q
  git diff --binary HEAD -- $REMOTE_PATHSPECS > $REMOTE_PATCH_Q
  while IFS= read -r -d '' path; do
    git diff --binary --no-index -- /dev/null \"\$path\" >> $REMOTE_PATCH_Q || {
      status=\$?
      if [ \"\$status\" -ne 1 ]; then
        exit \"\$status\"
      fi
    }
  done < <(git ls-files --others --exclude-standard -z -- $REMOTE_PATHSPECS)
  if [ ! -s $REMOTE_PATCH_Q ]; then
    echo '__NO_REMOTE_PATCH__'
    exit 0
  fi
  wc -c $REMOTE_PATCH_Q
  "
)"

echo "$remote_result"
if [[ "$remote_result" == *"__NO_REMOTE_PATCH__"* ]]; then
  echo "no remote tracked or untracked code changes to pull"
  exit 0
fi

if [ -z "$LOCAL_PATCH_PATH" ]; then
  mkdir -p remote_code_patches
  LOCAL_PATCH_PATH="remote_code_patches/$(date +%Y%m%d_%H%M%S)_gpu_pull.patch"
fi

mkdir -p "$(dirname "$LOCAL_PATCH_PATH")"
gpu_scp "$GPU_SSH_HOST:$REMOTE_PATCH_PATH" "$LOCAL_PATCH_PATH"

git apply --check "$LOCAL_PATCH_PATH"

echo "pulled remote code patch:"
echo "$LOCAL_PATCH_PATH"
echo "path filters: ${PATH_FILTERS[*]}"
echo "exclude filters: ${EXCLUDE_FILTERS[*]}"

if [ "$APPLY_PATCH" = "1" ]; then
  git apply "$LOCAL_PATCH_PATH"
  echo "applied patch locally"
else
  echo "patch was checked but not applied; rerun with --apply to apply it"
fi

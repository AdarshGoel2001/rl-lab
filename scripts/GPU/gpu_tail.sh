#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

LINES="${1:-80}"
REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
LINES_Q="$(gpu_quote "$LINES")"

gpu_ssh "
  set -e
  cd $REMOTE_REPO_Q
  latest=\$(find experiments -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [ -z \"\$latest\" ]; then
    echo 'no experiments found'
    exit 1
  fi
  echo \"== \$latest ==\"
  if [ -f \"\$latest/train.log\" ]; then
    tail -n $LINES_Q \"\$latest/train.log\"
  elif [ -f \"\$latest/logs/bash_output.log\" ]; then
    tail -n $LINES_Q \"\$latest/logs/bash_output.log\"
  else
    echo 'no train.log or logs/bash_output.log found'
    find \"\$latest\" -maxdepth 3 -type f | sort
  fi
"

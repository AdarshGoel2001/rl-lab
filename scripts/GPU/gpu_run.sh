#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_common.sh"

SESSION="rl-lab-planet"
EXPERIMENT="planet_dmc_cartpole_swingup"
BUDGET="planet_tiny"
RUN_LOG="${RL_LAB_GPU_RUN_LOG:-}"
USE_VENV=1
DRY_RUN=0
HYDRA_OVERRIDES=()
POSITIONALS=()

usage() {
  cat <<'USAGE'
Usage: scripts/GPU/gpu_run.sh [options] [session experiment budget] [-- override...]

Start a training command inside a remote tmux session. The old positional form
still works:
  scripts/GPU/gpu_run.sh rl-lab-run planet_dmc_cartpole_swingup planet_tiny

Options:
  --session NAME       Remote tmux session. Default: rl-lab-planet
  --experiment NAME    Hydra experiment name. Default: planet_dmc_cartpole_swingup
  --budget NAME        Hydra budget name. Default: planet_tiny
  --override VALUE     Extra Hydra override. May be repeated.
  --log PATH           Remote tee log path. Also configurable with RL_LAB_GPU_RUN_LOG.
  --no-venv            Do not source .venv/bin/activate before running.
  --dry-run            Print the remote command without starting tmux.

Environment:
  RL_LAB_GPU_SSH_HOST                  SSH host alias. Default: wsl
  RL_LAB_GPU_REPO                      Remote repo path. Default: /home/omkar/adarsh/rl-lab
  RL_LAB_GPU_RUN_LOG                   Optional remote tee log path.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session)
      SESSION="${2:?--session requires a value}"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="${2:?--experiment requires a value}"
      shift 2
      ;;
    --budget)
      BUDGET="${2:?--budget requires a value}"
      shift 2
      ;;
    --override)
      HYDRA_OVERRIDES+=("${2:?--override requires a value}")
      shift 2
      ;;
    --log)
      RUN_LOG="${2:?--log requires a value}"
      shift 2
      ;;
    --no-venv)
      USE_VENV=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        HYDRA_OVERRIDES+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONALS+=("$1")
      shift
      ;;
  esac
done

if [ "${#POSITIONALS[@]}" -gt 0 ]; then
  SESSION="${POSITIONALS[0]}"
fi
if [ "${#POSITIONALS[@]}" -gt 1 ]; then
  EXPERIMENT="${POSITIONALS[1]}"
fi
if [ "${#POSITIONALS[@]}" -gt 2 ]; then
  BUDGET="${POSITIONALS[2]}"
fi
if [ "${#POSITIONALS[@]}" -gt 3 ]; then
  for extra in "${POSITIONALS[@]:3}"; do
    HYDRA_OVERRIDES+=("$extra")
  done
fi

TRAIN_ARGS=("scripts/train.py" "+experiment=$EXPERIMENT" "budget=$BUDGET" "${HYDRA_OVERRIDES[@]}")
TRAIN_CMD="python $(gpu_join_quoted "${TRAIN_ARGS[@]}")"
REMOTE_REPO_Q="$(gpu_quote "$GPU_REMOTE_REPO")"
SESSION_Q="$(gpu_quote "$SESSION")"
REMOTE_CMD="cd $REMOTE_REPO_Q"
if [ "$USE_VENV" = "1" ]; then
  REMOTE_CMD+=" && source .venv/bin/activate"
fi
REMOTE_CMD+=" && $TRAIN_CMD"

if [ -n "$RUN_LOG" ]; then
  RUN_LOG_Q="$(gpu_quote "$RUN_LOG")"
  RUN_LOG_DIR_Q="$(gpu_quote "$(dirname "$RUN_LOG")")"
  REMOTE_CMD="mkdir -p $RUN_LOG_DIR_Q && { $REMOTE_CMD; } 2>&1 | tee -a $RUN_LOG_Q"
fi

REMOTE_CMD_Q="$(gpu_quote "$REMOTE_CMD")"
RUN_LOG_JSON="${RUN_LOG:-}"
RUN_LOG_JSON_Q="$(gpu_quote "$RUN_LOG_JSON")"
EXPERIMENT_Q="$(gpu_quote "$EXPERIMENT")"
BUDGET_Q="$(gpu_quote "$BUDGET")"

if [ "$DRY_RUN" = "1" ]; then
  echo "$REMOTE_CMD"
  exit 0
fi

gpu_ssh "
  set -e
  cd $REMOTE_REPO_Q
  if tmux has-session -t $SESSION_Q 2>/dev/null; then
    echo 'tmux session already exists: $SESSION'
    echo 'Attach with: ssh $GPU_SSH_HOST -t tmux attach -t $SESSION'
    exit 0
  fi
  mkdir -p .agent_runs
  tmux new-session -d -s $SESSION_Q bash -lc $REMOTE_CMD_Q
  {
    printf '{\n'
    printf '  \"session\": \"%s\",\n' $SESSION_Q
    printf '  \"experiment\": \"%s\",\n' $EXPERIMENT_Q
    printf '  \"budget\": \"%s\",\n' $BUDGET_Q
    printf '  \"command\": \"%s\",\n' $REMOTE_CMD_Q
    printf '  \"log_path\": \"%s\",\n' $RUN_LOG_JSON_Q
    printf '  \"remote_repo\": \"%s\",\n' $REMOTE_REPO_Q
    printf '  \"started_at\": \"%s\"\n' \"\$(date -Iseconds)\"
    printf '}\n'
  } > \".agent_runs/$SESSION.json\"
  echo 'started tmux session: $SESSION'
  echo 'command: $REMOTE_CMD'
  echo 'registry: $GPU_REMOTE_REPO/.agent_runs/$SESSION.json'
  echo 'attach: ssh $GPU_SSH_HOST -t tmux attach -t $SESSION'
"

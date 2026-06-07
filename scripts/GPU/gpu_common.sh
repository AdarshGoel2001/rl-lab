#!/usr/bin/env bash

GPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SSH_HOST="${RL_LAB_GPU_SSH_HOST:-${AGENT_GPU_SSH_HOST:-wsl}}"
GPU_REMOTE_REPO="${RL_LAB_GPU_REPO:-${AGENT_GPU_REPO:-/home/omkar/adarsh/rl-lab}}"
GPU_CONNECT_TIMEOUT="${RL_LAB_GPU_CONNECT_TIMEOUT:-${AGENT_GPU_CONNECT_TIMEOUT:-10}}"
GPU_SERVER_ALIVE_INTERVAL="${RL_LAB_GPU_SERVER_ALIVE_INTERVAL:-${AGENT_GPU_SERVER_ALIVE_INTERVAL:-5}}"
GPU_SERVER_ALIVE_COUNT_MAX="${RL_LAB_GPU_SERVER_ALIVE_COUNT_MAX:-${AGENT_GPU_SERVER_ALIVE_COUNT_MAX:-1}}"

GPU_SSH_OPTS=(
  -o ClearAllForwardings=yes
  -o ConnectTimeout="$GPU_CONNECT_TIMEOUT"
  -o ServerAliveInterval="$GPU_SERVER_ALIVE_INTERVAL"
  -o ServerAliveCountMax="$GPU_SERVER_ALIVE_COUNT_MAX"
)

gpu_ssh() {
  ssh -T "${GPU_SSH_OPTS[@]}" "$GPU_SSH_HOST" "$@"
}

gpu_scp() {
  scp -q "${GPU_SSH_OPTS[@]}" "$@"
}

gpu_rsync_ssh() {
  printf 'ssh -o ClearAllForwardings=yes -o ConnectTimeout=%q -o ServerAliveInterval=%q -o ServerAliveCountMax=%q' \
    "$GPU_CONNECT_TIMEOUT" \
    "$GPU_SERVER_ALIVE_INTERVAL" \
    "$GPU_SERVER_ALIVE_COUNT_MAX"
}

gpu_quote() {
  printf '%q' "$1"
}

gpu_join_quoted() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
}

gpu_repo_root() {
  git rev-parse --show-toplevel
}

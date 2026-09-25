#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${LLM_PORT:-18080}"
info_dir="${HOME}/.cache/oscar-llm"
session_dir="${info_dir}/active"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this script on a compute node inside an interact allocation." >&2
  exit 1
fi
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
  echo "LLM_PORT must be between 1 and 65535." >&2
  exit 1
fi
port=$((10#$port))
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the project venv or a conda environment before starting the service." >&2
  exit 1
fi
for command_name in python openssl hostname; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing dependency: ${command_name}. Install it before continuing." >&2
    exit 1
  fi
done
# This wrapper owns networking and authentication; other launcher options pass through.
model_selected=false
for argument in "$@"; do
  case "$argument" in
    --qwen38|--gemma|--config|--config=*) model_selected=true ;;
    --|--host|--host=*|--port|--port=*|--api-key*|-k)
      echo "Networking/authentication are managed here; use LLM_PORT for the port. Native -- arguments are not accepted." >&2
      exit 1
      ;;
  esac
done

node_host="$(hostname -s)"
node_ip="$(hostname -i | awk '{print $1}')"
if [[ ! "$node_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ || "$node_ip" == 127.* ]]; then
  echo "Expected a compute-node internal IPv4 address from hostname -i; got: ${node_ip}" >&2
  exit 1
fi

umask 077
mkdir -p "$info_dir"
if ! mkdir "$session_dir"; then
  echo "An LLM session is already registered at ${session_dir}." >&2
  echo "If its job has ended, remove that stale directory manually before restarting." >&2
  exit 1
fi

server_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]]; then
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -f "${session_dir}/current.env" "${session_dir}/current.env.tmp" "${session_dir}/api-key"
  rmdir "$session_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

api_key="$(openssl rand -hex 32)"
printf '%s\n' "$api_key" > "${session_dir}/api-key"
cd "$repo_root"
# Without a model selector, use the L40S Qwen3.8 serving preset.
if [[ "$model_selected" == false ]]; then set -- --qwen38 "$@"; fi
model_key="$(python -m llm_bench.serve "$@" --print-model-key)"
python -m llm_bench.serve "$@" --host "$node_ip" --port "$port" \
  -- --api-key-file "${session_dir}/api-key" &
server_pid=$!

{
  printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'NODE_HOST=%s\n' "$node_host"
  printf 'NODE_IP=%s\n' "$node_ip"
  printf 'REMOTE_PORT=%s\n' "$port"
  printf 'API_KEY=%s\n' "$api_key"
  printf 'MODEL_KEY=%s\n' "$model_key"
} > "${session_dir}/current.env.tmp"
mv "${session_dir}/current.env.tmp" "${session_dir}/current.env"

cat <<EOF
LLM service starting (wait for llama-server's model-ready log):
  Slurm job:   ${SLURM_JOB_ID}
  Node:        ${node_host}
  Listen:      ${node_ip}:${port}
  Model:       ${model_key}
  Credentials: ${session_dir}/current.env (private)

On your local Mac, run from GPU-Insights:
  bash scripts/oscar-llm-connect.sh
Keep this compute-node process running. Ctrl+C stops the service and removes its connection information.
EOF
wait "$server_pid"

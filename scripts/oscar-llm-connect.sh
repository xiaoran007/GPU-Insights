#!/usr/bin/env bash
set -euo pipefail

oscar_host="${OSCAR_HOST:-oscar}"
local_port="${LOCAL_PORT:-18080}"
if [[ ! "$local_port" =~ ^[0-9]{1,5}$ ]] || ((10#$local_port < 1 || 10#$local_port > 65535)); then
  echo "LOCAL_PORT must be between 1 and 65535." >&2
  exit 1
fi
local_port=$((10#$local_port))

umask 077
info_file="$(mktemp -t oscar-llm.XXXXXX)"
trap 'rm -f "$info_file"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! ssh "$oscar_host" 'cat "$HOME/.cache/oscar-llm/active/current.env"' > "$info_file"; then
  echo "Could not read the LLM session. Check the outer SSH tunnel and start scripts/llm-interact.sh in your GPU allocation." >&2
  exit 1
fi

# Read data, never execute the remote connection-information file as shell code.
job_id="" node_host="" node_ip="" remote_port="" api_key="" model_key=""
while IFS='=' read -r key value; do
  case "$key" in
    JOB_ID) job_id="$value" ;;
    NODE_HOST) node_host="$value" ;;
    NODE_IP) node_ip="$value" ;;
    REMOTE_PORT) remote_port="$value" ;;
    API_KEY) api_key="$value" ;;
    MODEL_KEY) model_key="$value" ;;
    *) echo "Unexpected connection-information field: ${key}" >&2; exit 1 ;;
  esac
done < "$info_file"
if [[ ! "$model_key" =~ ^[a-zA-Z0-9][a-zA-Z0-9_./:-]*$ ]]; then
  echo "Missing or invalid MODEL_KEY. Restart the compute-node service with the updated llm-interact.sh." >&2
  exit 1
fi
if [[ ! "$job_id" =~ ^[0-9]+$ || ! "$node_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ || ! "$remote_port" =~ ^[0-9]{1,5}$ || ! "$api_key" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid LLM connection information." >&2
  exit 1
fi
if ((10#$remote_port < 1 || 10#$remote_port > 65535)); then
  echo "Invalid remote port." >&2
  exit 1
fi
job_state="$(ssh "$oscar_host" "squeue -h -j '${job_id}' -o '%T'")"
if [[ "$job_state" != RUNNING ]]; then
  echo "Slurm job ${job_id} is not running (${job_state:-not found}); session information may be stale." >&2
  exit 1
fi

cat <<EOF
Opening the LLM tunnel via ${oscar_host}:
  Job:      ${job_id}
  Node:     ${node_host} (${node_ip}:${remote_port})
  Base URL: http://127.0.0.1:${local_port}/v1
  API key:  ${api_key}
  Model:    ${model_key}

Set the base URL and API key in your agent client.
SSH forwarding does not imply model readiness; wait for the compute-node server's ready log.
Keep this terminal open. Ctrl+C closes only the tunnel, leaving the model running.

Claude Code: copy this command into another terminal in your project directory.
Settings apply only to this invocation and use the server's configured model name.

claude --model '${model_key}' --settings '{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:${local_port}","ANTHROPIC_AUTH_TOKEN":"${api_key}","ANTHROPIC_DEFAULT_HAIKU_MODEL":"${model_key}","ANTHROPIC_DEFAULT_SONNET_MODEL":"${model_key}","ANTHROPIC_DEFAULT_OPUS_MODEL":"${model_key}"}}'
EOF
rm -f "$info_file"
trap - EXIT INT TERM
exec ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${local_port}:${node_ip}:${remote_port}" \
  "$oscar_host"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${repo_root}/models"
llm_dir="${models_dir}/llm"
min_free_gib="${GPU_INSIGHTS_VAST_MIN_FREE_GIB:-40}"

missing_commands=()
for command_name in python3 nvidia-smi; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    missing_commands+=("${command_name}")
  fi
done

if ((${#missing_commands[@]} > 0)); then
  echo "Missing required commands for a Vast LLM benchmark run:" >&2
  printf '  %s\n' "${missing_commands[@]}" >&2
  echo "Use a CUDA base image that includes the NVIDIA runtime tools, then install the missing OS packages manually." >&2
  exit 1
fi

mkdir -p "${llm_dir}"

available_kib="$(df -Pk "${llm_dir}" | awk 'NR == 2 {print $4}')"
required_kib=$((min_free_gib * 1024 * 1024))

if [[ -z "${available_kib}" ]]; then
  echo "Unable to inspect free disk space for ${llm_dir}." >&2
  exit 1
fi

if ((available_kib < required_kib)); then
  echo "Not enough free disk space for the fixed GGUF download:" >&2
  echo "  path:      ${llm_dir}" >&2
  echo "  available: $((available_kib / 1024 / 1024)) GiB" >&2
  echo "  required:  ${min_free_gib} GiB" >&2
  echo "Create the Vast instance with a larger disk before downloading the model." >&2
  exit 1
fi

echo "Vast LLM benchmark workspace ready:"
echo "  repo:      ${repo_root}"
echo "  model dir: ${llm_dir}"
echo "  free disk: $((available_kib / 1024 / 1024)) GiB"
echo
echo "Detected NVIDIA devices:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo
echo "Next steps:"
echo "  python3 scripts/download-llm-model.py"
echo "  bash scripts/bootstrap-llama-cpp.sh --backend cuda"
echo "  python3 -m llm_bench.cli"

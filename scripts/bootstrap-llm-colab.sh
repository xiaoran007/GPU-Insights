#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
drive_root="${GPU_INSIGHTS_COLAB_DRIVE_ROOT:-/content/drive/MyDrive}"
persistent_dir="${GPU_INSIGHTS_COLAB_LLM_DIR:-${drive_root}/GPU-Insights/llm}"
local_dir="${GPU_INSIGHTS_COLAB_LOCAL_LLM_DIR:-/content/gpu-insights-llm}"
model_filename="${GPU_INSIGHTS_COLAB_LLM_FILENAME:-Qwen3.6-27B-Q4_K_M.gguf}"
expected_bytes="${GPU_INSIGHTS_COLAB_EXPECTED_BYTES:-16817244384}"
models_dir="${repo_root}/models"
link_path="${models_dir}/llm"
persistent_model="${persistent_dir}/${model_filename}"
local_model="${local_dir}/${model_filename}"

file_size_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

copy_model() {
  if command -v rsync >/dev/null 2>&1; then
    rsync -ah --info=progress2 "${persistent_model}" "${local_model}"
  else
    cp -v "${persistent_model}" "${local_model}"
  fi
}

if [[ ! -d "${drive_root}" ]]; then
  cat >&2 <<EOF
Google Drive does not appear to be mounted at:
  ${drive_root}

Mount Drive in a Colab notebook first:
  from google.colab import drive
  drive.mount('/content/drive')
EOF
  exit 1
fi

if [[ ! -f "${persistent_model}" ]]; then
  cat >&2 <<EOF
Required GGUF was not found in Google Drive:
  ${persistent_model}

Place the fixed model file there first, then re-run this bootstrap.
Expected filename:
  ${model_filename}
EOF
  exit 1
fi

actual_bytes="$(file_size_bytes "${persistent_model}")"
if [[ -n "${expected_bytes}" && "${actual_bytes}" != "${expected_bytes}" ]]; then
  cat >&2 <<EOF
Google Drive GGUF size mismatch:
  ${persistent_model}
  got:      ${actual_bytes} bytes
  expected: ${expected_bytes} bytes

Refusing to copy a potentially incomplete or different model file.
EOF
  exit 1
fi

mkdir -p "${local_dir}"

if [[ -f "${local_model}" ]]; then
  local_bytes="$(file_size_bytes "${local_model}")"
  if [[ "${local_bytes}" == "${actual_bytes}" ]]; then
    echo "Local Colab model cache already matches Google Drive:"
    echo "  ${local_model}"
  else
    echo "Local Colab model cache exists but size differs; refreshing:"
    echo "  ${local_model}"
    copy_model
  fi
else
  echo "Copying GGUF from Google Drive to local Colab storage:"
  echo "  from: ${persistent_model}"
  echo "  to:   ${local_model}"
  copy_model
fi

mkdir -p "${models_dir}"

if [[ -L "${link_path}" ]]; then
  current_target="$(readlink "${link_path}")"
  if [[ "${current_target}" == "${local_dir}" ]]; then
    echo "LLM model symlink already configured:"
    echo "  ${link_path} -> ${local_dir}"
  else
    echo "Existing LLM symlink points somewhere else:" >&2
    echo "  ${link_path} -> ${current_target}" >&2
    echo "Remove or replace it manually before re-running this bootstrap." >&2
    exit 1
  fi
elif [[ -e "${link_path}" ]]; then
  echo "Path already exists and is not a symlink:" >&2
  echo "  ${link_path}" >&2
  echo "Move existing files aside, then replace ${link_path} with a symlink." >&2
  exit 1
else
  ln -s "${local_dir}" "${link_path}"
  echo "Configured Colab local LLM model cache:"
  echo "  ${link_path} -> ${local_dir}"
fi

echo
echo "Colab LLM model bootstrap complete:"
echo "  persistent: ${persistent_model}"
echo "  local:      ${local_model}"
echo
echo "Next step:"
echo "  python3 -m llm_bench.cli"

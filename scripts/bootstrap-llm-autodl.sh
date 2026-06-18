#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
persistent_dir="${GPU_INSIGHTS_AUTODL_LLM_DIR:-/root/autodl-fs/llm}"
models_dir="${repo_root}/models"
link_path="${models_dir}/llm"

mkdir -p "${persistent_dir}"
mkdir -p "${models_dir}"

if [[ -L "${link_path}" ]]; then
  current_target="$(readlink "${link_path}")"
  if [[ "${current_target}" == "${persistent_dir}" ]]; then
    echo "LLM model symlink already configured:"
    echo "  ${link_path} -> ${persistent_dir}"
    exit 0
  fi

  echo "Existing LLM symlink points somewhere else:"
  echo "  ${link_path} -> ${current_target}"
  echo "Remove or replace it manually before re-running this bootstrap."
  exit 1
fi

if [[ -e "${link_path}" ]]; then
  echo "Path already exists and is not a symlink:"
  echo "  ${link_path}"
  echo "Move existing files into ${persistent_dir}, then replace ${link_path} with a symlink."
  exit 1
fi

ln -s "${persistent_dir}" "${link_path}"

echo "Configured AutoDL persistent LLM model storage:"
echo "  ${link_path} -> ${persistent_dir}"
echo
echo "Next step:"
echo "  python3 scripts/download-llm-model.py"

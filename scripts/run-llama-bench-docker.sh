#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prebuilt_root="${GPU_INSIGHTS_LLAMA_BENCH_DOCKER_ROOT:-${repo_root}/third_party/llama-bench/current}"
llama_bench="${GPU_INSIGHTS_LLAMA_BENCH_DOCKER_BIN:-${prebuilt_root}/bin/llama-bench}"
manifest="${prebuilt_root}/BUILD-MANIFEST.json"
docker_gpus="${GPU_INSIGHTS_LLAMA_BENCH_DOCKER_GPUS:-all}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run-llama-bench-docker.sh [--check]
  scripts/run-llama-bench-docker.sh <llama-bench args...>

Environment overrides:
  GPU_INSIGHTS_LLAMA_BENCH_DOCKER_IMAGE   Docker image to run. Defaults from BUILD-MANIFEST cudaMajor.
  GPU_INSIGHTS_LLAMA_BENCH_DOCKER_GPUS    Docker --gpus value. Default: all.
  GPU_INSIGHTS_LLAMA_BENCH_DOCKER_ROOT    Prebuilt install root. Default: third_party/llama-bench/current.
  GPU_INSIGHTS_LLAMA_BENCH_DOCKER_BIN     llama-bench wrapper path inside the mounted repo.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is required for --docker LLM benchmark runs."
  docker info >/dev/null 2>&1 || die "docker daemon is not reachable."
}

require_linux_host() {
  [[ "$(uname -s)" == "Linux" ]] || die "--docker LLM benchmark runs require a Linux or WSL2 host with NVIDIA container support."
}

json_string_value() {
  local key="$1"
  local file="$2"
  sed -n "s/^[[:space:]]*\"${key}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" "${file}" | head -n 1
}

cuda_major() {
  [[ -f "${manifest}" ]] || die "prebuilt BUILD-MANIFEST.json was not found at ${manifest}."
  json_string_value "cudaMajor" "${manifest}"
}

docker_image() {
  if [[ -n "${GPU_INSIGHTS_LLAMA_BENCH_DOCKER_IMAGE:-}" ]]; then
    printf '%s\n' "${GPU_INSIGHTS_LLAMA_BENCH_DOCKER_IMAGE}"
    return
  fi

  local major
  major="$(cuda_major)"
  case "${major}" in
    12)
      printf '%s\n' "nvidia/cuda:12.6.3-runtime-ubuntu22.04"
      ;;
    13)
      printf '%s\n' "nvidia/cuda:13.0.2-runtime-ubuntu24.04"
      ;;
    *)
      die "unsupported CUDA major in ${manifest}: ${major:-<empty>}. Set GPU_INSIGHTS_LLAMA_BENCH_DOCKER_IMAGE explicitly."
      ;;
  esac
}

require_prebuilt() {
  [[ -x "${llama_bench}" ]] || die "prebuilt llama-bench wrapper was not found or is not executable: ${llama_bench}"
  [[ -x "${prebuilt_root}/bin/llama-bench.bin" ]] || die "prebuilt llama-bench binary was not found or is not executable: ${prebuilt_root}/bin/llama-bench.bin"
}

ensure_image() {
  local image="$1"
  if docker image inspect "${image}" >/dev/null 2>&1; then
    return
  fi
  echo "Docker image not found locally; pulling ${image} ..." >&2
  docker pull "${image}" >&2
}

container_smoke_check() {
  local image="$1"
  docker run --rm \
    --gpus "${docker_gpus}" \
    --entrypoint "" \
    -e "GPU_INSIGHTS_CONTAINER_LLAMA_BENCH=${llama_bench}" \
    -v "${repo_root}:${repo_root}:ro" \
    -w "${repo_root}" \
    "${image}" \
    /bin/sh -lc 'test -x "${GPU_INSIGHTS_CONTAINER_LLAMA_BENCH}" && test -d /usr/local/cuda/lib64'
}

check_environment() {
  require_linux_host
  require_docker
  require_prebuilt
  local image
  image="$(docker_image)"
  echo "Docker llama-bench environment:"
  echo "  image:       ${image}"
  echo "  gpus:        ${docker_gpus}"
  echo "  llama-bench: ${llama_bench}"
  ensure_image "${image}"
  container_smoke_check "${image}" || die "docker GPU smoke check failed. Verify NVIDIA Container Toolkit / WSL GPU integration."
}

model_mount_args() {
  local previous=""
  local value
  local model_dir
  local real_model_path
  local real_model_dir
  for value in "$@"; do
    if [[ "${previous}" == "-m" ]]; then
      if [[ "${value}" == /* ]]; then
        model_dir="$(cd "$(dirname "${value}")" && pwd)"
        if [[ "${model_dir}" != "${repo_root}" && "${model_dir}" != "${repo_root}/"* ]]; then
          printf '%s\n' "-v"
          printf '%s\n' "${model_dir}:${model_dir}:ro"
        fi

        real_model_path="$(realpath "${value}" 2>/dev/null || true)"
        if [[ "${real_model_path}" == /* ]]; then
          real_model_dir="$(dirname "${real_model_path}")"
          if [[ "${real_model_dir}" != "${model_dir}" && "${real_model_dir}" != "${repo_root}" && "${real_model_dir}" != "${repo_root}/"* ]]; then
            printf '%s\n' "-v"
            printf '%s\n' "${real_model_dir}:${real_model_dir}:ro"
          fi
        fi
      fi
      return
    fi
    previous="${value}"
  done
}

run_llama_bench() {
  require_linux_host
  require_docker
  require_prebuilt
  local image
  image="$(docker_image)"
  ensure_image "${image}"

  local docker_args=(
    run
    --rm
    --gpus "${docker_gpus}"
    --entrypoint ""
    -v "${repo_root}:${repo_root}:ro"
    -w "${repo_root}"
  )

  while IFS= read -r mount_arg; do
    docker_args+=("${mount_arg}")
  done < <(model_mount_args "$@")

  docker "${docker_args[@]}" "${image}" "${llama_bench}" "$@"
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --check)
    check_environment
    exit 0
    ;;
  *)
    run_llama_bench "$@"
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_repo="${GPU_INSIGHTS_LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
llama_ref="${GPU_INSIGHTS_LLAMA_CPP_REF:-origin/HEAD}"
out_dir="${GPU_INSIGHTS_LLAMA_BENCH_RELEASE_DIR:-dist/llama-bench}"
jobs="${GPU_INSIGHTS_LLAMA_CPP_JOBS:-}"
variant="all"
work_volume="${GPU_INSIGHTS_LLAMA_BENCH_WORK_VOLUME:-gpu-insights-llama-bench-work}"
clean_work=0
host_uid="$(id -u)"
host_gid="$(id -g)"

cuda12_image="${GPU_INSIGHTS_LLAMA_BENCH_CUDA12_IMAGE:-nvidia/cuda:12.6.3-devel-ubuntu22.04}"
cuda13_image="${GPU_INSIGHTS_LLAMA_BENCH_CUDA13_IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04}"
cuda12_architectures="${GPU_INSIGHTS_LLAMA_BENCH_CUDA12_ARCHS:-80;86;87;89;90}"
cuda13_architectures="${GPU_INSIGHTS_LLAMA_BENCH_CUDA13_ARCHS:-80;86;87;88;89;90;100;103;110;120;121}"
cuda_stub_dir="/usr/local/cuda/lib64/stubs"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/build-llama-bench-release.sh [--variant <all|cuda12|cuda13>] [--ref <git-ref>]

Options:
  --variant <name>        Build one variant or all. Default: all.
  --ref <git-ref>         llama.cpp tag, branch, or commit. Default: origin/HEAD.
  --out-dir <path>        Output directory for release tarballs. Default: dist/llama-bench.
  --jobs <n>              Parallel build jobs passed to CMake.
  --work-volume <name>    Docker volume mounted at /work for checkout/build cache.
  --clean-work            Clear the selected variant work directory before building.
  --cuda12-image <image>  Docker image for CUDA 12 builds.
  --cuda13-image <image>  Docker image for CUDA 13 builds.
  -h, --help              Show this help text.

Environment overrides:
  GPU_INSIGHTS_LLAMA_CPP_REPO
  GPU_INSIGHTS_LLAMA_CPP_REF
  GPU_INSIGHTS_LLAMA_CPP_JOBS
  GPU_INSIGHTS_LLAMA_BENCH_RELEASE_DIR
  GPU_INSIGHTS_LLAMA_BENCH_WORK_VOLUME
  GPU_INSIGHTS_LLAMA_BENCH_CUDA12_IMAGE
  GPU_INSIGHTS_LLAMA_BENCH_CUDA13_IMAGE
  GPU_INSIGHTS_LLAMA_BENCH_CUDA12_ARCHS
  GPU_INSIGHTS_LLAMA_BENCH_CUDA13_ARCHS
USAGE
}

require_option_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "Missing value for ${option}."
    usage
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      require_option_value "$1" "${2:-}"
      variant="$2"
      shift 2
      ;;
    --ref)
      require_option_value "$1" "${2:-}"
      llama_ref="$2"
      shift 2
      ;;
    --out-dir)
      require_option_value "$1" "${2:-}"
      out_dir="$2"
      shift 2
      ;;
    --jobs)
      require_option_value "$1" "${2:-}"
      jobs="$2"
      shift 2
      ;;
    --work-volume)
      require_option_value "$1" "${2:-}"
      work_volume="$2"
      shift 2
      ;;
    --clean-work)
      clean_work=1
      shift
      ;;
    --cuda12-image)
      require_option_value "$1" "${2:-}"
      cuda12_image="$2"
      shift 2
      ;;
    --cuda13-image)
      require_option_value "$1" "${2:-}"
      cuda13_image="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "${variant}" in
  all|cuda12|cuda13) ;;
  *)
    echo "Unsupported variant: ${variant}"
    echo "Expected one of: all, cuda12, cuda13"
    exit 1
    ;;
esac

case "${out_dir}" in
  /*)
    echo "--out-dir must be relative to the repository root so Docker can write it through /workspace." >&2
    exit 1
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build Linux amd64 release assets." >&2
  exit 1
fi

build_variant() {
  local name="$1"
  local cuda_major="$2"
  local image="$3"
  local architectures="$4"
  local builder_image="gpu-insights-llama-bench-builder:${name}"
  local container_work="/work/${name}"
  local cmake_flags="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${architectures} -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,${cuda_stub_dir} -DCMAKE_BUILD_TYPE=Release"

  echo
  echo "Building llama-bench release variant:"
  echo "  variant:        ${name}"
  echo "  docker image:   ${image}"
  echo "  llama.cpp ref:  ${llama_ref}"
  echo "  architectures: ${architectures}"
  echo "  work volume:   ${work_volume}"
  echo "  output:         ${out_dir}"
  if [[ "${clean_work}" -eq 1 ]]; then
    echo "  clean work:     yes"
  fi
  echo

  docker build \
    --platform linux/amd64 \
    --build-arg "CUDA_IMAGE=${image}" \
    -f "${repo_root}/docker/llama-bench-release.Dockerfile" \
    -t "${builder_image}" \
    "${repo_root}"

  docker run --rm \
    --platform linux/amd64 \
    -e "LLAMA_REPO=${llama_repo}" \
    -e "LLAMA_REF=${llama_ref}" \
    -e "CUDA_MAJOR=${cuda_major}" \
    -e "CUDA_ARCHITECTURES=${architectures}" \
    -e "CMAKE_FLAGS=${cmake_flags}" \
    -e "JOBS=${jobs}" \
    -e "CLEAN_WORK=${clean_work}" \
    -e "HOST_UID=${host_uid}" \
    -e "HOST_GID=${host_gid}" \
    -v "${work_volume}:/work" \
    -v "${repo_root}:/workspace" \
    -w /workspace \
    "${builder_image}" \
    bash -lc "$(container_build_script "${container_work}")"
}

container_build_script() {
  local container_work="$1"
  cat <<EOF
set -euo pipefail
if [[ "\${CLEAN_WORK}" == "1" ]]; then
  rm -rf "${container_work}"
fi
mkdir -p "${container_work}"
if [[ ! -d "${container_work}/llama.cpp/.git" ]]; then
  git clone "\${LLAMA_REPO}" "${container_work}/llama.cpp"
fi
if [[ -n "\$(git -C "${container_work}/llama.cpp" status --porcelain)" ]]; then
  echo "Cached llama.cpp checkout has local changes:" >&2
  echo "  ${container_work}/llama.cpp" >&2
  echo "Re-run with --clean-work or inspect the Docker volume ${work_volume}." >&2
  exit 1
fi
git -C "${container_work}/llama.cpp" fetch --tags origin
git -C "${container_work}/llama.cpp" remote set-head origin -a >/dev/null 2>&1 || true
git -C "${container_work}/llama.cpp" checkout "\${LLAMA_REF}"
cmake -S "${container_work}/llama.cpp" -B "${container_work}/build" -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  "-DCMAKE_CUDA_ARCHITECTURES=\${CUDA_ARCHITECTURES}" \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,${cuda_stub_dir}" \
  -DCMAKE_BUILD_TYPE=Release
parallel_jobs="\${JOBS:-}"
if [[ -z "\${parallel_jobs}" ]]; then
  parallel_jobs="\$(nproc)"
fi
cmake --build "${container_work}/build" --target llama-bench --parallel "\${parallel_jobs}"
bash /workspace/scripts/package-llama-bench-release.sh \
  --src-dir "${container_work}/llama.cpp" \
  --build-dir "${container_work}/build" \
  --out-dir "/workspace/${out_dir}" \
  --platform linux-amd64 \
  --backend cuda \
  --cuda-major "\${CUDA_MAJOR}" \
  --cuda-architectures "\${CUDA_ARCHITECTURES}" \
  --cmake-flags "\${CMAKE_FLAGS}"
chown -R "\${HOST_UID}:\${HOST_GID}" "/workspace/${out_dir}"
EOF
}

mkdir -p "${out_dir}"

if [[ "${variant}" == "all" || "${variant}" == "cuda12" ]]; then
  build_variant "linux-amd64-cuda12" "12" "${cuda12_image}" "${cuda12_architectures}"
fi

if [[ "${variant}" == "all" || "${variant}" == "cuda13" ]]; then
  build_variant "linux-amd64-cuda13" "13" "${cuda13_image}" "${cuda13_architectures}"
fi

echo
echo "Release assets written to:"
echo "  ${out_dir}"

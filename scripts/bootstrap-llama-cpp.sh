#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

repo_url="${GPU_INSIGHTS_LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
src_dir="${GPU_INSIGHTS_LLAMA_CPP_DIR:-${repo_root}/third_party/llama.cpp}"
build_dir="${GPU_INSIGHTS_LLAMA_CPP_BUILD_DIR:-${src_dir}/build}"
llama_ref="${GPU_INSIGHTS_LLAMA_CPP_REF:-}"
backend="${GPU_INSIGHTS_LLAMA_CPP_BACKEND:-}"
jobs="${GPU_INSIGHTS_LLAMA_CPP_JOBS:-}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/bootstrap-llama-cpp.sh --ref <tag-or-commit> [--backend <backend>]

Options:
  --ref <git-ref>       llama.cpp tag, branch, or commit to check out.
  --backend <backend>   One of: cpu, cuda, hip, vulkan, sycl, metal.
  --dir <path>          llama.cpp checkout directory.
  --build-dir <path>    CMake build directory.
  --jobs <n>            Parallel build jobs passed to CMake.
  -h, --help            Show this help text.

Environment overrides:
  GPU_INSIGHTS_LLAMA_CPP_REF
  GPU_INSIGHTS_LLAMA_CPP_BACKEND
  GPU_INSIGHTS_LLAMA_CPP_DIR
  GPU_INSIGHTS_LLAMA_CPP_BUILD_DIR
  GPU_INSIGHTS_LLAMA_CPP_JOBS
  GPU_INSIGHTS_LLAMA_CPP_REPO
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
    --ref)
      require_option_value "$1" "${2:-}"
      llama_ref="$2"
      shift 2
      ;;
    --backend)
      require_option_value "$1" "${2:-}"
      backend="$2"
      shift 2
      ;;
    --dir)
      require_option_value "$1" "${2:-}"
      src_dir="$2"
      shift 2
      ;;
    --build-dir)
      require_option_value "$1" "${2:-}"
      build_dir="$2"
      shift 2
      ;;
    --jobs)
      require_option_value "$1" "${2:-}"
      jobs="$2"
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

prompt_ref() {
  if [[ -n "${llama_ref}" ]]; then
    return
  fi

  if [[ ! -t 0 ]]; then
    echo "Missing llama.cpp git ref. Pass --ref <tag-or-commit> or set GPU_INSIGHTS_LLAMA_CPP_REF."
    exit 1
  fi

  read -r -p "llama.cpp git ref to checkout (tag/commit/branch): " llama_ref
  if [[ -z "${llama_ref}" ]]; then
    echo "A llama.cpp git ref is required for reproducible benchmark builds."
    exit 1
  fi
}

prompt_backend() {
  if [[ -n "${backend}" ]]; then
    return
  fi

  if [[ ! -t 0 ]]; then
    echo "Missing backend. Pass --backend <cpu|cuda|hip|vulkan|sycl|metal> or set GPU_INSIGHTS_LLAMA_CPP_BACKEND."
    exit 1
  fi

  echo "Select llama.cpp backend:"
  echo "  1) cpu"
  echo "  2) cuda    (NVIDIA CUDA)"
  echo "  3) hip     (AMD ROCm/HIP)"
  echo "  4) vulkan  (cross-vendor Vulkan)"
  echo "  5) sycl    (Intel GPU / oneAPI)"
  echo "  6) metal   (macOS Metal)"
  read -r -p "Backend [1-6]: " choice

  case "${choice}" in
    1|cpu) backend="cpu" ;;
    2|cuda) backend="cuda" ;;
    3|hip) backend="hip" ;;
    4|vulkan) backend="vulkan" ;;
    5|sycl) backend="sycl" ;;
    6|metal) backend="metal" ;;
    *)
      echo "Unsupported backend selection: ${choice}"
      exit 1
      ;;
  esac
}

normalize_backend() {
  backend="$(printf '%s' "${backend}" | tr '[:upper:]' '[:lower:]')"
  case "${backend}" in
    cpu|cuda|hip|vulkan|sycl|metal) ;;
    *)
      echo "Unsupported backend: ${backend}"
      echo "Expected one of: cpu, cuda, hip, vulkan, sycl, metal"
      exit 1
      ;;
  esac
}

missing=()
warnings=()

require_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    missing+=("${cmd} (${hint})")
  fi
}

require_cxx_toolchain() {
  if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    missing+=("C compiler (install gcc or clang)")
  fi

  if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    missing+=("C++ compiler (install g++ or clang++)")
  fi
}

preflight_dependencies() {
  require_cmd git "required to clone and check out llama.cpp"
  require_cmd cmake "required to configure and build llama.cpp"

  case "${backend}" in
    cpu)
      require_cxx_toolchain
      ;;
    cuda)
      require_cxx_toolchain
      require_cmd nvcc "install the NVIDIA CUDA toolkit and ensure nvcc is on PATH"
      ;;
    hip)
      require_cxx_toolchain
      require_cmd hipconfig "install ROCm/HIP and ensure hipconfig is on PATH"
      if command -v hipconfig >/dev/null 2>&1; then
        local hip_clang
        hip_clang="$(hipconfig -l 2>/dev/null || true)/clang"
        if [[ ! -x "${hip_clang}" ]]; then
          missing+=("${hip_clang} (ROCm clang reported by hipconfig -l)")
        fi
      fi
      ;;
    vulkan)
      require_cxx_toolchain
      require_cmd glslc "install Vulkan SDK or distro Vulkan shader compiler package"
      if ! command -v vulkaninfo >/dev/null 2>&1; then
        warnings+=("vulkaninfo not found; install Vulkan SDK/tools if CMake cannot locate Vulkan")
      fi
      ;;
    sycl)
      require_cmd icx "source oneAPI setvars.sh or install Intel oneAPI compiler"
      require_cmd icpx "source oneAPI setvars.sh or install Intel oneAPI compiler"
      if ! command -v sycl-ls >/dev/null 2>&1; then
        warnings+=("sycl-ls not found; source /opt/intel/oneapi/setvars.sh before building Intel GPU backend")
      fi
      ;;
    metal)
      if [[ "$(uname -s)" != "Darwin" ]]; then
        missing+=("macOS host (Metal backend only builds on macOS)")
      fi
      require_cmd xcrun "install Xcode Command Line Tools"
      ;;
  esac

  if ((${#warnings[@]} > 0)); then
    echo "Dependency warnings:"
    printf '  - %s\n' "${warnings[@]}"
    echo
  fi

  if ((${#missing[@]} > 0)); then
    echo "Missing build dependencies for backend '${backend}':"
    printf '  - %s\n' "${missing[@]}"
    echo
    echo "Install or activate the required driver/toolkit/compiler environment, then re-run this script."
    exit 1
  fi
}

prepare_checkout() {
  mkdir -p "$(dirname "${src_dir}")"

  if [[ -e "${src_dir}" && ! -d "${src_dir}/.git" ]]; then
    echo "Checkout path exists but is not a git repository:"
    echo "  ${src_dir}"
    echo "Move it aside or pass --dir <path>."
    exit 1
  fi

  if [[ ! -d "${src_dir}/.git" ]]; then
    echo "Cloning llama.cpp:"
    echo "  ${repo_url} -> ${src_dir}"
    git clone "${repo_url}" "${src_dir}"
  fi

  if [[ -n "$(git -C "${src_dir}" status --porcelain)" ]]; then
    echo "Existing llama.cpp checkout has local changes:"
    echo "  ${src_dir}"
    echo "Commit, stash, or remove those changes before re-running this bootstrap."
    exit 1
  fi

  echo "Fetching llama.cpp tags and refs..."
  git -C "${src_dir}" fetch --tags origin

  echo "Checking out llama.cpp ref:"
  echo "  ${llama_ref}"
  git -C "${src_dir}" checkout "${llama_ref}"

  echo "Resolved llama.cpp commit:"
  echo "  $(git -C "${src_dir}" rev-parse HEAD)"
}

configure_and_build() {
  local -a cmake_args
  cmake_args=(-S "${src_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release)

  case "${backend}" in
    cpu)
      ;;
    cuda)
      cmake_args+=(-DGGML_CUDA=ON)
      ;;
    hip)
      cmake_args+=(-DGGML_HIP=ON)
      ;;
    vulkan)
      cmake_args+=(-DGGML_VULKAN=1)
      ;;
    sycl)
      cmake_args+=(-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON)
      ;;
    metal)
      cmake_args+=(-DGGML_METAL=ON)
      ;;
  esac

  echo
  echo "Configuring llama.cpp (${backend})..."
  if [[ "${backend}" == "hip" ]]; then
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake "${cmake_args[@]}"
  else
    cmake "${cmake_args[@]}"
  fi

  local -a build_args
  build_args=(--build "${build_dir}" --config Release --target llama-bench)
  if [[ -n "${jobs}" ]]; then
    build_args+=(--parallel "${jobs}")
  else
    build_args+=(--parallel)
  fi

  echo
  echo "Building llama-bench..."
  cmake "${build_args[@]}"
}

print_result() {
  local -a candidates
  candidates=(
    "${build_dir}/bin/llama-bench"
    "${build_dir}/bin/Release/llama-bench"
    "${build_dir}/bin/llama-bench.exe"
    "${build_dir}/bin/Release/llama-bench.exe"
  )

  echo
  echo "Build complete."
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "llama-bench:"
      echo "  ${candidate}"
      echo
      echo "Run GPU-Insights LLM benchmark with:"
      printf '  python3 -m llm_bench.cli --llama-bench %q\n' "${candidate}"
      return
    fi
  done

  echo "Could not locate llama-bench under ${build_dir}/bin."
  echo "Inspect the CMake output above or search the build directory manually."
}

prompt_ref
prompt_backend
normalize_backend

echo "llama.cpp bootstrap configuration:"
echo "  ref:       ${llama_ref}"
echo "  backend:   ${backend}"
echo "  checkout:  ${src_dir}"
echo "  build dir: ${build_dir}"
if [[ -n "${jobs}" ]]; then
  echo "  jobs:      ${jobs}"
fi
echo

preflight_dependencies
prepare_checkout
configure_and_build
print_result

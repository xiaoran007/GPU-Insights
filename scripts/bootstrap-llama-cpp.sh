#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

repo_url="${GPU_INSIGHTS_LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
src_dir="${GPU_INSIGHTS_LLAMA_CPP_DIR:-${repo_root}/third_party/llama.cpp}"
build_dir="${GPU_INSIGHTS_LLAMA_CPP_BUILD_DIR:-${src_dir}/build}"
llama_ref="${GPU_INSIGHTS_LLAMA_CPP_REF:-}"
backend="${GPU_INSIGHTS_LLAMA_CPP_BACKEND:-}"
jobs="${GPU_INSIGHTS_LLAMA_CPP_JOBS:-}"
cuda_host_compiler="${GPU_INSIGHTS_LLAMA_CPP_CUDA_HOST_COMPILER:-}"
native="${GPU_INSIGHTS_LLAMA_CPP_NATIVE:-auto}"
prebuilt="${GPU_INSIGHTS_LLAMA_BENCH_PREBUILT:-auto}"
prebuilt_repo="${GPU_INSIGHTS_LLAMA_BENCH_RELEASE_REPO:-xiaoran007/GPU-Insights}"
prebuilt_release_tag="${GPU_INSIGHTS_LLAMA_BENCH_RELEASE_TAG:-latest}"
prebuilt_dir="${GPU_INSIGHTS_LLAMA_BENCH_PREBUILT_DIR:-${repo_root}/third_party/llama-bench}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/bootstrap-llama-cpp.sh [--backend <backend>] [--ref <git-ref>]

Options:
  --ref <git-ref>       llama.cpp tag, branch, or commit to check out. Defaults to origin/HEAD.
  --backend <backend>   One of: cpu, cuda, hip, vulkan, sycl, metal.
  --dir <path>          llama.cpp checkout directory.
  --build-dir <path>    CMake build directory.
  --jobs <n>            Parallel build jobs passed to CMake.
  --cuda-host-compiler <path>
                       Host C++ compiler for CUDA builds.
  --native <auto|on|off>
                       CPU native optimizations. Default: auto (off for CUDA).
  --prebuilt <auto|on|off>
                       For CUDA, try GPU-Insights prebuilt llama-bench first. Default: auto.
  --release-repo <owner/repo>
                       GitHub repo for prebuilt llama-bench release assets.
  --release-tag <tag|latest>
                       GitHub release tag for prebuilt assets. Default: latest.
  -h, --help            Show this help text.

Environment overrides:
  GPU_INSIGHTS_LLAMA_CPP_REF
  GPU_INSIGHTS_LLAMA_CPP_BACKEND
  GPU_INSIGHTS_LLAMA_CPP_DIR
  GPU_INSIGHTS_LLAMA_CPP_BUILD_DIR
  GPU_INSIGHTS_LLAMA_CPP_JOBS
  GPU_INSIGHTS_LLAMA_CPP_REPO
  GPU_INSIGHTS_LLAMA_CPP_CUDA_HOST_COMPILER
  GPU_INSIGHTS_LLAMA_CPP_NATIVE
  GPU_INSIGHTS_LLAMA_BENCH_PREBUILT
  GPU_INSIGHTS_LLAMA_BENCH_RELEASE_REPO
  GPU_INSIGHTS_LLAMA_BENCH_RELEASE_TAG
  GPU_INSIGHTS_LLAMA_BENCH_PREBUILT_DIR
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
    --cuda-host-compiler)
      require_option_value "$1" "${2:-}"
      cuda_host_compiler="$2"
      shift 2
      ;;
    --native)
      require_option_value "$1" "${2:-}"
      native="$2"
      shift 2
      ;;
    --prebuilt)
      require_option_value "$1" "${2:-}"
      prebuilt="$2"
      shift 2
      ;;
    --release-repo)
      require_option_value "$1" "${2:-}"
      prebuilt_repo="$2"
      shift 2
      ;;
    --release-tag)
      require_option_value "$1" "${2:-}"
      prebuilt_release_tag="$2"
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

normalize_native() {
  native="$(printf '%s' "${native}" | tr '[:upper:]' '[:lower:]')"
  case "${native}" in
    auto|on|off) ;;
    *)
      echo "Unsupported native setting: ${native}"
      echo "Expected one of: auto, on, off"
      exit 1
      ;;
  esac
}

normalize_prebuilt() {
  prebuilt="$(printf '%s' "${prebuilt}" | tr '[:upper:]' '[:lower:]')"
  case "${prebuilt}" in
    auto|on|off) ;;
    *)
      echo "Unsupported prebuilt setting: ${prebuilt}"
      echo "Expected one of: auto, on, off"
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

detect_gnu_major() {
  local compiler="$1"
  local version
  version="$("${compiler}" -dumpfullversion -dumpversion 2>/dev/null | head -n 1 || true)"
  if [[ "${version}" =~ ^([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

detect_gnu_version() {
  local compiler="$1"
  "${compiler}" -dumpfullversion -dumpversion 2>/dev/null | head -n 1 || true
}

detect_nvcc_major() {
  nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1
}

preflight_cuda_host_compiler() {
  local host_compiler="${cuda_host_compiler}"

  if [[ -n "${host_compiler}" ]]; then
    if [[ ! -x "${host_compiler}" ]]; then
      missing+=("${host_compiler} (CUDA host compiler path is not executable)")
      return
    fi
  elif command -v g++ >/dev/null 2>&1; then
    host_compiler="$(command -v g++)"
  elif command -v c++ >/dev/null 2>&1; then
    host_compiler="$(command -v c++)"
  fi

  if [[ -z "${host_compiler}" ]]; then
    return
  fi

  local nvcc_major
  local gnu_major
  local gnu_version
  nvcc_major="$(detect_nvcc_major)"
  gnu_major="$(detect_gnu_major "${host_compiler}")"
  gnu_version="$(detect_gnu_version "${host_compiler}")"

  if [[ "${nvcc_major}" == "12" && -n "${gnu_major}" && "${gnu_major}" -gt 14 ]]; then
    missing+=("CUDA 12.x nvcc does not support GCC ${gnu_version} as host compiler; load GCC 14 or older, or pass --cuda-host-compiler /path/to/g++-14")
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
      if command -v nvcc >/dev/null 2>&1; then
        preflight_cuda_host_compiler
      fi
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

detect_prebuilt_platform() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Prebuilt llama-bench assets currently support Linux only." >&2
    return 1
  fi

  case "$(uname -m)" in
    x86_64|amd64)
      printf 'linux-amd64\n'
      ;;
    *)
      echo "Prebuilt llama-bench assets currently support linux-amd64 only." >&2
      return 1
      ;;
  esac
}

detect_cuda_major_for_prebuilt() {
  local major=""

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to select a CUDA prebuilt llama-bench asset." >&2
    return 1
  fi

  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia-smi did not report any visible NVIDIA GPU." >&2
    return 1
  fi

  if command -v nvcc >/dev/null 2>&1; then
    major="$(detect_nvcc_major)"
  fi

  if [[ -z "${major}" && -f /usr/local/cuda/version.txt ]]; then
    major="$(sed -n 's/.*CUDA Version \([0-9][0-9]*\)\..*/\1/p' /usr/local/cuda/version.txt | head -n 1)"
  fi

  if [[ -z "${major}" && -f /usr/local/cuda/version.json ]]; then
    major="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([0-9][0-9]*\)\..*/\1/p' /usr/local/cuda/version.json | head -n 1)"
  fi

  if [[ -z "${major}" ]]; then
    major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
  fi

  case "${major}" in
    12|13)
      printf '%s\n' "${major}"
      ;;
    "")
      echo "Could not detect CUDA major version for prebuilt llama-bench selection." >&2
      return 1
      ;;
    *)
      echo "No GPU-Insights prebuilt llama-bench asset for CUDA ${major}; supported majors: 12, 13." >&2
      return 1
      ;;
  esac
}

release_api_url() {
  if [[ "${prebuilt_release_tag}" == "latest" ]]; then
    printf 'https://api.github.com/repos/%s/releases/latest\n' "${prebuilt_repo}"
  else
    printf 'https://api.github.com/repos/%s/releases/tags/%s\n' "${prebuilt_repo}" "${prebuilt_release_tag}"
  fi
}

select_prebuilt_assets() {
  local platform="$1"
  local cuda_major="$2"
  local prefix="gpu-insights-llama-bench-${platform}-cuda${cuda_major}-"

  python3 - "$(release_api_url)" "${prefix}" <<'PY'
import json
import sys
import urllib.request

api_url = sys.argv[1]
prefix = sys.argv[2]

try:
    with urllib.request.urlopen(api_url) as response:
        release = json.load(response)
except Exception as exc:
    raise SystemExit(f"Could not read GitHub release metadata from {api_url}: {exc}")

assets = release.get("assets", [])
archive = None
for asset in assets:
    name = asset.get("name", "")
    if name.startswith(prefix) and name.endswith(".tar.zst"):
        archive = asset
        break

if archive is None:
    raise SystemExit(f"Could not find release asset matching {prefix}*.tar.zst")

checksum_name = archive["name"] + ".sha256"
checksum = next((asset for asset in assets if asset.get("name") == checksum_name), None)
if checksum is None:
    raise SystemExit(f"Could not find release asset {checksum_name}")

print(archive["name"])
print(archive["browser_download_url"])
print(checksum["browser_download_url"])
PY
}

current_prebuilt_path() {
  printf '%s/current/bin/llama-bench\n' "${prebuilt_dir}"
}

print_prebuilt_result() {
  local executable="$1"
  echo
  echo "Prebuilt llama-bench ready:"
  echo "  ${executable}"
  echo
  echo "Run GPU-Insights LLM benchmark with:"
  printf '  python3 -m llm_bench.cli --llama-bench %q\n' "${executable}"
}

install_cuda_prebuilt() {
  local platform
  local cuda_major
  local asset_info
  local asset_name
  local archive_url
  local checksum_url
  local download_dir
  local archive_path
  local checksum_path
  local expected_sha
  local actual_sha
  local asset_stem
  local target_dir
  local tmp_dir
  local current_link

  for command_name in python3 curl zstd tar sha256sum; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo "${command_name} is required to install prebuilt llama-bench assets." >&2
      return 1
    fi
  done

  platform="$(detect_prebuilt_platform)" || return 1
  cuda_major="$(detect_cuda_major_for_prebuilt)" || return 1
  asset_info="$(select_prebuilt_assets "${platform}" "${cuda_major}")" || return 1

  asset_name="$(printf '%s\n' "${asset_info}" | sed -n '1p')"
  archive_url="$(printf '%s\n' "${asset_info}" | sed -n '2p')"
  checksum_url="$(printf '%s\n' "${asset_info}" | sed -n '3p')"
  asset_stem="${asset_name%.tar.zst}"
  target_dir="${prebuilt_dir}/prebuilt/${asset_stem}"
  current_link="${prebuilt_dir}/current"

  if [[ -x "${target_dir}/bin/llama-bench" ]]; then
    mkdir -p "${prebuilt_dir}"
    if [[ -e "${current_link}" && ! -L "${current_link}" ]]; then
      echo "Cannot update ${current_link}; it exists and is not a symlink." >&2
      return 1
    fi
    rm -f "${current_link}"
    ln -s "prebuilt/${asset_stem}" "${current_link}"
    print_prebuilt_result "$(current_prebuilt_path)"
    return 0
  fi

  download_dir="${prebuilt_dir}/downloads"
  mkdir -p "${download_dir}"
  archive_path="${download_dir}/${asset_name}"
  checksum_path="${download_dir}/${asset_name}.sha256"

  echo "Downloading prebuilt llama-bench:"
  echo "  repo:     ${prebuilt_repo}"
  echo "  release:  ${prebuilt_release_tag}"
  echo "  asset:    ${asset_name}"
  echo "  platform: ${platform}"
  echo "  CUDA:     ${cuda_major}"

  curl -L --fail --retry 3 --output "${archive_path}" "${archive_url}" || return 1
  curl -L --fail --retry 3 --output "${checksum_path}" "${checksum_url}" || return 1

  expected_sha="$(awk '{print $1; exit}' "${checksum_path}")"
  actual_sha="$(sha256sum "${archive_path}" | awk '{print $1}')" || return 1
  if [[ -z "${expected_sha}" || "${expected_sha}" != "${actual_sha}" ]]; then
    echo "Downloaded prebuilt llama-bench checksum mismatch:" >&2
    echo "  expected: ${expected_sha}" >&2
    echo "  actual:   ${actual_sha}" >&2
    return 1
  fi

  tmp_dir="${target_dir}.tmp.$$"
  rm -rf "${tmp_dir}" || return 1
  mkdir -p "${tmp_dir}" || return 1
  zstd -dc "${archive_path}" | tar -xf - -C "${tmp_dir}" || return 1

  if [[ ! -x "${tmp_dir}/bin/llama-bench" ]]; then
    echo "Downloaded prebuilt archive did not contain bin/llama-bench." >&2
    rm -rf "${tmp_dir}"
    return 1
  fi

  rm -rf "${target_dir}" || return 1
  mv "${tmp_dir}" "${target_dir}" || return 1

  if [[ -e "${current_link}" && ! -L "${current_link}" ]]; then
    echo "Cannot update ${current_link}; it exists and is not a symlink." >&2
    return 1
  fi
  rm -f "${current_link}"
  ln -s "prebuilt/${asset_stem}" "${current_link}"

  print_prebuilt_result "$(current_prebuilt_path)"
  return 0
}

continue_with_source_build() {
  if [[ "${prebuilt}" == "on" ]]; then
    return 1
  fi

  if [[ ! -t 0 ]]; then
    echo "Prebuilt llama-bench was unavailable; continuing with source build in non-interactive mode."
    return 0
  fi

  local choice
  read -r -p "Build llama.cpp from source instead? [y/N]: " choice
  case "${choice}" in
    y|Y|yes|YES)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

try_cuda_prebuilt_or_continue() {
  if [[ "${backend}" != "cuda" || "${prebuilt}" == "off" ]]; then
    return 1
  fi

  echo "Trying GPU-Insights prebuilt llama-bench for CUDA..."
  if install_cuda_prebuilt; then
    return 0
  fi

  echo
  echo "Prebuilt llama-bench install failed."
  if continue_with_source_build; then
    echo
    echo "Falling back to source build."
    return 1
  fi

  echo "Stopped before source build."
  exit 1
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
  git -C "${src_dir}" remote set-head origin -a >/dev/null 2>&1 || true

  echo "Checking out llama.cpp ref:"
  echo "  ${llama_ref}"
  git -C "${src_dir}" checkout "${llama_ref}"

  echo "Resolved llama.cpp commit:"
  echo "  $(git -C "${src_dir}" rev-parse HEAD)"
}

configure_and_build() {
  local -a cmake_args
  cmake_args=(-S "${src_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release)

  case "${native}" in
    on)
      cmake_args+=(-DGGML_NATIVE=ON)
      ;;
    off)
      cmake_args+=(-DGGML_NATIVE=OFF)
      ;;
    auto)
      if [[ "${backend}" == "cuda" ]]; then
        cmake_args+=(-DGGML_NATIVE=OFF)
      fi
      ;;
  esac

  case "${backend}" in
    cpu)
      ;;
    cuda)
      cmake_args+=(-DGGML_CUDA=ON)
      if [[ -n "${cuda_host_compiler}" ]]; then
        cmake_args+=("-DCMAKE_CUDA_HOST_COMPILER=${cuda_host_compiler}")
      fi
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

if [[ -z "${llama_ref}" ]]; then
  llama_ref="origin/HEAD"
fi
prompt_backend
normalize_backend
normalize_native
normalize_prebuilt

echo "llama.cpp bootstrap configuration:"
echo "  ref:       ${llama_ref}"
echo "  backend:   ${backend}"
echo "  checkout:  ${src_dir}"
echo "  build dir: ${build_dir}"
echo "  native:    ${native}"
echo "  prebuilt:  ${prebuilt}"
if [[ "${backend}" == "cuda" && "${prebuilt}" != "off" ]]; then
  echo "  release:   ${prebuilt_repo}@${prebuilt_release_tag}"
  echo "  install:   ${prebuilt_dir}"
fi
if [[ -n "${jobs}" ]]; then
  echo "  jobs:      ${jobs}"
fi
if [[ -n "${cuda_host_compiler}" ]]; then
  echo "  CUDA host compiler: ${cuda_host_compiler}"
fi
echo

if try_cuda_prebuilt_or_continue; then
  exit 0
fi

preflight_dependencies
prepare_checkout
configure_and_build
print_result

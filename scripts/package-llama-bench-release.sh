#!/usr/bin/env bash
set -euo pipefail

src_dir=""
build_dir=""
out_dir=""
platform="linux-amd64"
backend="cuda"
cuda_major=""
cuda_architectures=""
cmake_flags=""
asset_prefix="gpu-insights-llama-bench"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/package-llama-bench-release.sh --src-dir <llama.cpp> --build-dir <build> --out-dir <dir>

Options:
  --src-dir <path>              llama.cpp checkout directory.
  --build-dir <path>            llama.cpp CMake build directory.
  --out-dir <path>              Output directory for release assets.
  --platform <name>             Release platform label. Default: linux-amd64.
  --backend <name>              Backend label. Default: cuda.
  --cuda-major <n>              CUDA major version label.
  --cuda-architectures <list>   CMake CUDA architectures list.
  --cmake-flags <text>          CMake flags to record in BUILD-MANIFEST.json.
  --asset-prefix <name>         Asset filename prefix.
  -h, --help                    Show this help text.
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
    --src-dir)
      require_option_value "$1" "${2:-}"
      src_dir="$2"
      shift 2
      ;;
    --build-dir)
      require_option_value "$1" "${2:-}"
      build_dir="$2"
      shift 2
      ;;
    --out-dir)
      require_option_value "$1" "${2:-}"
      out_dir="$2"
      shift 2
      ;;
    --platform)
      require_option_value "$1" "${2:-}"
      platform="$2"
      shift 2
      ;;
    --backend)
      require_option_value "$1" "${2:-}"
      backend="$2"
      shift 2
      ;;
    --cuda-major)
      require_option_value "$1" "${2:-}"
      cuda_major="$2"
      shift 2
      ;;
    --cuda-architectures)
      require_option_value "$1" "${2:-}"
      cuda_architectures="$2"
      shift 2
      ;;
    --cmake-flags)
      require_option_value "$1" "${2:-}"
      cmake_flags="$2"
      shift 2
      ;;
    --asset-prefix)
      require_option_value "$1" "${2:-}"
      asset_prefix="$2"
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

if [[ -z "${src_dir}" || -z "${build_dir}" || -z "${out_dir}" ]]; then
  echo "Missing required --src-dir, --build-dir, or --out-dir."
  usage
  exit 1
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd} is required to package llama-bench release assets." >&2
    exit 1
  fi
}

find_llama_bench() {
  local -a candidates=(
    "${build_dir}/bin/llama-bench"
    "${build_dir}/bin/Release/llama-bench"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

json_escape() {
  printf '%s' "$1" \
    | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g'
}

json_string_array_from_semicolon_list() {
  local value="$1"
  local first=1
  local item
  local -a items=()
  printf '['
  IFS=';' read -r -a items <<< "${value}"
  for item in "${items[@]}"; do
    if [[ -z "${item}" ]]; then
      continue
    fi
    if [[ "${first}" -eq 0 ]]; then
      printf ', '
    fi
    printf '"%s"' "$(json_escape "${item}")"
    first=0
  done
  printf ']'
}

copy_llama_libraries() {
  local binary="$1"
  local target_dir="$2"
  local build_root
  local lib_path
  local lib_name
  build_root="$(cd "${build_dir}" && pwd)"
  while IFS= read -r lib_path; do
    if [[ ! -f "${lib_path}" ]]; then
      continue
    fi
    case "${lib_path}" in
      "${build_root}"/*) ;;
      *)
        continue
        ;;
    esac
    lib_name="$(basename "${lib_path}")"
    case "${lib_name}" in
      lib*.so*)
        cp -L "${lib_path}" "${target_dir}/${lib_name}"
        ;;
    esac
  done < <(ldd "${binary}" | awk '/=>/ {print $(NF-1)} /^[[:space:]]*\// {print $1}')
}

strip_release_binaries() {
  local file
  strip --strip-unneeded "${stage_dir}/bin/llama-bench.bin"
  while IFS= read -r file; do
    strip --strip-unneeded "${file}"
  done < <(find "${stage_dir}/lib" -maxdepth 1 -type f -name '*.so*' | sort)
}

write_wrapper() {
  local wrapper="$1"
  cat > "${wrapper}" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${root}/lib:${LD_LIBRARY_PATH:-}"
exec "${root}/bin/llama-bench.bin" "$@"
WRAPPER
  chmod +x "${wrapper}"
}

require_cmd ldd
require_cmd sha256sum
require_cmd strip
require_cmd zstd

llama_bench="$(find_llama_bench)" || {
  echo "Could not locate built llama-bench under ${build_dir}/bin." >&2
  exit 1
}

mkdir -p "${out_dir}"
stage_dir="$(mktemp -d)"
trap 'rm -rf "${stage_dir}"' EXIT

mkdir -p "${stage_dir}/bin" "${stage_dir}/lib" "${stage_dir}/licenses"
cp "${llama_bench}" "${stage_dir}/bin/llama-bench.bin"
write_wrapper "${stage_dir}/bin/llama-bench"
copy_llama_libraries "${llama_bench}" "${stage_dir}/lib"
strip_release_binaries

if [[ -f "${src_dir}/LICENSE" ]]; then
  cp "${src_dir}/LICENSE" "${stage_dir}/licenses/LICENSE.llama.cpp"
else
  echo "Missing llama.cpp LICENSE at ${src_dir}/LICENSE." >&2
  exit 1
fi

llama_commit="$(git -C "${src_dir}" rev-parse HEAD)"
llama_commit_short="$(git -C "${src_dir}" rev-parse --short=12 HEAD)"
cuda_label=""
if [[ -n "${cuda_major}" ]]; then
  cuda_label="cuda${cuda_major}"
else
  cuda_label="${backend}"
fi
asset_name="${asset_prefix}-${platform}-${cuda_label}-${llama_commit_short}.tar.zst"

packaged_libraries="$(find "${stage_dir}/lib" -maxdepth 1 -type f -printf '%f\n' | sort | paste -sd ';' -)"

cat > "${stage_dir}/BUILD-MANIFEST.json" <<EOF
{
  "schemaVersion": "1.0",
  "name": "gpu-insights-llama-bench",
  "platform": "$(json_escape "${platform}")",
  "backend": "$(json_escape "${backend}")",
  "cudaMajor": "$(json_escape "${cuda_major}")",
  "cudaArchitectures": $(json_string_array_from_semicolon_list "${cuda_architectures}"),
  "ggmlNative": false,
  "llamaCppRepo": "https://github.com/ggml-org/llama.cpp",
  "llamaCppCommit": "$(json_escape "${llama_commit}")",
  "cmakeFlags": "$(json_escape "${cmake_flags}")",
  "artifact": "$(json_escape "${asset_name}")",
  "compression": "zstd",
  "stripped": true,
  "packagedLibraries": $(json_string_array_from_semicolon_list "${packaged_libraries}"),
  "externalRuntimeRequirements": [
    "NVIDIA driver",
    "CUDA ${cuda_major} runtime libraries",
    "cuBLAS/cuBLASLt"
  ]
}
EOF

(
  cd "${stage_dir}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

tar \
  --sort=name \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "${stage_dir}" \
  -cf - . | zstd -T0 -19 -f -o "${out_dir}/${asset_name}"
sha256sum "${out_dir}/${asset_name}" > "${out_dir}/${asset_name}.sha256"

echo "Packaged llama-bench release asset:"
echo "  ${out_dir}/${asset_name}"
echo "Packaged libraries:"
find "${stage_dir}/lib" -maxdepth 1 -type f -printf '  %f\n' | sort
echo "SHA256:"
cat "${out_dir}/${asset_name}.sha256"

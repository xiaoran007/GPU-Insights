# GPU-Insights

Multi-model GPU/NPU training performance benchmark suite. Measures compute throughput across diverse deep learning workloads and hardware platforms.

## Features

- **Smart Launcher** — One command auto-selects device, precision, ABS, and CUDA DDP
- **5 Benchmark Models** — CNN, ResNet-50, ViT, UNet, DDPM covering classification, segmentation, and diffusion
- **6 Device Backends** — CUDA, MPS, NPU (Huawei Ascend), MUSA (Moore Threads), TPU, auto-detection
- **DDP Multi-GPU** — Distributed data-parallel training via `torchrun`
- **Auto Batch Size** — Calibration-table-based automatic batch size selection (NVML)
- **Unified Scoring** — Throughput-based scoring system consistent across all models
- **LLM Inference Track** — Standalone full-GPU llama.cpp benchmark for coding-agent workloads

## Quick Start

```shell
# Install dependencies
pip install torch torchvision

# Run one model with the smart launcher
python3 main_auto.py -mt resnet50

# Omit --model to run the major model set: resnet50, vit, unet, ddpm
python3 main_auto.py

# Preview what the smart launcher will do
python3 main_auto.py -mt vit --dry-run

# Force a single precision or batch size when needed
python3 main_auto.py -mt unet --dtype FP32
python3 main_auto.py -mt ddpm -bs 32
```

Legacy expert entrypoints remain available:

```shell
python main.py -mt resnet50 -s 512 -e 2 -dt FP32
torchrun --nproc_per_node=2 main_ddp.py -mt resnet50 -s 512 -e 2 -abs -dt FP16
python main_tpu.py -mt resnet50 -s 512 -e 2 -dt BF16
```

## LLM Inference Benchmark

The LLM inference track is separate from the training benchmark path. It uses
`llama-bench` from an external llama.cpp installation and records prompt
processing (PP) plus token generation (TG) throughput for coding-agent-shaped
cases.

GPU-Insights does not install or configure llama.cpp, CUDA, ROCm, Vulkan, or
SYCL. Build/install llama.cpp for your platform first. The launcher checks the
bootstrap build output under `third_party/llama.cpp/build/bin/`, then `PATH`,
or you can pass an explicit path with `--llama-bench`.

### Prepare llama.cpp

GPU-Insights calls the `llama-bench` binary produced by llama.cpp. Follow the
upstream llama.cpp build guide for your platform:

- Official build guide: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- Intel GPU / SYCL guide: <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md>

GPU-Insights provides a helper that clones llama.cpp, checks out upstream
`origin/HEAD` by default, asks which backend to build, and builds only
`llama-bench`. Pass `--ref <git-ref>` only when you want to pin a specific
llama.cpp commit, branch, or tag. The helper does not install GPU drivers,
CUDA, ROCm, Vulkan SDK, oneAPI, compilers, or CMake; missing build commands are
reported before cloning/building.

```shell
# Interactive backend selection
bash scripts/bootstrap-llama-cpp.sh

# Non-interactive CUDA example
bash scripts/bootstrap-llama-cpp.sh --backend cuda --jobs 16

# Optional pinned-ref build
bash scripts/bootstrap-llama-cpp.sh --ref <llama.cpp-commit> --backend cuda --jobs 16
```

On CUDA systems, if the active GCC is newer than the CUDA toolkit supports,
load a compatible compiler module first or pass it explicitly:

```shell
bash scripts/bootstrap-llama-cpp.sh --backend cuda --cuda-host-compiler /path/to/g++-14
```

By default the helper uses `third_party/llama.cpp`, which is ignored by git.
Override paths with `--dir`, `--build-dir`, or the corresponding
`GPU_INSIGHTS_LLAMA_CPP_*` environment variables printed by `--help`.

General source checkout:

```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Common local builds:

```shell
# CPU-only sanity build
cmake -B build
cmake --build build --config Release -j

# NVIDIA CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# AMD ROCm / HIP on Linux
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Vulkan, useful for cross-vendor Windows/Linux setups
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j
```

On Windows, use a Visual Studio 2022 Developer Command Prompt or the toolchain
prompt required by the selected GPU backend, then run the same CMake build shape
with the relevant `GGML_*` backend flag. On macOS, Metal is enabled by default
in llama.cpp, so the regular CMake build is the expected local GPU build.

For Intel GPU, llama.cpp recommends the SYCL backend with Intel oneAPI. After
installing Intel GPU drivers and oneAPI, source the oneAPI environment and
build with SYCL:

```shell
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build --config Release -j
```

After building, either add `llama.cpp/build/bin` to `PATH` or pass the binary
explicitly:

```shell
python3 -m llm_bench.cli --llama-bench /path/to/llama.cpp/build/bin/llama-bench
```

Keep GPU memory behavior strict for benchmark submissions. Do not enable CUDA
unified-memory/system-RAM fallback for submitted full-GPU results; if a case
does not fit in VRAM, let it fail and import the failed case status.

### Fixed model and cases

The default contract lives in `llm_bench/configs/default.json`.

Model:

- Repo: `unsloth/Qwen3.6-27B-GGUF`
- Revision: `82d411acf4a06cfb8d9b073a5211bf410bfc29bf`
- File: `Qwen3.6-27B-Q4_K_M.gguf`
- Default local path: `models/llm/Qwen3.6-27B-Q4_K_M.gguf`

Cases:

| Case | Prompt Tokens | Generation Tokens | Purpose |
|------|--------------:|------------------:|---------|
| `agent_step_small` | 2,048 | 128 | Short tool result or agent loop step |
| `single_file_edit` | 8,192 | 512 | One file plus focused context |
| `multi_file_patch` | 16,384 | 1,024 | Multi-file edit with longer patch output |
| `repo_context_plan` | 32,768 | 512 | Large repo context with concise planning |
| `long_context_debug` | 65,536 | 1,024 | Long logs/diffs/context with substantial response |

Results are full-GPU only: the default config uses `nGpuLayers: -1` and does
not fall back to partial CPU offload. If a GPU cannot fit a case, that case is
recorded as `status: failed` with the error text.

The default runtime profile is a single representative coding-agent setup, not
a tuning matrix: full GPU offload, F16 KV cache, flash attention enabled,
`batchSize: 2048`, `ubatchSize: 512`, and an explicit per-case context size
rounded up from `prompt + generation + 256` tokens. MTP/speculative decoding is
not part of the canonical baseline.

### Download the fixed GGUF

```shell
# Oscar cluster: link models/llm to persistent storage before downloading
bash scripts/bootstrap-llm-oscar.sh

# AutoDL: link models/llm to persistent storage before downloading
bash scripts/bootstrap-llm-autodl.sh

# Colab: copy an existing Drive GGUF into local runtime storage before running
bash scripts/bootstrap-llm-colab.sh

# Preview the exact Hugging Face URL and output path
python3 scripts/download-llm-model.py --dry-run

# Download to models/llm/Qwen3.6-27B-Q4_K_M.gguf
python3 scripts/download-llm-model.py
```

The downloader is only a model-file helper. It does not install llama.cpp or
configure GPU runtime libraries.

If the final GGUF already exists and matches the configured expected byte size,
the downloader exits without downloading. Interrupted downloads are kept as a
`.part` file and resumed on the next run when the server honors HTTP Range
requests. Progress shows human-readable downloaded and total sizes; when
resuming, downloaded includes the bytes already present in the `.part` file.

On Oscar, `scripts/bootstrap-llm-oscar.sh` links the repo-local `models/llm`
path to `/users/tfang11/tfang/llm` so the GGUF survives compute-node teardown
and does not need to be downloaded again. Override the target with
`GPU_INSIGHTS_OSCAR_LLM_DIR=/path/to/llm` if needed.

On AutoDL, `scripts/bootstrap-llm-autodl.sh` links `models/llm` to
`/root/autodl-fs/llm`. Override the target with
`GPU_INSIGHTS_AUTODL_LLM_DIR=/path/to/llm` if needed.

On Colab, `scripts/bootstrap-llm-colab.sh` expects the fixed GGUF to already
exist in Google Drive at `/content/drive/MyDrive/GPU-Insights/llm`, copies it
into `/content/gpu-insights-llm`, then links `models/llm` to that local runtime
cache. This keeps Google Drive as persistent storage while avoiding Drive I/O on
the benchmark hot path. Mount Drive in the notebook first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Override the Drive cache with `GPU_INSIGHTS_COLAB_LLM_DIR=/path/to/llm`, or the
local runtime cache with `GPU_INSIGHTS_COLAB_LOCAL_LLM_DIR=/content/path`. If the
Drive GGUF is missing or has the wrong byte size, the script exits without
copying or downloading.

### Run the LLM benchmark

```shell
# Run all configured coding-agent cases
python3 -m llm_bench.cli

# Use a specific llama-bench binary
python3 -m llm_bench.cli --llama-bench /path/to/llama-bench

# Run one case
python3 -m llm_bench.cli --case repo_context_plan

# List configured cases
python3 -m llm_bench.cli --list-cases
```

By default the launcher looks for `llama-bench` in the bootstrap build output
under `third_party/llama.cpp/build/bin/` before falling back to `PATH`. During a
run it prints the selected runtime, model path, per-case progress, PP/TG
throughput results, a summary, and finally a short import command. `llama-bench`
stderr is streamed live with a `llama-bench |` prefix so backend/debug messages
remain visible while stdout is still parsed as JSON.

By default the dashboard import payload is written to `outputs/llm-bench/`, and
a sidecar `*.debug.json` file keeps the full raw llama-bench rows for debugging:

```text
LLM_RESULT_PAYLOAD_FILE=outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json
LLM_DEBUG_PAYLOAD_FILE=outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.debug.json

Import:
  python3 scripts/manage-data.py l outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json
```

Import it into the dashboard data:

```shell
python3 scripts/manage-data.py l outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json
```

Pass `--emit-base64` if you need the legacy `LLM_RESULT_PAYLOAD_B64=...` line
for copy-paste workflows.

For development without running llama.cpp:

```shell
python3 -m llm_bench.cli \
  --mock-result-file llm_bench/mock/llama-bench-qwen3_6_27b-q4.json \
  --pretty
```

## Models

| Model | Parameters | Input Size | Task | Aliases |
|-------|-----------|------------|------|---------|
| CNN | 62K | 3×32×32 | Classification | `cnn` |
| ResNet-50 | 23.5M | 3×32×32 | Classification | `resnet50`, `ResNet-50`, `ResNet50` |
| ViT-Base/16 | 85.8M | 3×224×224 | Classification | `vit`, `vit-base` |
| UNet | 31.0M | 3×256×256 | Segmentation | `unet` |
| DDPM | 62.3M | 3×64×64 | Diffusion (noise prediction) | `ddpm` |

## CLI Arguments

### Smart launcher (`main_auto.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `-mt`, `--model` | Model to benchmark. Omit to run `resnet50`, `vit`, `unet`, `ddpm` | all four major models |
| `-s`, `--size` | Data size in MB | `1024` |
| `-e`, `--epochs` | Training epochs | `5` |
| `-d`, `--device` | Device: `auto`, `cuda`, `mps`, `npu`, `musa`, `tpu` | `auto` |
| `-gpu`, `--gpu_id` | CUDA GPU ids, e.g. `all` or `0,1` | `all` |
| `-dt`, `--dtype` | Run a single precision instead of auto-selection | auto |
| `--no-abs` | Disable auto batch size | off |
| `-bs`, `--batch` | Batch size override | `0` |
| `--single-process` | Disable automatic CUDA DDP | off |
| `--dry-run` | Print launch plan without running | off |

Default smart-launch behavior:

- Auto-detects the backend using the existing backend priority.
- Enables ABS by default unless `-bs` or `--no-abs` is provided.
- Runs `BF16 + FP32` on BF16-capable devices, otherwise `FP16 + FP32` when AMP is supported.
- Automatically switches to CUDA DDP when multiple CUDA GPUs are visible and the model supports DDP.
- Runs `resnet50`, `vit`, `unet`, and `ddpm` in order when `--model` is omitted.
- Prints a final `RESULT_PAYLOAD_B64=...` line after the human summary so benchmark results can be copied into an update script.

### Smart launcher output payload

At the end of a real `main_auto.py` run, the launcher prints one machine-readable line:

```text
RESULT_PAYLOAD_B64=<base64-json>
```

The decoded JSON includes:

- `schema_version` and `generated_at`
- `source` launcher metadata
- `host` metadata such as `vendor`, `architecture`, `device`, `memory`, `platform`, and `driver_runtime`
- `benchmarks`, one normalized entry per model

Payload compatibility notes:

- `FP32` fills `fp32` / `fp32bs`
- `FP16` fills `fp16` / `fp16bs`
- `BF16` is intentionally exported into `fp16` / `fp16bs` for compatibility with the current dashboard data schema
- per-precision status metadata is preserved in the payload for downstream tooling

### Legacy entrypoints (`main.py`, `main_ddp.py`, `main_tpu.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `-mt`, `--model` | Model to benchmark | `resnet50` |
| `-s`, `--size` | Data size in MB | `1024` |
| `-e`, `--epochs` | Training epochs | `5` |
| `-dt`, `--data_type` | Precision: `FP32`, `FP16`, `BF16` | `FP32` |
| `-bs`, `--batch` | Batch size (0 = model default) | `0` |
| `-abs`, `--auto_batch_size` | Auto batch size via calibration table | off |
| `-d`, `--device` | Device: `auto`, `cuda`, `mps`, `npu`, `musa`, `tpu` | `auto` |
| `-gpu`, `--gpu_id` | GPU ID(s), e.g. `0` or `0,1` | `0` |
| `-cudnn`, `--cudnn_benchmark` | Enable cuDNN benchmark mode | off |

## Makefile Targets

```shell
make smart      # Smart launcher (set MODEL=vit, MODEL=unet, etc.)
make tpu        # ResNet50 on TPU single-core
make tpu-multi  # ResNet50 on TPU 8-core
make calibrate  # Run memory calibration
make docs       # Build visualization website
make docs-dev   # Start docs dev server
make help       # Show current targets and variables
```

## Device Backends

| Backend | Hardware | Requirements |
|---------|----------|-------------|
| `cuda` | NVIDIA GPUs | PyTorch with CUDA |
| `mps` | Apple Silicon | PyTorch ≥ 1.12, macOS |
| `npu` | Huawei Ascend | `torch_npu` |
| `musa` | Moore Threads | `torch_musa` |
| `tpu` | Google TPU | `torch_xla` |
| `auto` | Auto-detect | Tries CUDA → NPU → MUSA → MPS |

Use `--device` to select a specific backend, or leave as `auto` (default).

## Environment Probe Script

Use the dedicated probe helper to inspect the normalized host metadata used by the smart launcher payload:

```shell
python3 scripts/probe_benchmark_env.py --pretty
python3 scripts/probe_benchmark_env.py -d cuda -gpu 0
```

On NVIDIA systems, the script prefers NVML (`pynvml`) for device name, total memory, and driver version, then combines that with CUDA compute capability from PyTorch to map the GPU architecture name.

## DDP Multi-GPU Training

```shell
# Smart launcher
make smart MODEL=vit
python3 main_auto.py -mt resnet50

# 2 GPUs (default)
torchrun --nproc_per_node=2 main_ddp.py -mt resnet50 -s 512 -e 2 -abs -dt FP16

# DDP with ViT
torchrun --nproc_per_node=4 main_ddp.py -mt vit -s 512 -e 2 -bs 32 -dt FP16
```

## Auto Batch Size (ABS)

When `-abs` is enabled, the benchmark automatically selects a batch size based on an NVML-calibrated memory profile table. The selection logic:

1. Queries the device's total VRAM
2. Applies a 10% safety margin (90% usable)
3. Looks up pre-measured `(model, dtype)` peak memory data from the calibration table
4. Picks the largest batch size whose peak memory fits within the usable budget
5. Falls back to the model's default batch size if no calibration data exists

The calibration table lives in `benchmark/calibration.py`. To generate calibration data for your GPU, see [Memory Calibration](#memory-calibration) below.

**Backend support:** CUDA is the primary target. NPU/MUSA use CUDA calibration data as a proxy. MPS/TPU fall back to model defaults.

## Memory Calibration

The calibration tool measures real peak VRAM via NVML (`pynvml`) during short training runs:

```shell
# Install dependency
pip install pynvml

# Full calibration (all models × all precisions)
python calibrate_memory.py

# Specific model/precision
python calibrate_memory.py -mt resnet50 -dt FP16

# Custom batch sizes
python calibrate_memory.py -mt vit -dt FP32 -bs 8,16,32,64

# JSON output (for programmatic use)
python calibrate_memory.py --json

# Specify GPU
python calibrate_memory.py -gpu 1
```

After running, paste the output into `benchmark/calibration.py` `CALIBRATION_TABLE`.

## How to Understand Results

The benchmark evaluates hardware training throughput under a fixed workload. Output is a **score** representing compute performance — higher is better. Scores are affected by compute capability, memory bandwidth, and PCIe/interconnect bandwidth.

## Results

For a visual dashboard, visit the [GPU-Insights Dashboard](https://xiaoran007.github.io/GPU-Insights/).

## Data Management

```shell
# Import the final smart-launcher payload directly into the dashboard data
python3 scripts/manage-data.py import-payload 'RESULT_PAYLOAD_B64=...'

# Decode a payload for inspection
python3 scripts/manage-data.py decode-payload 'RESULT_PAYLOAD_B64=...' --pretty

# Import an LLM inference payload
python3 scripts/manage-data.py l outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json

# Validate benchmark data
python3 scripts/manage-data.py validate

# Show statistics
python3 scripts/manage-data.py stats

# Add a benchmark entry
python3 scripts/manage-data.py add \
  --vendor nvidia --architecture Ada \
  --device "RTX 4090" --memory "24GB" \
  --platform "Linux" --fp32 24000 --fp32bs 512 \
  --fp16 43000 --fp16bs 1024

# Migrate version field
python3 scripts/manage-data.py migrate-version
```

The smart launcher payload is designed to be script-friendly. The recommended update flow is:

```shell
python3 main_auto.py
python3 scripts/manage-data.py import-payload '<paste RESULT_PAYLOAD_B64 value here>'
```

`import-payload` accepts either the full `RESULT_PAYLOAD_B64=...` line or the raw Base64 value. Failed models with no successful precision result are skipped automatically. When `model + vendor + architecture + device + memory` all match an existing entry, that entry is updated in place; otherwise a new entry is appended. Exact duplicate payload rows are treated as no-op updates and skipped.

## Project Structure

```
├── main_auto.py         # Smart launcher with auto device / precision / DDP planning
├── main.py              # Single-device entry point
├── main_ddp.py          # DDP multi-GPU entry point
├── main_tpu.py          # TPU entry point
├── calibrate_memory.py  # NVML memory calibration tool
├── scripts/
│   ├── probe_benchmark_env.py  # Host/device metadata probe used by main_auto.py payload export
│   ├── bootstrap-llm-oscar.sh  # Link models/llm to Oscar persistent storage
│   ├── bootstrap-llm-autodl.sh # Link models/llm to AutoDL persistent storage
│   ├── bootstrap-llm-colab.sh  # Copy Drive GGUF into Colab local runtime cache
│   ├── download-llm-model.py   # Fixed Qwen3.6-27B Q4_K_M GGUF downloader
│   └── manage-data.py          # Benchmark data management
├── Makefile             # Smart launcher and developer convenience targets
├── llm_bench/           # Standalone llama.cpp LLM inference benchmark adapter
├── benchmark/
│   ├── Bench.py         # Orchestrator
│   ├── cli.py           # Unified CLI parsing
│   ├── scoring.py       # Scoring system
│   ├── calibration.py   # Calibration table + auto batch size logic
│   ├── models/          # BenchModel implementations
│   │   ├── base.py      # BenchModel ABC
│   │   ├── cnn.py       # Simple CNN (62K params)
│   │   ├── resnet50.py  # ResNet50 (23.5M params)
│   │   ├── vit.py       # ViT-Base/16 (85.8M params)
│   │   ├── unet.py      # UNet segmentation (31.0M params)
│   │   └── ddpm.py      # DDPM diffusion (62.3M params)
│   ├── devices/         # DeviceBackend implementations
│   │   ├── base.py      # DeviceBackend ABC
│   │   ├── cuda_device.py
│   │   ├── macos_info.py
│   │   ├── mps_device.py
│   │   ├── npu_device.py
│   │   ├── musa_device.py
│   │   └── tpu_device.py
│   ├── runners/         # Training runners
│   │   ├── common.py    # Shared training utilities
│   │   ├── single_runner.py
│   │   └── ddp_runner.py
│   └── data/            # Dataset utilities
└── docs/                # GitHub Pages dashboard
```

## License

See [LICENSE](LICENSE).

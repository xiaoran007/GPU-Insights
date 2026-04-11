# GPU-Insights

Multi-model GPU/NPU training performance benchmark suite. Measures compute throughput across diverse deep learning workloads and hardware platforms.

## Features

- **Smart Launcher** — One command auto-selects device, precision, ABS, and CUDA DDP
- **5 Benchmark Models** — CNN, ResNet-50, ViT, UNet, DDPM covering classification, segmentation, and diffusion
- **6 Device Backends** — CUDA, MPS, NPU (Huawei Ascend), MUSA (Moore Threads), TPU, auto-detection
- **DDP Multi-GPU** — Distributed data-parallel training via `torchrun`
- **Auto Batch Size** — Calibration-table-based automatic batch size selection (NVML)
- **Unified Scoring** — Throughput-based scoring system consistent across all models

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

The smart launcher payload is designed to be script-friendly. A follow-up updater can consume the final line directly, for example:

```shell
python3 new_data.py '<paste RESULT_PAYLOAD_B64 value here>'
```

## Project Structure

```
├── main_auto.py         # Smart launcher with auto device / precision / DDP planning
├── main.py              # Single-device entry point
├── main_ddp.py          # DDP multi-GPU entry point
├── main_tpu.py          # TPU entry point
├── calibrate_memory.py  # NVML memory calibration tool
├── scripts/
│   ├── probe_benchmark_env.py  # Host/device metadata probe used by main_auto.py payload export
│   └── manage-data.py          # Benchmark data management
├── Makefile             # Smart launcher and developer convenience targets
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

# GPU-Insights

Multi-model GPU/NPU training performance benchmark suite. Measures compute throughput across diverse deep learning workloads and hardware platforms.

## Features

- **5 Benchmark Models** — CNN, ResNet50, ViT, UNet, DDPM covering classification, segmentation, and diffusion
- **6 Device Backends** — CUDA, MPS, NPU (Huawei Ascend), MUSA (Moore Threads), TPU, auto-detection
- **DDP Multi-GPU** — Distributed data-parallel training via `torchrun`
- **Auto Batch Size** — Automatic memory-optimal batch size detection
- **Unified Scoring** — Throughput-based scoring system consistent across all models

## Quick Start

```shell
# Install dependencies
pip install torch torchvision

# Run default benchmark (ResNet50, auto-detect device)
python main.py -mt resnet50 -s 512 -e 2 -abs -dt FP32

# Run ViT benchmark
python main.py -mt vit -s 512 -e 2 -bs 32 -dt FP16
```

## Models

| Model | Parameters | Input Size | Task | Aliases |
|-------|-----------|------------|------|---------|
| CNN | 62K | 3×32×32 | Classification | `cnn` |
| ResNet50 | 23.5M | 3×32×32 | Classification | `resnet50`, `resnet` |
| ViT-Base/16 | 85.8M | 3×224×224 | Classification | `vit`, `vit-base` |
| UNet | 31.0M | 3×256×256 | Segmentation | `unet` |
| DDPM | 29.9M | 3×64×64 | Diffusion (noise prediction) | `ddpm` |

## CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-mt`, `--model` | Model to benchmark | `resnet50` |
| `-s`, `--size` | Data size in MB | `1024` |
| `-e`, `--epochs` | Training epochs | `5` |
| `-dt`, `--data_type` | Precision: `FP32`, `FP16`, `BF16` | `FP32` |
| `-bs`, `--batch` | Batch size (0 = model default) | `0` |
| `-abs`, `--auto_batch_size` | Auto batch size optimisation | off |
| `-d`, `--device` | Device: `auto`, `cuda`, `mps`, `npu`, `musa`, `tpu` | `auto` |
| `-cudnn`, `--cudnn_benchmark` | Enable cuDNN benchmark mode | off |

## Makefile Targets

```shell
make run        # ResNet50 FP16 + FP32
make abs        # ResNet50 with auto batch size
make vit        # ViT-Base FP16 + FP32
make unet       # UNet segmentation FP16 + FP32
make ddpm       # DDPM diffusion FP16 + FP32
make ddp        # ResNet50 DDP (GPU=2 by default)
make ddp-abs    # ResNet50 DDP with auto batch size
make tpu        # ResNet50 on TPU single-core
make tpu-multi  # ResNet50 on TPU 8-core
make help       # Show all targets
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

## DDP Multi-GPU Training

```shell
# 2 GPUs (default)
torchrun --nproc_per_node=2 main_ddp.py -mt resnet50 -s 512 -e 2 -abs -dt FP16

# 4 GPUs
make ddp GPU=4

# DDP with ViT
torchrun --nproc_per_node=4 main_ddp.py -mt vit -s 512 -e 2 -bs 32 -dt FP16
```

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

## Project Structure

```
├── main.py              # Single-device entry point
├── main_ddp.py          # DDP multi-GPU entry point
├── main_tpu.py          # TPU entry point
├── calibrate_memory.py  # Memory calibration utility
├── Makefile             # Convenience targets
├── benchmark/
│   ├── Bench.py         # Orchestrator
│   ├── cli.py           # Unified CLI parsing
│   ├── scoring.py       # Scoring system
│   ├── models/          # BenchModel implementations
│   │   ├── base.py      # BenchModel ABC
│   │   ├── cnn.py       # Simple CNN (62K params)
│   │   ├── resnet50.py  # ResNet50 (23.5M params)
│   │   ├── vit.py       # ViT-Base/16 (85.8M params)
│   │   ├── unet.py      # UNet segmentation (31.0M params)
│   │   └── ddpm.py      # DDPM diffusion (29.9M params)
│   ├── devices/         # DeviceBackend implementations
│   │   ├── base.py      # DeviceBackend ABC
│   │   ├── cuda_device.py
│   │   ├── mps_device.py
│   │   ├── npu_device.py
│   │   ├── musa_device.py
│   │   └── tpu_device.py
│   ├── runners/         # Training runners
│   │   ├── common.py    # Shared training utilities
│   │   ├── single_runner.py
│   │   └── ddp_runner.py
│   └── data/            # Dataset utilities
├── scripts/
│   └── manage-data.py   # Benchmark data management
└── docs/                # GitHub Pages dashboard
```

## License

See [LICENSE](LICENSE).

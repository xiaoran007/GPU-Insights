# AGENTS.md — GPU-Insights Coding Agent Context

## Project Overview

GPU-Insights is a **cross-platform GPU training benchmark** that measures compute throughput across multiple deep learning models, precisions, and device backends. It generates a standardised score and throughput (samples/sec) metric.

**Active branch:** `dev/deep-refactor` — major architecture overhaul in progress.
**Main branch:** `main` — legacy codebase, will be replaced once refactor merges.

## Repository Structure

```
GPU-Insights/
├── main.py                  # Single-device entry point
├── main_ddp.py              # DDP (multi-GPU) entry point via torchrun
├── main_tpu.py              # TPU entry point (xla_spawn)
├── Makefile                 # Convenience targets for all modes
├── calibrate_memory.py      # NVML-based memory calibration tool
├── check_env.py             # Environment diagnostic tool
├── helper.py                # Misc utilities
├── macos_hw_detector.py     # Apple hardware detection
│
├── benchmark/               # Core benchmark framework
│   ├── Bench.py             # Orchestrator: wires device + model + runner
│   ├── cli.py               # Unified CLI argument parsing
│   ├── scoring.py           # Score computation and display
│   ├── calibration.py       # Calibration table + auto batch size logic
│   ├── data/                # Synthetic dataset (FakeDataset)
│   ├── models/              # Model specs (see below)
│   ├── devices/             # Device backends (see below)
│   └── runners/             # Training runners (see below)
│
├── docs-src/                # React + Vite + TypeScript dashboard source
│   ├── src/
│   │   ├── components/      # React components (Layout, Charts, Tables, etc.)
│   │   ├── data/            # JSON benchmark result data
│   │   ├── hooks/           # Data loading hooks
│   │   ├── types/           # TypeScript type definitions
│   │   └── utils/           # Helper utilities
│   └── package.json
│
├── docs/                    # Built dashboard (GitHub Pages deployment)
├── scripts/manage-data.py   # Data management for benchmark results
└── .github/workflows/       # CI: pages.yml (GitHub Pages)
```

## Architecture — Three-Pillar Design

The framework uses three orthogonal extension axes, each with an abstract base + registry pattern:

### 1. Models (`benchmark/models/`)

**Base class:** `BenchModel` (ABC) in `models/base.py`

Each model spec defines **what** to benchmark — model architecture, input shape, loss function, dataset, and capability flags. Models are registered in `models/__init__.py` via `register_model()`.

| Model | File | Params | Input | Task |
|-------|------|--------|-------|------|
| CNN | `cnn.py` | 62K | 3×32×32 | Classification (archived baseline) |
| ResNet50 | `resnet50.py` | 23.5M | 3×32×32 | Classification |
| ViT | `vit.py` | 85.8M | 3×224×224 | Classification (ViT-Base/16) |
| UNet | `unet.py` | 31M | 3×256×256 | Segmentation |
| DDPM | `ddpm.py` | 62.3M | 3×64×64 | Diffusion noise prediction |

**Key extension points in BenchModel:**
- `create_model()` — returns `nn.Module`
- `create_dataset()` — synthetic data factory (override for non-classification tasks)
- `compute_loss()` — forward + loss (override for diffusion, segmentation, etc.)
- `get_criterion()` — loss function
- `use_channels_last` — property; True for conv-heavy models, False for ViT
- `supports_ddp`, `supports_amp`, `supports_compile` — capability flags

**To add a new model:** Create a file in `models/`, subclass `BenchModel`, implement all abstract methods, then `register_model(YourModel())` in `models/__init__.py`.

### 2. Device Backends (`benchmark/devices/`)

**Base class:** `DeviceBackend` (ABC) in `devices/base.py`

Each backend handles device-specific operations — detection, synchronisation, AMP, compilation, memory tracking. Backends are registered in `devices/__init__.py` via `register_device()`.

| Backend | File | Notes |
|---------|------|-------|
| CUDA | `cuda_device.py` | Primary target; supports compile, DDP, channels_last, TF32 |
| MPS | `mps_device.py` | Apple Silicon; no channels_last backward, no compile |
| TPU | `tpu_device.py` | Google TPU via torch_xla; custom optimizer_step, dataloader |
| NPU | `npu_device.py` | Huawei Ascend via torch_npu |
| MUSA | `musa_device.py` | Moore Threads via torch_musa |

**Key methods in DeviceBackend:**
- `detect_devices()`, `get_device_name()`, `get_device_memory()`
- `synchronize()` — block until device idle
- `get_autocast_context()`, `get_grad_scaler()` — AMP support
- `supports_channels_last()` — guards memory format optimisation
- `supports_compile()`, `try_compile_model()` — torch.compile support
- `setup_precision()` — backend-specific precision (TF32, matmul precision)
- `optimizer_step()` — overridden by TPU for xm.optimizer_step

**To add a new backend:** Create a file in `devices/`, subclass `DeviceBackend`, implement all abstract methods, then `register_device(YourBackend())` in `devices/__init__.py` (wrap in try/except for optional deps).

### 3. Runners (`benchmark/runners/`)

**Base class:** `BenchRunner` (ABC) in `runners/base.py`

Runners implement the training loop. They are configured by `Bench.py` (the orchestrator) which selects the right runner based on CLI flags.

| Runner | File | Use Case |
|--------|------|----------|
| SingleRunner | `single_runner.py` | Single GPU / DataParallel |
| DDPRunner | `ddp_runner.py` | Distributed Data Parallel (torchrun) |

**Shared logic** lives in `runners/common.py`:
- `train_step()` — one step with AMP handling, delegates to `model_spec.compute_loss()`

**Auto batch size** logic lives in `benchmark/calibration.py`:
- `find_optimal_batch_size()` — calibration-table lookup based on NVML-measured peak VRAM
- `CALIBRATION_TABLE` — maps `(model_name, dtype)` to `[(batch_size, peak_memory_mb), ...]`

**Runner flow:** preload data → warmup → timed training loop → score computation

## Orchestration Flow

```
main.py  →  cli.py (parse args)  →  Bench.__init__()
                                        ├── auto_detect_backend()  →  DeviceBackend
                                        ├── get_model()            →  BenchModel
                                        └── _build_runner()        →  SingleRunner / DDPRunner
                                                                        └── .run()  →  BenchResult
```

## PyTorch Optimisations (Current)

These are applied conditionally based on model + backend capabilities:

1. **channels_last memory format** — applied when `model.use_channels_last and backend.supports_channels_last()` (CUDA only; MPS backward pass fails for BatchNorm models)
2. **cudnn.benchmark** — enabled by default (`Bench.py`)
3. **float32 matmul precision** — set to `'high'` on CUDA (`cuda_device.py`)
4. **Non-blocking data preload** — `non_blocking=True` on `.to(device)` with post-transfer sync
5. **Reduced loss.item() sync** — logged every ~10 steps, not every step
6. **torch.compile** — attempted on CUDA for models with `supports_compile=True`

## CLI Usage

```bash
# Single-device benchmark
python main.py -mt <model> -s <size_mb> -e <epochs> -dt <FP32|FP16|BF16>

# With auto batch size (uses calibration table)
python main.py -mt <model> -s <size_mb> -e <epochs> -abs -dt <type>

# DDP multi-GPU
torchrun --nproc_per_node=N main_ddp.py -mt <model> -s <size_mb> -e <epochs> -dt <type>

# TPU
python main_tpu.py -mt <model> -s <size_mb> -e <epochs> -dt BF16

# Memory calibration (NVML)
python calibrate_memory.py                        # all models, all dtypes
python calibrate_memory.py -mt resnet50 -dt FP16  # targeted
python calibrate_memory.py --json                  # machine-readable output

# Available models: cnn, resnet50, vit, unet, ddpm
# Makefile targets: run, abs, vit, unet, ddpm, ddp, ddp-abs, tpu, tpu-multi, calibrate, docs, docs-dev
```

## Scoring

Defined in `benchmark/scoring.py`. The score formula:

```
score = (total_samples / time_seconds) × (epochs / 10) × 100
```

Primary metric is `throughput` (samples/sec). The `score` field exists for backward compatibility with legacy ResNet50 results.

## Visualization Dashboard

- **Source:** `docs-src/` — React + Vite + TypeScript
- **Build:** `make docs` (output to `docs/`)
- **Dev:** `make docs-dev` (hot reload)
- **Deploy:** GitHub Pages via `.github/workflows/pages.yml`
- **Data:** Benchmark results stored as JSON in `docs-src/src/data/`

## Known Constraints

- **channels_last + MPS:** Backward pass fails for models using BatchNorm (ResNet50, UNet) due to PyTorch MPS backend limitation. DDPM (GroupNorm) works. Guard: `supports_channels_last()` returns False on MPS.
- **Auto Batch Size (ABS):** Based on NVML-calibrated memory profile table in `benchmark/calibration.py`. Requires `pynvml` for calibration. CUDA is the primary target; NPU/MUSA use CUDA data as proxy; MPS/TPU fall back to model defaults.
- **torch.compile + MPS:** Not supported; `MPSDeviceBackend.supports_compile()` returns False.
- **TPU/NPU/MUSA backends:** Functional but less tested than CUDA and MPS.

## Development Notes

- **Python environment:** Requires PyTorch 2.x+. Optional: `torch_xla` (TPU), `torch_npu` (NPU), `torch_musa` (MUSA). `pynvml` for calibration.
- **Testing models locally:** `python main.py -mt <model> -s 16 -e 1 -dt FP32` for a quick smoke test.
- **Registries are import-time:** Models and devices auto-register when their `__init__.py` is imported.
- **Calibration workflow:** Run `python calibrate_memory.py` on a CUDA machine, then paste output into `benchmark/calibration.py` `CALIBRATION_TABLE`.
- **Do not create doc or script files** unless explicitly instructed (see `.github/copilot-instructions.md`).
- **Commit every small changes:** Commit every small changes with meaningful commit message. 

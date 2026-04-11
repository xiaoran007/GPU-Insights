# AGENTS.md — GPU-Insights Coding Agent Context

## Project Overview

GPU-Insights is a **cross-platform GPU training benchmark** that measures compute throughput across multiple deep learning models, precisions, and device backends. It generates a standardised score and throughput (samples/sec) metric.

**Active branch:** `main`
**Main branch:** `main`

## Repository Structure

```
GPU-Insights/
├── main_auto.py             # Smart launcher with auto device / precision / CUDA DDP planning
├── main.py                  # Single-device entry point
├── main_ddp.py              # DDP (multi-GPU) entry point via torchrun
├── main_tpu.py              # TPU entry point (xla_spawn)
├── Makefile                 # Smart launcher and developer convenience targets
├── calibrate_memory.py      # NVML-based memory calibration tool
├── check_env.py             # Environment diagnostic tool
├── helper.py                # Misc utilities
│
├── benchmark/               # Core benchmark framework
│   ├── Bench.py             # Orchestrator: wires device + model + runner
│   ├── cli.py               # Unified CLI argument parsing
│   ├── scoring.py           # Score computation and display
│   ├── calibration.py       # Calibration table + auto batch size logic
│   ├── data/                # Synthetic dataset (FakeDataset)
│   ├── models/              # Model specs (see below)
│   ├── devices/             # Device backends + platform-specific device helpers
│   └── runners/             # Training runners (see below)
│
├── docs-src/                # React + Vite + TypeScript dashboard source
│   ├── src/
│   │   ├── components/      # React components (Layout, Charts, Tables, etc.)
│   │   ├── hooks/           # Data loading hooks
│   │   ├── types/           # TypeScript type definitions
│   │   └── utils/           # Helper utilities
│   └── public/data/         # JSON benchmark result data
│   └── package.json
│
├── docs/                    # Built dashboard (GitHub Pages deployment)
├── scripts/probe_benchmark_env.py  # Normalized host/device metadata probe for launcher result export
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
| ResNet-50 | `resnet50.py` | 23.5M | 3×32×32 | Classification |
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

1. **AMP autocast** — FP16 with GradScaler, BF16 without scaler (`runners/common.py`)
2. **channels_last memory format** — applied when `model.use_channels_last and backend.supports_channels_last()` (CUDA only; MPS backward pass fails for BatchNorm models)
3. **cudnn.benchmark** — off by default in CLI; enable with `-cudnn` flag (`Bench.py`)
4. **TF32 / matmul precision** — `float32_matmul_precision='high'` + `allow_tf32=True` on CUDA CC≥8.0 (`cuda_device.py`)
5. **Non-blocking data preload** — `non_blocking=True` on `.to(device)` for images and labels
6. **Data pre-loading to GPU** — entire dataset transferred before timing starts, eliminates DataLoader overhead from benchmark
7. **Reduced loss.item() sync** — logged ~10 times per epoch, not every step
8. **torch.compile** — attempted on CUDA (`default` mode) and TPU (`openxla`) for models with `supports_compile=True`
9. **Fused SGD optimizer** — `fused=True` on CUDA PyTorch 2.0+ (`cuda_device.py`)
10. **`optimizer.zero_grad(set_to_none=True)`** — avoids zeroing memory, saves allocation
11. **DataLoader tuning** — `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` (PyTorch 2.0+)
12. **DDP gradient contiguity hooks** — ensures contiguous gradients for efficient all-reduce (`ddp_runner.py`)

## CLI Usage

```bash
# Recommended: smart launcher
python3 main_auto.py -mt <model>

# Run the major model set (resnet50, vit, unet, ddpm)
python3 main_auto.py

# Preview smart-launch decisions
python3 main_auto.py -mt <model> --dry-run

# Expert single-device benchmark
python main.py -mt <model> -s <size_mb> -e <epochs> -dt <FP32|FP16|BF16>

# Expert DDP multi-GPU benchmark
torchrun --nproc_per_node=N main_ddp.py -mt <model> -s <size_mb> -e <epochs> -dt <type>

# TPU benchmark
python main_tpu.py -mt <model> -s <size_mb> -e <epochs> -dt BF16

# Memory calibration (NVML)
python calibrate_memory.py
python calibrate_memory.py -mt resnet50 -dt FP16
python calibrate_memory.py --json

# Available models: cnn, resnet50, vit, unet, ddpm
# Makefile targets: smart, tpu, tpu-multi, calibrate, docs, docs-dev, help
```

`main_auto.py` behavior notes:
- `--model` is optional; omitting it runs `resnet50`, `vit`, `unet`, and `ddpm` in order
- the launcher prints a final `RESULT_PAYLOAD_B64=...` line after the summary for script-friendly result export
- payload benchmark entries are normalized to the current dashboard schema, with BF16 intentionally mapped into the existing `fp16` / `fp16bs` fields

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
- **Data:** Benchmark results stored as JSON in `docs-src/public/data/benchmark-data.json`

## Known Constraints

- **channels_last + MPS:** Backward pass fails for models using BatchNorm (ResNet50, UNet) due to PyTorch MPS backend limitation. DDPM (GroupNorm) works. Guard: `supports_channels_last()` returns False on MPS.
- **Auto Batch Size (ABS):** Based on NVML-calibrated memory profile table in `benchmark/calibration.py`. Requires `pynvml` for calibration. CUDA is the primary target; NPU/MUSA use CUDA data as proxy; MPS/TPU fall back to model defaults.
- **Launcher metadata probe:** `scripts/probe_benchmark_env.py` is the canonical source for exported host/device metadata. On NVIDIA systems it prefers NVML for device facts and maps CUDA compute capability to architecture names explicitly.
- **torch.compile + MPS:** Not supported; `MPSDeviceBackend.supports_compile()` returns False.
- **TPU/NPU/MUSA backends:** Functional but less tested than CUDA and MPS.

## Development Notes

- **Python environment:** Requires PyTorch 2.x+. Optional: `torch_xla` (TPU), `torch_npu` (NPU), `torch_musa` (MUSA). `pynvml` for calibration.
- **Preferred entrypoint:** `python3 main_auto.py -mt <model>` for the default benchmark workflow.
- **Batch benchmark workflow:** `python3 main_auto.py` is the default way to run the major model set and capture one copy-pasteable result payload for downstream JSON update scripts.
- **Testing models locally:** `python main.py -mt <model> -s 16 -e 1 -dt FP32` for a quick smoke test.
- **Registries are import-time:** Models and devices auto-register when their `__init__.py` is imported.
- **Calibration workflow:** Run `python calibrate_memory.py` on a CUDA machine, then paste output into `benchmark/calibration.py` `CALIBRATION_TABLE`.
- **Do not create doc or script files** unless explicitly instructed (see `.github/copilot-instructions.md`).
- **Commit every small changes:** Commit every small changes with meaningful commit message. 

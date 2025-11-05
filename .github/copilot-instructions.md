# GPU-Insights AI Agent Guide

## Project Overview
GPU-Insights is a cross-platform GPU benchmarking suite focused on measuring training performance across diverse GPU vendors (NVIDIA, AMD, Intel, Apple, Huawei, Mthreads). The project generates standardized performance scores using ResNet50 and CNN models with different precision modes (FP32, FP16, BF16).

## Architecture & Data Flow

### Core Components
1. **Entry Point** (`main.py`) → Command-line interface with argparse
2. **Hardware Detection** (`macos_hw_detector.py`) → macOS-specific GPU detection via `system_profiler`
3. **Benchmark Orchestration** (`benchmark/Bench.py`) → Device selection, memory calculation, benchmark dispatch
4. **Model Implementations** (`benchmark/bench/`) → Actual training loops with device-specific optimizations
5. **Data Management** (`scripts/manage-data.py`) → JSON validation, CRUD operations for benchmark results
6. **Web Frontend** (`docs/index.html`) → Chart.js visualization consuming `docs/data/benchmark-data.json`

### Multi-Vendor Device Support Pattern
The project uses **conditional imports with fallback chains** to support diverse accelerators:

```python
# Pattern used in benchmark/Bench.py and resnet50_bench.py
if self.huawei:
    import torch_npu  # Huawei Ascend NPUs
elif self.mthreads:
    import torch_musa  # Mthreads GPUs
elif torch.cuda.is_available():
    # NVIDIA/AMD via CUDA/ROCm
elif torch.backends.mps.is_available():
    # Apple Metal Performance Shaders
elif torch.xpu.is_available():
    # Intel Arc/Data Center GPUs
```

**Critical**: When adding device support, maintain this fallback order. Each vendor requires specific synchronization calls (see `_bench()` methods).

## Key Development Workflows

### Running Benchmarks
```bash
# Standard ResNet50 benchmark (matches README defaults)
python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32 -gpu 0

# Multi-GPU example (comma-separated IDs)
python main.py -m -mt resnet50 -bs 1024 -dt FP16 -gpu 0,1

# Quick test via Makefile
make run  # Runs FP16 then FP32 with preset configs
```

**Memory Sizing Logic**: The `-s` flag sets CUDA memory in MB, converted to image count via `data_size = int((size * 1024 * 1024 / 12296) / 1)` (12296 bytes per image). Auto mode (`-a`) calculates 70% of available VRAM but is currently disabled.

### Managing Benchmark Data
Results are stored in `docs/data/benchmark-data.json` with a strict schema validated by `manage-data.py`:

```bash
# Workflow: Edit data.sh → Run update.sh → Auto-validates and updates JSON
source ./data.sh && bash ./update.sh

# Direct validation
python scripts/manage-data.py validate

# View statistics
python scripts/manage-data.py stats
```

**Schema Requirements** (from `manage-data.py:validate_entry`):
- Required fields: `vendor`, `architecture`, `device`, `memory`, `platform`, `fp32`, `fp32bs`, `fp16`, `fp16bs`, `note`, `date`
- Valid vendors: `{"nvidia", "amd", "intel", "apple", "huawei", "mthreads"}`
- Date format: `YYYY.M.DD` (e.g., "2025.3.20")
- Scores can be `null` for unsupported precision modes

## Project-Specific Conventions

### Score Calculation Formula
```python
# From cnn_bench.py and resnet50_bench.py
basic_score = self.data_size / time_usage  # Images per second
final_score = basic_score * (self.epochs / 10)  # Normalized to 10-epoch baseline
```
Higher scores indicate better performance. Score reflects both compute throughput and memory bandwidth.

### Mixed Precision Implementation
- **FP16/BF16** require `GradScaler` for loss scaling (except BF16 which has wider dynamic range)
- **Device compatibility**: `GradScaler(device=GS_dev)` where `GS_dev = "cuda"` for XPU/NPU, else actual device type
- **AMP Import Strategy**: Try `torch.amp` (PyTorch 2.0+) → fallback to `torch_npu.npu.amp` → fallback to `torch.cuda.amp`

### Multi-GPU Strategy
- Uses `nn.DataParallel` when `len(gpu_ids) > 1` (see `resnet50_bench.py:_bench()`)
- **Batch size scaling**: README recommends BS=2048 (FP16) / BS=1024 (FP32) for multi-GPU
- Only ResNet50 supports multi-GPU; CNN bench uses single device

### Warmup & Timing Protocol
All benchmarks follow this pattern (see `cnn_bench.py:48-63`):
1. Preload all data to target device
2. Run 5 warmup batches (or fewer if dataset is small)
3. Call device-specific `synchronize()` before starting timer
4. Execute training loop
5. Call `synchronize()` again before stopping timer

**Why**: Eliminates cold start effects and ensures accurate GPU timing by blocking until all kernels complete.

## Critical Files & Their Roles

- **`benchmark/Bench.py`**: Device detection logic, memory sizing, backend selection
- **`benchmark/bench/resnet50_bench.py`**: Primary benchmark (23.5M params), handles AMP fallbacks
- **`main.py`**: CLI parsing, model/dtype validation, GPU ID parsing
- **`data.sh`**: Staging file for new benchmark results (edit this, not JSON directly)
- **`scripts/manage-data.py`**: Data validation, prevents malformed entries in JSON
- **`docs/data/benchmark-data.json`**: Source of truth for web visualization

## Common Pitfalls

1. **Adding Results**: Never manually edit `benchmark-data.json` - use `data.sh` + `update.sh` workflow to ensure validation
2. **New Device Support**: Must add vendor to `VALID_VENDORS` in `manage-data.py` and implement device detection in `Bench._get_gpu_device()`
3. **Batch Size**: Default is model-specific (CNN=2048, ResNet50=4). Use `-bs 0` to trigger defaults, not omitting the flag
4. **Date Format**: Use `YYYY.M.DD` not `YYYY.MM.DD` (no zero-padding for single-digit months/days)
5. **GradScaler Compatibility**: Some vendors (Intel Arc) have non-functional GradScaler - check "Note" field in results table

## Testing & Validation

```bash
# Validate data integrity before committing
python scripts/manage-data.py validate

# Quick sanity check (2 epochs, small memory)
python main.py -m -s 256 -e 2 -mt resnet50 -bs 128 -dt FP32 -gpu 0
```

## Extending the Project

### Adding a New Model
1. Create `benchmark/bench/{model_name}_bench.py` implementing `start()` and `_bench(device)` methods
2. Add model selection logic in `Bench._load_backend()`
3. Update `main.py` argparse choices and validation
4. Define default batch size in model constructor

### Adding Vendor Support
1. Add conditional import block in `Bench._get_gpu_device()`
2. Add synchronization call in benchmark `_bench()` methods
3. Update `VALID_VENDORS` set in `scripts/manage-data.py`
4. Document any vendor-specific quirks in README results table

---

**Website**: Results visualized at https://xiaoran007.github.io/GPU-Insights/

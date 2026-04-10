"""NVML-based GPU memory calibration tool.

Measures **real** peak VRAM usage (via NVML / ``pynvml``) for each
``(model, dtype, batch_size)`` combination.  The results are used to
populate ``benchmark/calibration.py:CALIBRATION_TABLE`` so that the
auto-batch-size logic can pick a safe batch size on any CUDA device.

Workflow
--------
1. Run this script on a CUDA machine::

       python calibrate_memory.py                        # full sweep
       python calibrate_memory.py -mt resnet50 -dt FP16  # targeted
       python calibrate_memory.py --json                  # machine-readable

2. Copy the output into ``benchmark/calibration.py`` ``CALIBRATION_TABLE``.

Prerequisites
-------------
``pip install pynvml``  (ships with most NVIDIA driver installs).
"""

import argparse
import json
import sys
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.models import get_model, list_models
from benchmark.devices import auto_detect_backend
from benchmark.runners.common import train_step


# ---------------------------------------------------------------------------
# NVML helpers
# ---------------------------------------------------------------------------

def nvml_init() -> bool:
    """Initialise NVML.  Returns True on success."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"Error: pynvml is required for NVML calibration.  Install with:")
        print(f"  pip install pynvml")
        print(f"  (Underlying error: {e})")
        return False


def nvml_get_handle(gpu_id: int):
    import pynvml
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)


def nvml_get_used_mb(handle) -> float:
    """Return current GPU memory usage in MB as reported by NVML."""
    import pynvml
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 * 1024)


def nvml_get_total_mb(handle) -> float:
    import pynvml
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.total / (1024 * 1024)


def nvml_get_device_name(handle) -> str:
    import pynvml
    return pynvml.nvmlDeviceGetName(handle)


# ---------------------------------------------------------------------------
# Default batch-size ranges per model
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZES: Dict[str, List[int]] = {
    "cnn":      [256, 512, 1024, 2048, 4096],
    "resnet50": [32, 64, 128, 256, 512, 1024],
    "vit":      [8, 16, 32, 64, 128, 256],
    "unet":     [2, 4, 8, 16, 32, 64],
    "ddpm":     [8, 16, 32, 64, 128, 256],
}


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def calibrate_one(
    model_name: str,
    data_type: str,
    batch_size: int,
    gpu_id: int,
    nvml_handle,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> Optional[float]:
    """Run a short training burst and return peak NVML VRAM in MB.

    Returns ``None`` on OOM or other fatal error.
    """
    device = torch.device(f"cuda:{gpu_id}")
    backend = auto_detect_backend(device="cuda")
    if backend is None:
        return None

    model_spec = get_model(model_name)
    use_fp16 = (data_type == "FP16")
    use_bf16 = (data_type == "BF16")
    num_classes = model_spec.get_num_classes()

    try:
        # --- Setup ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        model = model_spec.create_model(num_classes=num_classes).to(device)
        if model_spec.use_channels_last and backend.supports_channels_last():
            model = model.to(memory_format=torch.channels_last)

        criterion = model_spec.get_criterion()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        if use_bf16:
            scaler = None
        elif use_fp16:
            scaler = backend.get_grad_scaler(enabled=True)
        else:
            scaler = backend.get_grad_scaler(enabled=False)

        # --- Synthetic data ---
        data_size = max(batch_size * (warmup_steps + measure_steps + 2), 256)
        train_dataset = model_spec.create_dataset(data_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        cl = model_spec.use_channels_last and backend.supports_channels_last()
        data_preloaded = [
            (
                images.to(device, memory_format=torch.channels_last, non_blocking=True)
                if cl else images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            for images, labels in train_loader
        ]
        torch.cuda.synchronize(device)

        if len(data_preloaded) < warmup_steps + measure_steps:
            # Not enough batches; unlikely but guard against it
            return None

        # --- Warmup ---
        model.train()
        for i in range(warmup_steps):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            train_step(
                model_spec, backend, model, images, labels,
                criterion, optimizer, scaler, use_fp16, use_bf16, device,
            )
        torch.cuda.synchronize(device)

        # --- Measurement (NVML peak) ---
        peak_nvml_mb = nvml_get_used_mb(nvml_handle)
        for i in range(warmup_steps, warmup_steps + measure_steps):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            train_step(
                model_spec, backend, model, images, labels,
                criterion, optimizer, scaler, use_fp16, use_bf16, device,
            )
            torch.cuda.synchronize(device)
            current = nvml_get_used_mb(nvml_handle)
            peak_nvml_mb = max(peak_nvml_mb, current)

        # --- Cleanup ---
        del model, optimizer, scaler, criterion, data_preloaded, train_loader, train_dataset
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        return peak_nvml_mb

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NVML-based GPU memory calibration for auto batch size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python calibrate_memory.py                        # all models, all dtypes\n"
            "  python calibrate_memory.py -mt resnet50 -dt FP16  # single combo\n"
            "  python calibrate_memory.py -bs 16,32,64,128       # custom batch sizes\n"
            "  python calibrate_memory.py --json                  # JSON output\n"
        ),
    )
    parser.add_argument(
        "-gpu", type=int, default=0,
        help="CUDA GPU index (default: 0).",
    )
    parser.add_argument(
        "-mt", "--model", type=str, default=None,
        help=f"Model to calibrate.  Available: {', '.join(list_models())}.  Default: all.",
    )
    parser.add_argument(
        "-dt", "--data_type", type=str, default=None,
        help="Precision: FP32, FP16, BF16.  Default: all three.",
    )
    parser.add_argument(
        "-bs", "--batch_sizes", type=str, default=None,
        help="Comma-separated list of batch sizes to test (e.g. 16,32,64,128).",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Output results in JSON format.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup steps per measurement (default: 3).",
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of measurement steps per batch size (default: 10).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_id = args.gpu

    # --- CUDA check ---
    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        sys.exit(1)

    # --- NVML init ---
    if not nvml_init():
        sys.exit(1)

    handle = nvml_get_handle(gpu_id)
    gpu_name = nvml_get_device_name(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    total_mb = nvml_get_total_mb(handle)

    # --- Resolve targets ---
    models = [args.model] if args.model else list_models()
    dtypes = [args.data_type.upper()] if args.data_type else ["FP32", "FP16", "BF16"]

    # BF16 availability check
    cuda_device = torch.device(f"cuda:{gpu_id}")
    bf16_ok = True
    try:
        bf16_ok = torch.cuda.is_bf16_supported(including_emulation=False)
    except Exception:
        bf16_ok = False

    # --- Header ---
    if not args.json:
        print(f"\n{'='*55}")
        print(f"  NVML Memory Calibration")
        print(f"{'='*55}")
        print(f"  GPU:           {gpu_name}")
        print(f"  Total VRAM:    {total_mb:.0f} MB")
        print(f"  Models:        {', '.join(models)}")
        print(f"  Precisions:    {', '.join(dtypes)}")
        print(f"  Warmup steps:  {args.warmup}")
        print(f"  Measure steps: {args.steps}")
        print(f"{'='*55}\n")

    # --- Run calibration ---
    results: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for model_name in models:
        results[model_name] = {}
        for dt in dtypes:
            if dt == "BF16" and not bf16_ok:
                if not args.json:
                    print(f"  [{model_name}/{dt}] Skipped — BF16 not supported on this GPU.")
                continue

            batch_sizes = (
                [int(x) for x in args.batch_sizes.split(",")]
                if args.batch_sizes
                else DEFAULT_BATCH_SIZES.get(model_name, [32, 64, 128, 256])
            )

            entries = []
            if not args.json:
                print(f"  [{model_name}/{dt}]")

            for bs in sorted(batch_sizes):
                if not args.json:
                    print(f"    bs={bs:<6} ", end="", flush=True)
                peak = calibrate_one(
                    model_name, dt, bs, gpu_id, handle,
                    warmup_steps=args.warmup,
                    measure_steps=args.steps,
                )
                if peak is None:
                    if not args.json:
                        print("OOM — skipping larger sizes.")
                    break
                entries.append((bs, round(peak, 1)))
                if not args.json:
                    print(f"peak = {peak:.1f} MB")

            results[model_name][dt] = entries
            if not args.json:
                print()

    # --- Output ---
    if args.json:
        json_out = {
            "gpu": gpu_name,
            "total_memory_mb": round(total_mb, 1),
            "results": {
                model: {
                    dt: [[bs, mem] for bs, mem in entries]
                    for dt, entries in dt_data.items()
                }
                for model, dt_data in results.items()
            },
        }
        print(json.dumps(json_out, indent=2))
    else:
        # Print copy-paste-ready Python dict for calibration.py
        print(f"\n{'='*55}")
        print(f"  Calibration Complete")
        print(f"{'='*55}")
        print(f"\nPaste the following into benchmark/calibration.py CALIBRATION_TABLE:\n")
        print("CALIBRATION_TABLE = {")
        for model_name, dt_data in results.items():
            for dt, entries in dt_data.items():
                if entries:
                    items = ", ".join(f"({bs}, {mem})" for bs, mem in entries)
                    print(f'    ("{model_name}", "{dt}"): [{items}],')
        print("}")
        print()

    # --- Cleanup NVML ---
    import pynvml
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

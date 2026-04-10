"""Calibration-based auto batch size selection.

This module holds the NVML-calibrated memory profile table and the
``find_optimal_batch_size()`` function that runners call when ``-abs`` is
enabled.

**How it works:**

1. ``CALIBRATION_TABLE`` maps ``(model_name, dtype_str)`` to a list of
   ``(batch_size, peak_vram_mb)`` tuples sorted by batch_size ascending.
   The data is obtained by running ``calibrate_memory.py`` on a CUDA device.

2. ``find_optimal_batch_size()`` reads the device's total memory, applies a
   safety margin, then picks the largest batch size whose measured peak VRAM
   fits within the usable budget.

3. If no calibration data exists for the requested model/dtype, the function
   falls back to ``model_spec.get_default_batch_size()``.

To regenerate calibration data run::

    python calibrate_memory.py --json
"""

from typing import Dict, List, Optional, Tuple

from benchmark.devices.base import DeviceBackend
from benchmark.models.base import BenchModel

# Safety factor: use at most this fraction of total device memory.
# 0.90 means 10% is reserved for OS, display, CUDA context, etc.
SAFETY_FACTOR = 0.90

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
# Mapping of (model_name, dtype) → [(batch_size, peak_vram_mb), ...]
# Sorted by batch_size ascending.  Populated by running calibrate_memory.py.
#
# Reference GPU: NVIDIA A100-SXM4-40GB (40960 MB).
# To recalibrate for a different GPU: python calibrate_memory.py
# ---------------------------------------------------------------------------

CALIBRATION_TABLE: Dict[Tuple[str, str], List[Tuple[int, float]]] = {
    # Calibrated on NVIDIA A100-SXM4-40GB via calibrate_memory.py (pynvml).
    # Each entry: (batch_size, peak_vram_mb) — sorted ascending by batch_size.
    ("cnn", "FP32"):      [(256, 1148.6), (512, 1208.6), (1024, 1380.6), (2048, 1682.6), (4096, 2230.6)],
    ("cnn", "FP16"):      [(256, 1140.6), (512, 1198.6), (1024, 1312.6), (2048, 1544.6), (4096, 1996.6)],
    ("cnn", "BF16"):      [(256, 1148.6), (512, 1206.6), (1024, 1350.6), (2048, 1608.6), (4096, 2118.6)],
    ("resnet50", "FP32"): [(32, 2120.6), (64, 3036.6), (128, 4874.6), (256, 8406.6), (512, 15400.6), (1024, 29478.6)],
    ("resnet50", "FP16"): [(32, 1664.6), (64, 2138.6), (128, 3036.6), (256, 4894.6), (512, 8542.6), (1024, 15812.6)],
    ("resnet50", "BF16"): [(32, 1666.6), (64, 2140.6), (128, 3038.6), (256, 4896.6), (512, 8544.6), (1024, 15814.6)],
    ("vit", "FP32"):      [(8, 2710.6), (16, 3738.6), (32, 5882.6), (64, 9980.6), (128, 18294.6), (256, 35066.6)],
    ("vit", "FP16"):      [(8, 2398.6), (16, 3030.6), (32, 4268.6), (64, 6698.6), (128, 11406.6), (256, 20874.6)],
    ("vit", "BF16"):      [(8, 2398.6), (16, 3030.6), (32, 4268.6), (64, 6698.6), (128, 11406.6), (256, 20874.6)],
    ("unet", "FP32"):     [(2, 2496.6), (4, 3414.6), (8, 5306.6), (16, 9134.6), (32, 16946.6), (64, 32570.6)],
    ("unet", "FP16"):     [(2, 2084.6), (4, 2502.6), (8, 3394.6), (16, 5214.6), (32, 9080.6), (64, 16842.6)],
    ("unet", "BF16"):     [(2, 2084.6), (4, 2502.6), (8, 3398.6), (16, 5216.6), (32, 9082.6), (64, 16844.6)],
    ("ddpm", "FP32"):     [(8, 2786.6), (16, 4176.6), (32, 6942.6), (64, 12430.6), (128, 23454.6), (256, 39584.6)],
    ("ddpm", "FP16"):     [(8, 2526.6), (16, 3608.6), (32, 5528.6), (64, 9630.6), (128, 17812.6), (256, 34086.6)],
    ("ddpm", "BF16"):     [(8, 2526.6), (16, 3608.6), (32, 5528.6), (64, 9630.6), (128, 17820.6), (256, 34088.6)],
}


def find_optimal_batch_size(
    model_spec: BenchModel,
    backend: DeviceBackend,
    device,
    data_type: str,
    is_main_process: bool = True,
) -> int:
    """Select the largest safe batch size based on calibration data.

    Parameters
    ----------
    model_spec : BenchModel
        The model being benchmarked (used for name + fallback default).
    backend : DeviceBackend
        Active device backend (used to query total memory).
    device : torch.device
        Target device.
    data_type : str
        One of ``"FP32"``, ``"FP16"``, ``"BF16"``.
    is_main_process : bool
        If ``True``, print diagnostic logs.

    Returns
    -------
    int
        The chosen batch size.
    """
    model_name = model_spec.name
    key = (model_name, data_type)

    # --- Determine usable memory ---
    total_bytes = backend.get_device_memory(device)
    total_mb = total_bytes / (1024 * 1024) if total_bytes > 0 else 0
    usable_mb = total_mb * SAFETY_FACTOR

    if is_main_process:
        print(f"  Device:         {backend.get_device_name(device)}")
        print(f"  Total VRAM:     {total_mb:.0f} MB")
        print(f"  Safety factor:  {SAFETY_FACTOR:.0%} → usable {usable_mb:.0f} MB")

    # --- Look up calibration data ---
    entries = CALIBRATION_TABLE.get(key)

    # For non-CUDA backends (NPU, MUSA) try CUDA calibration data as proxy
    if entries is None and backend.name in ("npu", "musa"):
        entries = CALIBRATION_TABLE.get(key)  # same key — data is device-agnostic
        if entries is None and is_main_process:
            print(f"  No calibration data for ({model_name}, {data_type}) on {backend.name}; "
                  f"attempting CUDA reference data.")
        # The table stores CUDA results; they serve as a conservative proxy.

    if entries is None or len(entries) == 0:
        fallback_bs = model_spec.get_default_batch_size(data_type)
        if is_main_process:
            print(f"  Calibration:    no data for ({model_name}, {data_type})")
            print(f"  Fallback:       using model default batch size = {fallback_bs}")
        return fallback_bs

    if total_mb <= 0:
        fallback_bs = model_spec.get_default_batch_size(data_type)
        if is_main_process:
            print(f"  Warning:        cannot query device memory for {backend.name}")
            print(f"  Fallback:       using model default batch size = {fallback_bs}")
        return fallback_bs

    # --- Pick the largest batch size that fits ---
    # entries are sorted ascending by batch_size; scan from largest.
    chosen_bs = None
    chosen_peak = None
    for bs, peak_mb in reversed(entries):
        if peak_mb <= usable_mb:
            chosen_bs = bs
            chosen_peak = peak_mb
            break

    if chosen_bs is not None:
        if is_main_process:
            print(f"  Calibration:    matched ({model_name}, {data_type})")
            print(f"  Selected:       batch_size={chosen_bs}  (peak {chosen_peak:.0f} MB ≤ {usable_mb:.0f} MB)")
        return chosen_bs

    # Even the smallest calibrated batch size doesn't fit
    smallest_bs, smallest_peak = entries[0]
    fallback_bs = model_spec.get_default_batch_size(data_type)
    if is_main_process:
        print(f"  Calibration:    smallest entry bs={smallest_bs} needs {smallest_peak:.0f} MB "
              f"> usable {usable_mb:.0f} MB")
        print(f"  Fallback:       using model default batch size = {fallback_bs}")
    return fallback_bs

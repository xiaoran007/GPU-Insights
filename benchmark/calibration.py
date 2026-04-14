"""Calibration-based auto batch size selection.

This module holds the NVML-calibrated memory profile table and the
``find_optimal_batch_size()`` function that runners call when ``-abs`` is
enabled.

**How it works:**

1. ``CALIBRATION_TABLE`` maps ``(model_name, dtype_str)`` to a list of
   ``(batch_size, peak_vram_mb)`` tuples sorted by batch_size ascending.
   The stored values are a conservative envelope built from calibration runs.

2. ``find_optimal_batch_size()`` reads the device's total memory, applies a
   safety margin, then scans the calibrated matrix from large to small.

3. If the usable memory falls outside the calibrated matrix, the function
   estimates only additional powers-of-two batch sizes using a conservative
   linear upper bound derived from the existing points.

4. If no calibration data exists for the requested model/dtype, the function
   falls back to ``model_spec.get_default_batch_size()``.

To regenerate calibration data run::

    python calibrate_memory.py --json
"""

from typing import Dict, List, Tuple

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
# Conservative envelope built from NVIDIA A100-SXM4-40GB and H100 runs.
# To refresh the envelope, rerun calibration and update the maxima here.
# ---------------------------------------------------------------------------

CALIBRATION_TABLE: Dict[Tuple[str, str], List[Tuple[int, float]]] = {
    # Each entry: (batch_size, peak_vram_mb) — sorted ascending by batch_size.
    ("cnn", "FP32"):      [(256, 1425.1), (512, 1485.1), (1024, 1597.1), (2048, 1891.1), (4096, 2411.1)],
    ("cnn", "FP16"):      [(256, 1409.1), (512, 1485.1), (1024, 1613.1), (2048, 1877.1), (4096, 2389.1)],
    ("cnn", "BF16"):      [(256, 1409.1), (512, 1485.1), (1024, 1613.1), (2048, 1875.1), (4096, 2391.1)],
    ("resnet50", "FP32"): [(32, 2351.1), (64, 3253.1), (128, 5045.1), (256, 8563.1), (512, 15689.1), (1024, 30657.1)],
    ("resnet50", "FP16"): [(32, 1923.1), (64, 2397.1), (128, 3295.1), (256, 5153.1), (512, 8797.1), (1024, 16087.1)],
    ("resnet50", "BF16"): [(32, 1925.1), (64, 2399.1), (128, 3295.1), (256, 5155.1), (512, 8801.1), (1024, 16077.1)],
    ("vit", "FP32"):      [(8, 2969.1), (16, 3995.1), (32, 6175.1), (64, 10239.1), (128, 18585.1), (256, 35323.1)],
    ("vit", "FP16"):      [(8, 2657.1), (16, 3289.1), (32, 4561.1), (64, 6997.1), (128, 11673.1), (256, 21117.1)],
    ("vit", "BF16"):      [(8, 2659.1), (16, 3291.1), (32, 4563.1), (64, 6999.1), (128, 11673.1), (256, 21117.1)],
    ("unet", "FP32"):     [(2, 2871.1), (4, 3845.1), (8, 5953.1), (16, 10965.1), (32, 18677.1), (64, 35773.1)],
    ("unet", "FP16"):     [(2, 2343.1), (4, 2763.1), (8, 3659.1), (16, 5471.1), (32, 9341.1), (64, 17103.1)],
    ("unet", "BF16"):     [(2, 2343.1), (4, 2763.1), (8, 3659.1), (16, 5471.1), (32, 9343.1), (64, 17105.1)],
    ("ddpm", "FP32"):     [(8, 3307.1), (16, 5127.1), (32, 8317.1), (64, 15009.1), (128, 28285.1), (256, 54799.1)],
    ("ddpm", "FP16"):     [(8, 2791.1), (16, 3889.1), (32, 5787.1), (64, 9891.1), (128, 18077.1), (256, 34345.1)],
    ("ddpm", "BF16"):     [(8, 2793.1), (16, 3891.1), (32, 5791.1), (64, 9893.1), (128, 18079.1), (256, 34349.1)],
}


def _build_conservative_linear_upper_bound(
    entries: List[Tuple[int, float]],
) -> Tuple[float, float]:
    """Return ``(slope, base)`` for a conservative linear upper bound.

    The resulting line satisfies ``base + slope * batch_size >= peak_mb`` for
    every calibrated point in *entries*.
    """
    if len(entries) == 1:
        _, peak_mb = entries[0]
        return 0.0, peak_mb

    slopes = []
    for (bs1, peak1), (bs2, peak2) in zip(entries, entries[1:]):
        if bs2 > bs1:
            slopes.append((peak2 - peak1) / (bs2 - bs1))

    slope = max(slopes) if slopes else 0.0
    base = max(peak_mb - slope * batch_size for batch_size, peak_mb in entries)
    return slope, base


def _estimate_peak_memory_mb(
    entries: List[Tuple[int, float]],
    batch_size: int,
) -> float:
    """Estimate peak memory for out-of-matrix powers-of-two batch sizes."""
    slope, base = _build_conservative_linear_upper_bound(entries)
    return base + slope * batch_size


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

    # --- Pick the largest calibrated batch size that fits ---
    # entries are sorted ascending by batch_size; scan from largest.
    chosen_bs = None
    chosen_peak = None
    for bs, peak_mb in reversed(entries):
        if peak_mb <= usable_mb:
            chosen_bs = bs
            chosen_peak = peak_mb
            break

    if chosen_bs is not None:
        largest_bs, largest_peak = entries[-1]
        if chosen_bs == largest_bs:
            next_bs = largest_bs * 2
            next_peak = _estimate_peak_memory_mb(entries, next_bs)
            while next_peak <= usable_mb:
                chosen_bs = next_bs
                chosen_peak = next_peak
                next_bs *= 2
                next_peak = _estimate_peak_memory_mb(entries, next_bs)

        if is_main_process:
            print(f"  Calibration:    matched ({model_name}, {data_type})")
            if chosen_bs > largest_bs:
                print(f"  Estimation:     expanded beyond calibrated max bs={largest_bs}")
                print(f"  Selected:       batch_size={chosen_bs}  (estimated peak {chosen_peak:.0f} MB ≤ {usable_mb:.0f} MB)")
            else:
                print(f"  Selected:       batch_size={chosen_bs}  (peak {chosen_peak:.0f} MB ≤ {usable_mb:.0f} MB)")
        return chosen_bs

    # Even the smallest calibrated batch size doesn't fit. Keep halving in
    # powers of two using the same conservative upper bound.
    smallest_bs, smallest_peak = entries[0]
    estimated_bs = max(1, smallest_bs // 2)
    estimated_peak = _estimate_peak_memory_mb(entries, estimated_bs)
    while estimated_bs > 1 and estimated_peak > usable_mb:
        estimated_bs = max(1, estimated_bs // 2)
        estimated_peak = _estimate_peak_memory_mb(entries, estimated_bs)

    if is_main_process:
        print(f"  Calibration:    smallest entry bs={smallest_bs} needs {smallest_peak:.0f} MB "
              f"> usable {usable_mb:.0f} MB")
        print(f"  Estimation:     falling back to lower powers of two")
        print(f"  Selected:       batch_size={estimated_bs}  (estimated peak {estimated_peak:.0f} MB)")
    return estimated_bs

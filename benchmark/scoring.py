"""Unified scoring system for GPU-Insights benchmarks."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BenchScore:
    """Multi-dimensional benchmark score."""

    # Core throughput metric
    throughput: float = 0.0          # samples/sec

    # Normalised score (compatible with legacy ResNet50 formula)
    score: float = 0.0

    # Timing breakdown
    time_total: float = 0.0          # total training time (seconds)
    time_per_epoch: float = 0.0      # average seconds per epoch
    time_per_step: float = 0.0       # average seconds per step

    # Memory (0 if unavailable)
    peak_memory_mb: float = 0.0

    # Context
    total_samples: int = 0
    batch_size: int = 0
    epochs: int = 0
    num_steps: int = 0
    device_name: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)


def compute_score(
    total_samples: int,
    time_usage: float,
    epochs: int,
    num_steps: int = 0,
    batch_size: int = 0,
    peak_memory_mb: float = 0.0,
    device_name: str = "",
) -> BenchScore:
    """Compute a standardised BenchScore from raw benchmark data.

    The ``score`` field follows the ResNet50 legacy formula so that old and
    new results remain roughly comparable:

        score = (total_samples / time_usage) * (epochs / 10) * 100
    """
    if time_usage <= 0:
        time_usage = 1e-9  # avoid division by zero

    throughput = total_samples / time_usage
    score = throughput * (epochs / 10) * 100

    time_per_epoch = time_usage / epochs if epochs > 0 else 0
    time_per_step = time_usage / num_steps if num_steps > 0 else 0

    return BenchScore(
        throughput=throughput,
        score=score,
        time_total=time_usage,
        time_per_epoch=time_per_epoch,
        time_per_step=time_per_step,
        peak_memory_mb=peak_memory_mb,
        total_samples=total_samples,
        batch_size=batch_size,
        epochs=epochs,
        num_steps=num_steps,
        device_name=device_name,
    )


def print_score(score: BenchScore, is_main_process: bool = True) -> None:
    """Pretty-print a BenchScore to stdout."""
    if not is_main_process:
        return

    print(f"\n{'='*50}")
    print(f"  Benchmark Results")
    print(f"{'='*50}")
    print(f"  Device:            {score.device_name}")
    print(f"  Score:             {score.score:.0f}")
    print(f"  Throughput:        {score.throughput:.1f} samples/sec")
    print(f"  Total Time:        {score.time_total:.2f} s")
    print(f"  Time/Epoch:        {score.time_per_epoch:.2f} s")
    if score.time_per_step > 0:
        print(f"  Time/Step:         {score.time_per_step * 1000:.1f} ms")
    if score.peak_memory_mb > 0:
        print(f"  Peak Memory:       {score.peak_memory_mb:.0f} MB")
    print(f"  Epochs:            {score.epochs}")
    print(f"  Batch Size:        {score.batch_size}")
    print(f"  Total Samples:     {score.total_samples}")
    print(f"{'='*50}\n")

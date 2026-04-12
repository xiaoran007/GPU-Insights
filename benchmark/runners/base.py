from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class BenchResult:
    """Benchmark result container."""
    time_usage: float = 0.0
    score: int = 0
    throughput: float = 0.0
    data_size: int = 0
    total_data_size: int = 0
    batch_size: int = 0
    epochs: int = 0
    device_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class BenchRunner(ABC):
    """Abstract base class for benchmark runners (single-GPU, DDP, etc.)."""

    @abstractmethod
    def run(self) -> BenchResult:
        """Execute the benchmark and return results."""
        ...

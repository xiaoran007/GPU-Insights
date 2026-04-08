"""Unified CLI argument parsing for GPU-Insights benchmark entry points."""

import argparse
from typing import Tuple

from benchmark.models import resolve_model_name, list_models


def build_common_parser(description: str = "GPU-Insights Benchmark") -> argparse.ArgumentParser:
    """Build an ArgumentParser with flags shared by all entry points."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-mt", "--model", type=str, default="resnet50",
        help=f"Model to benchmark. Available: {', '.join(list_models())}",
    )
    parser.add_argument(
        "-s", "--size", type=int, default=1024,
        help="Data size in MB (used when --auto is not set).",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-dt", "--data_type", type=str, default="FP32",
        help="Precision: FP32, FP16, or BF16.",
    )
    parser.add_argument(
        "-bs", "--batch", type=int, default=0,
        help="Batch size (0 = model default).",
    )
    parser.add_argument(
        "-abs", "--auto_batch_size", action="store_true", default=False,
        help="Enable automatic batch size optimisation.",
    )
    parser.add_argument(
        "-cudnn", "--cudnn_benchmark", action="store_true", default=False,
        help="Enable cuDNN benchmark mode.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="auto",
        help="Device backend: auto, cuda, mps, npu, musa, tpu.",
    )

    return parser


def resolve_data_type(raw: str) -> str:
    """Normalise a data-type string to one of FP32 / FP16 / BF16."""
    upper = raw.upper().strip()
    if upper in ("FP16", "FP32", "BF16"):
        return upper
    raise ValueError(f"Unknown data type: {raw}. Expected FP32, FP16, or BF16.")


def parse_common_args(parser: argparse.ArgumentParser, args=None) -> argparse.Namespace:
    """Parse and post-process common arguments."""
    ns = parser.parse_args(args)
    ns.model = resolve_model_name(ns.model)
    ns.data_type = resolve_data_type(ns.data_type)
    return ns

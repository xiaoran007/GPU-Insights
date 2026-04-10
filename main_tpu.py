"""TPU multi-core benchmark entry point.

Usage (Colab / GCP TPU VM):
    python main_tpu.py -mt resnet50 -s 1024 -e 5 -dt BF16

For multi-core (all 8 TPU cores):
    python main_tpu.py -mt resnet50 -s 1024 -e 5 -dt BF16 --num_cores 8

Prerequisites:
    pip install torch_xla
"""

import argparse
import os

from benchmark.cli import build_common_parser, parse_common_args


def _mp_fn(index, args):
    """Function executed on each TPU core via xla_spawn."""
    import torch
    import torch_xla.core.xla_model as xm

    rank = xm.get_ordinal()

    from benchmark.Bench import Bench

    if rank == 0 and args.data_type == "FP16":
        print("Note: BF16 is TPU's native precision and generally faster than FP16.")

    b = Bench(
        device="tpu",
        size=args.size,
        epochs=args.epochs,
        method=args.model,
        batch_size=args.batch,
        data_type=args.data_type,
        gpu_ids=[0],
        auto_batch_size=False,
    )
    b.start()


def main():
    parser = build_common_parser("GPU-Insights TPU benchmark")
    parser.add_argument(
        "--num_cores", type=int, default=1,
        help="Number of TPU cores (1 for single-core, 8 for full TPU).",
    )

    args = parse_common_args(parser)

    if args.num_cores > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)
    else:
        _mp_fn(0, args)


if __name__ == "__main__":
    main()

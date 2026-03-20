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


def _mp_fn(index, args):
    """Function executed on each TPU core via xla_spawn."""
    import torch
    import torch_xla.core.xla_model as xm

    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    from benchmark.Bench import Bench

    model = "cnn"
    if args.model in ["resnet50", "ResNet-50"]:
        model = "resnet50"
    elif args.model in ["cnn", "CNN"]:
        model = "cnn"

    data_type = "FP32"
    if args.data_type in ["FP16", "fp16"]:
        data_type = "FP16"
    elif args.data_type in ["FP32", "fp32"]:
        data_type = "FP32"
    elif args.data_type in ["BF16", "bf16"]:
        data_type = "BF16"

    if rank == 0:
        if data_type == "FP16":
            print("Note: BF16 is TPU's native precision and generally faster than FP16.")

    b = Bench(
        auto=False,
        tpu=True,
        size=args.size,
        epochs=args.epochs,
        method=model,
        batch_size=args.batch,
        data_type=data_type,
        gpu_ids=[0],
        auto_batch_size=False,
    )
    b.start()


def main():
    parser = argparse.ArgumentParser(description="TPU benchmark for GPU-Insights.")

    parser.add_argument("-s", "--size", type=int, default=1024,
                        help="Data size in MB.")
    parser.add_argument("-e", "--epochs", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("-mt", "--model", type=str, default="resnet50",
                        help="Model type (resnet50 or cnn).")
    parser.add_argument("-dt", "--data_type", type=str, default="BF16",
                        help="Data type (FP32, FP16, BF16). BF16 recommended for TPU.")
    parser.add_argument("-bs", "--batch", type=int, default=0,
                        help="Batch size.")
    parser.add_argument("--num_cores", type=int, default=1,
                        help="Number of TPU cores (1 for single-core, 8 for full TPU).")

    args = parser.parse_args()

    if args.num_cores > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)
    else:
        _mp_fn(0, args)


if __name__ == "__main__":
    main()

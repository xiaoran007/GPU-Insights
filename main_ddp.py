import torch
import torch.distributed as dist
import os

# Suppress profiler warnings
os.environ['TORCH_LOGS'] = '-all'

from benchmark.Bench import Bench
from benchmark.cli import build_common_parser, parse_common_args


def setup_ddp():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = build_common_parser("GPU-Insights DDP benchmark (use with torchrun)")

    args = parse_common_args(parser)

    rank, local_rank, world_size = setup_ddp()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DDP Training with torchrun")
        print(f"World Size: {world_size} processes")
        print(f"Model: {args.model}, Data Type: {args.data_type}")
        batch_label = args.batch if args.batch > 0 else ('auto' if args.auto_batch_size else 'default')
        print(f"Batch Size per GPU: {batch_label}")
        print(f"{'='*60}\n")

    if not torch.cuda.is_available():
        if rank == 0:
            print("Error: CUDA is not available. DDP requires CUDA GPUs.")
        cleanup_ddp()
        return

    b = Bench(
        auto=False,
        device="cuda",
        size=args.size,
        epochs=args.epochs,
        method=args.model,
        batch_size=args.batch,
        cudnn_benchmark=args.cudnn_benchmark,
        data_type=args.data_type,
        gpu_ids=[local_rank],
        auto_batch_size=args.auto_batch_size,
        use_ddp=True,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    b.start()

    cleanup_ddp()


if __name__ == "__main__":
    main()

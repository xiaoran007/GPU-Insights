import torch
import torch.distributed as dist
import argparse
import os

# Suppress profiler warnings (informational only, doesn't affect training)
os.environ['TORCH_LOGS'] = '-all'  # Disable all internal PyTorch logging warnings
# Alternative: Set to specific level if you want some logs
# os.environ['TORCH_LOGS'] = 'ERROR'  # Only show errors

from benchmark.Bench import Bench


def setup_ddp():
    # torchrun sets these automatically
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize the process group (torchrun sets backend via env vars if needed)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main function for torchrun-based DDP training."""
    parser = argparse.ArgumentParser(description="DDP benchmark for GPU-Insights (use with torchrun).")
    
    parser.add_argument("-s", "--size", type=int, required=False, default=1024,
                        help="Set the CUDA memory size in MB.")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=5,
                        help="Set the epochs.")
    parser.add_argument("-mt", "--model", type=str, required=False, default="resnet50",
                        help="Set the model type (resnet50 or cnn).")
    parser.add_argument("-dt", "--data_type", type=str, required=False, default="FP32",
                        help="Set the data type (FP32, FP16 or BF16).")
    parser.add_argument("-bs", "--batch", type=int, required=False, default=0,
                        help="Set the batch size per GPU.")
    parser.add_argument("-abs", "--auto_batch_size", action="store_true", default=False,
                        help="Enable automatic batch size optimization (only for ResNet50).")
    parser.add_argument("-cudnn", "--cudnn_benchmark", action="store_true", default=False,
                        help="Enable cudnn benchmark.")
    
    args = parser.parse_args()
    
    # Setup DDP (torchrun mode)
    rank, local_rank, world_size = setup_ddp()
    
    # Only rank 0 prints messages
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DDP Training with torchrun")
        print(f"World Size: {world_size} processes")
        print(f"Model: {args.model}, Data Type: {args.data_type}")
        print(f"Batch Size per GPU: {args.batch if args.batch > 0 else 'auto' if args.auto_batch_size else 'default'}")
        if world_size > 1:
            print(f"Total Batch Size: {(args.batch if args.batch > 0 else 4) * world_size}")
        print(f"{'='*60}\n")
    
    # Validate
    if not torch.cuda.is_available():
        if rank == 0:
            print("Error: CUDA is not available. DDP requires CUDA GPUs.")
        cleanup_ddp()
        return
    
    # Parse model type
    model = "cnn"
    if args.model in ["resnet50", "ResNet-50"]:
        model = "resnet50"
    elif args.model in ["cnn", "CNN"]:
        model = "cnn"
    
    # Parse data type
    data_type = "FP32"
    if args.data_type in ["FP16", "fp16"]:
        data_type = "FP16"
    elif args.data_type in ["FP32", "fp32"]:
        data_type = "FP32"
    elif args.data_type in ["BF16", "bf16"]:
        data_type = "BF16"
    
    # Check if auto_batch_size is enabled for non-resnet50 models
    if args.auto_batch_size and model != "resnet50":
        if rank == 0:
            print("Warning: Auto batch size is only supported for ResNet50 model. Ignoring -abs flag.")
        args.auto_batch_size = False
    
    # Check if model supports DDP
    if model not in ["resnet50"]:
        if rank == 0:
            print(f"Warning: DDP is currently only supported for ResNet50 model.")
        cleanup_ddp()
        return
    
    # Create benchmark instance with DDP parameters
    # local_rank is used as the GPU index
    b = Bench(
        auto=False,
        huawei=False,
        mthreads=False,
        size=args.size,
        epochs=args.epochs,
        method=model,
        batch_size=args.batch,
        cudnn_benchmark=args.cudnn_benchmark,
        data_type=data_type,
        gpu_ids=[local_rank],  # Use local_rank as GPU index
        auto_batch_size=args.auto_batch_size,
        use_ddp=True,
        ddp_rank=rank,
        ddp_world_size=world_size
    )
    
    # Run benchmark
    b.start()
    
    # Clean up
    cleanup_ddp()



if __name__ == "__main__":
    main()


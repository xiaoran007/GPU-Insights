import torch
import argparse
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.bench.resnet50_bench import ResNet50Bench

def get_gpu_info(device_id):
    """Gets information about the specified GPU."""
    if not torch.cuda.is_available():
        return "N/A", 0
    
    properties = torch.cuda.get_device_properties(device_id)
    total_memory_gb = properties.total_memory / (1024**3)
    return properties.name, total_memory_gb

def measure_memory_usage(precision, batch_size, gpu_id):
    """
    Runs the ResNet50 benchmark for a single epoch and measures the peak GPU memory usage.
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    # Reset memory stats before the run
    torch.cuda.reset_peak_memory_stats(device)
    
    print(f"Testing {precision} with batch size: {batch_size} on GPU {gpu_id}...")
    
    try:
        # Configure and run the benchmark
        use_fp16 = precision == 'FP16'
        
        # We only need to run for one epoch and a small data size for this test
        bench = ResNet50Bench(
            gpu_device=[device],
            cpu_device=torch.device("cpu"),
            epochs=1,
            batch_size=batch_size,
            data_size=1024, # Use a reasonable data size
            use_fp16=use_fp16
        )
        
        # The benchmark internally handles the training loop
        bench.start()
        
        # Get peak memory usage after the run
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
        print(f"✓ Success! Peak memory usage: {peak_memory_mb:.2f} MB")
        return peak_memory_mb

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"✗ Error: CUDA out of memory with batch size {batch_size}. Please try with a smaller batch size.")
            return -1
        else:
            print(f"An unexpected runtime error occurred: {e}")
            raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Calibrate memory usage for ResNet50 on CUDA.")
    parser.add_argument('-gpu', type=str, default='0', help='GPU ID to use for calibration.')
    args = parser.parse_args()

    gpu_id = int(args.gpu)
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a CUDA-enabled GPU.")
        return

    device_name, total_memory_gb = get_gpu_info(gpu_id)
    print(f"--- GPU Memory Calibration for ResNet50 ---")
    print(f"GPU Detected: {device_name}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print("-" * 45)

    batch_sizes_to_test = [128, 256, 512]
    results = {}

    for precision in ['FP32', 'FP16']:
        results[precision] = {}
        print(f"\n--- Starting calibration for {precision} ---")
        for bs in batch_sizes_to_test:
            peak_mem = measure_memory_usage(precision, bs, gpu_id)
            if peak_mem != -1:
                results[precision][bs] = peak_mem
            else:
                # If OOM, we can't continue for this precision
                break
    
    # --- Report ---
    print("\n\n--- CALIBRATION REPORT ---")
    print(f"GPU Model: {device_name}")
    print(f"Total VRAM: {total_memory_gb:.2f} GB")
    print("-" * 28)
    
    for precision, bs_data in results.items():
        print(f"Precision: {precision}")
        if bs_data:
            for bs, mem in bs_data.items():
                print(f"  - Batch Size: {bs:<5} -> Peak Memory: {mem:.2f} MB")
        else:
            print("  - Measurement failed (likely out of memory).")
    
    print("-" * 28)

if __name__ == '__main__':
    main()

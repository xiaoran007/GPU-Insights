
import torch
import argparse
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from benchmark.Bench import Bench

def get_gpu_info(device_id):
    """Gets information about the specified GPU."""
    if not torch.cuda.is_available():
        # Add stubs for other devices if needed, e.g., MPS, XPU
        if torch.backends.mps.is_available():
            return "Apple MPS Device", 0 # MPS doesn't have easily queryable dedicated memory
        return "N/A", 0
    
    properties = torch.cuda.get_device_properties(device_id)
    total_memory_gb = properties.total_memory / (1024**3)
    return properties.name, total_memory_gb

def find_max_batch_size(precision, gpu_id, start_bs=2, step=2):
    """
    Finds the maximum batch size for a given precision by iteratively running the benchmark
    until an out-of-memory error occurs.
    """
    print(f"--- Starting search for {precision} on GPU {gpu_id} ---")
    
    batch_size = start_bs
    last_successful_bs = 0
    
    while True:
        print(f"Trying batch size: {batch_size}...")
        try:
            # We only need to run for one epoch and a small data size for this test
            bench_args = [
                '-m',
                '-mt', 'resnet50',
                '-dt', precision,
                '-bs', str(batch_size),
                '-e', '1',       # Only 1 epoch needed
                '-s', '128',      # Small data size is fine
                '-gpu', str(gpu_id)
            ]
            
            # Suppress argparse errors for internal calls
            try:
                # Use a simplified Bench object for testing
                bench = Bench(bench_args)
                bench.start()
                # If successful, update the last successful batch size
                last_successful_bs = batch_size
                print(f"Success with batch size: {batch_size}")
                batch_size += step

            except SystemExit:
                # Argparse calls sys.exit(), we can ignore it for this script
                pass

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM error at batch size: {batch_size}. Maximum successful batch size is {last_successful_bs}.")
                print(f"--- Finished search for {precision} ---")
                return last_successful_bs
            else:
                print(f"An unexpected runtime error occurred: {e}")
                # Reraise if it's not an OOM error
                raise e
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            
    print(f"--- Finished search for {precision} ---")
    return last_successful_bs

def main():
    parser = argparse.ArgumentParser(description="Calibrate maximum batch size for ResNet50.")
    parser.add_argument('-gpu', type=str, default='0', help='GPU ID to use for calibration.')
    args = parser.parse_args()

    gpu_id = int(args.gpu)
    
    if torch.cuda.is_available():
        device_name, total_memory_gb = get_gpu_info(gpu_id)
        print(f"GPU Detected: {device_name}")
        print(f"Total Memory: {total_memory_gb:.2f} GB")
    elif torch.backends.mps.is_available():
        print("MPS device detected. Memory is unified, so results will reflect system pressure.")
    else:
        print("No compatible GPU (CUDA/MPS) found.")
        return

    print("\nStarting calibration process...")
    
    # Find max batch size for FP32
    max_bs_fp32 = find_max_batch_size('FP32', gpu_id)
    
    # Find max batch size for FP16
    max_bs_fp16 = find_max_batch_size('FP16', gpu_id)

    # --- Report ---
    print("\n\n--- CALIBRATION REPORT ---")
    if torch.cuda.is_available():
        print(f"GPU Model: {device_name}")
        print(f"Total VRAM (GB): {total_memory_gb:.2f}")
    else:
        print("Device: Apple MPS")
        
    print("-" * 28)
    print(f"Precision: FP32 -> Max Batch Size: {max_bs_fp32}")
    print(f"Precision: FP16 -> Max Batch Size: {max_bs_fp16}")
    print("-" * 28)
    print("\nPlease copy and paste this report back to me.")

if __name__ == '__main__':
    main()

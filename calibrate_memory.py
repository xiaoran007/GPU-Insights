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
    Runs a simplified ResNet50 training loop and measures the real peak GPU memory usage during training.
    This directly implements the training loop to capture accurate memory statistics.
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    print(f"Testing {precision} with batch size: {batch_size} on GPU {gpu_id}...")
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        from benchmark.bench.resnet50_bench import ResNet50, FakeDataset
        
        # Configure precision
        use_fp16 = precision == 'FP16'
        
        # Import AMP modules
        try:
            from torch.amp import autocast, GradScaler
            version_flag = True
        except ImportError:
            from torch.cuda.amp import autocast, GradScaler
            version_flag = False
        
        # Create model and move to device
        model = ResNet50().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        
        # Setup GradScaler for FP16
        if use_fp16:
            scaler = GradScaler(device="cuda", enabled=True)
        else:
            scaler = None
        
        # Create dataset and dataloader
        data_size = 1024
        train_dataset = FakeDataset(size=data_size, image_size=(3, 32, 32), num_classes=10)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        
        # Preload data to GPU
        data_preloaded = [(images.to(device), labels.to(device)) for images, labels in train_loader]
        
        # Warmup phase (to trigger all initialization)
        model.train()
        warmup_batches = min(3, len(data_preloaded))
        for i in range(warmup_batches):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            
            if use_fp16:
                if version_flag:
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(dtype=torch.float16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Synchronize and reset memory stats AFTER warmup
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        
        # Run actual training for a few batches and monitor peak memory
        training_batches = min(10, len(data_preloaded))
        peak_memory_mb = 0.0
        
        for i in range(training_batches):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            
            if use_fp16:
                if version_flag:
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(dtype=torch.float16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Synchronize and check memory after each batch
            torch.cuda.synchronize(device)
            current_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            peak_memory_mb = max(peak_memory_mb, current_peak)
        
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

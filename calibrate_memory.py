import torch
import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.models import get_model


def get_gpu_info(device_id):
    if not torch.cuda.is_available():
        return "N/A", 0
    properties = torch.cuda.get_device_properties(device_id)
    total_memory_gb = properties.total_memory / (1024**3)
    return properties.name, total_memory_gb


def measure_memory_usage(model_name, precision, batch_size, gpu_id):
    """Run a simplified training loop and measure peak GPU memory."""
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Testing {model_name} {precision} with batch size: {batch_size} on GPU {gpu_id}...")

    try:
        import torch.nn as nn
        import torch.optim as optim

        model_spec = get_model(model_name)

        try:
            from torch.amp import autocast, GradScaler
            version_flag = True
        except ImportError:
            from torch.cuda.amp import autocast, GradScaler
            version_flag = False

        use_fp16 = precision == 'FP16'
        num_classes = model_spec.get_num_classes()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        model = model_spec.create_model(num_classes=num_classes).to(device)
        criterion = model_spec.get_criterion()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        scaler = GradScaler(device="cuda", enabled=True) if use_fp16 else None

        data_size = 1024
        train_dataset = model_spec.create_dataset(data_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        data_preloaded = [
            (images.to(device), labels.to(device))
            for images, labels in train_loader
        ]

        # Warmup
        model.train()
        for i in range(min(3, len(data_preloaded))):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            if use_fp16:
                ctx = autocast(device_type="cuda", dtype=torch.float16, enabled=True) if version_flag else autocast(dtype=torch.float16, enabled=True)
                with ctx:
                    loss = model_spec.compute_loss(model, images, labels, criterion, device)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model_spec.compute_loss(model, images, labels, criterion, device)
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize(device)

        # Measure
        peak_memory_reserved = 0.0
        peak_memory_allocated = 0.0
        for i in range(min(10, len(data_preloaded))):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            if use_fp16:
                ctx = autocast(device_type="cuda", dtype=torch.float16, enabled=True) if version_flag else autocast(dtype=torch.float16, enabled=True)
                with ctx:
                    loss = model_spec.compute_loss(model, images, labels, criterion, device)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model_spec.compute_loss(model, images, labels, criterion, device)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize(device)
            current_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
            current_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            peak_memory_reserved = max(peak_memory_reserved, current_reserved)
            peak_memory_allocated = max(peak_memory_allocated, current_allocated)

        print(f"✓ Success!")
        print(f"  - Peak Reserved (≈nvidia-smi): {peak_memory_reserved:.2f} MB")
        print(f"  - Peak Allocated (actual tensors): {peak_memory_allocated:.2f} MB")
        print(f"  - Memory Pool Overhead: {peak_memory_reserved - peak_memory_allocated:.2f} MB")
        return peak_memory_reserved

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"✗ Error: CUDA out of memory with batch size {batch_size}.")
            return -1
        raise
    except Exception:
        raise


def main():
    parser = argparse.ArgumentParser(description="Calibrate memory usage on CUDA.")
    parser.add_argument('-gpu', type=str, default='0', help='GPU ID.')
    parser.add_argument('-mt', '--model', type=str, default='resnet50',
                        help='Model to calibrate (resnet50, cnn, vit, unet, ddpm).')
    args = parser.parse_args()

    gpu_id = int(args.gpu)

    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return

    device_name, total_memory_gb = get_gpu_info(gpu_id)
    model_name = args.model
    print(f"--- GPU Memory Calibration for {model_name} ---")
    print(f"GPU Detected: {device_name}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print("-" * 45)

    batch_sizes_to_test = [128, 256, 512]
    results = {}

    for precision in ['FP32', 'FP16']:
        results[precision] = {}
        print(f"\n--- Starting calibration for {precision} ---")
        for bs in batch_sizes_to_test:
            peak_mem = measure_memory_usage(model_name, precision, bs, gpu_id)
            if peak_mem != -1:
                results[precision][bs] = peak_mem
            else:
                break

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time

# PyTorch 2.x optimizations
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
TORCH_2_PLUS = TORCH_VERSION >= (2, 0)

try:
    from torch.amp import autocast, GradScaler
    version_flag = True
except ImportError:
    print("torch.amp is not available, assume using torch_npu.npu.amp instead.")
    try:
        from torch_npu.npu.amp import autocast, GradScaler
        npu_flag = True
    except ImportError:
        print("torch_npu.npu.amp is not available, assume using torch.cuda.amp instead.")
        from torch.cuda.amp import autocast, GradScaler
        npu_flag = False
    version_flag = False


class ResNet50Bench(object):
    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10, use_fp16=False, use_bf16=False, monitor_performance=False, auto_batch_size=False):
        self.gpu_devices = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.lr = lr
        self.data_size = data_size
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.monitor_performance = monitor_performance  # Optional: monitor batch-level performance
        self.auto_batch_size = auto_batch_size

        if use_bf16:
            if torch.cuda.is_bf16_supported(including_emulation=False):
                self.use_bf16 = True
            else:
                print("BF16 is not supported, using FP16 instead.")
                self.use_bf16 = False
                self.use_fp16 = True

        # Auto batch size calculation if enabled
        if auto_batch_size and gpu_device is not None:
            print("\n=== Auto Batch Size Calculation ===")
            batch_size = self._find_optimal_batch_size()
            print(f"✓ Optimal batch size determined: {batch_size}")
            print("===================================\n")
        
        self.batch_size = batch_size
        
        self.train_dataset = FakeDataset(size=data_size, image_size=image_size, num_classes=num_classes)
        # PyTorch 2.x: Enable persistent_workers for better performance
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=1, 
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if TORCH_2_PLUS else False,
            prefetch_factor=2 if TORCH_2_PLUS else 2
        )

    def _find_optimal_batch_size(self):
        """
        Calculate optimal batch size based on available GPU memory.
        Uses empirical memory formulas for ResNet50 to directly compute the best batch size.
        Only supports CUDA devices.
        """
        main_device = self.gpu_devices[0]
        
        # Only support CUDA devices
        if main_device.type != 'cuda':
            print(f"Warning: Auto batch size only supports CUDA devices, current device: {main_device.type}")
            print(f"Falling back to default batch size: 4")
            return 4
        
        # Get GPU memory information
        total_memory = torch.cuda.get_device_properties(main_device).total_memory
        # Reserve some memory for CUDA context and other overhead
        reserved_memory = 500 * 1024 * 1024  # 500MB reserved
        available_memory = total_memory - reserved_memory
        1
        print(f"GPU: {torch.cuda.get_device_name(main_device)}")
        print(f"Total Memory: {total_memory / 1024**2:.2f} MB")
        print(f"Available Memory: {available_memory / 1024**2:.2f} MB")
        
        # ResNet50 memory usage estimation
        # Calibrated using torch.cuda.memory_reserved() which matches nvidia-smi
        # Based on RTX 3090 measurements with BS=[128, 256, 512]
        # Data includes PyTorch memory pool overhead for accurate real-world usage
        
        if self.use_fp16 or self.use_bf16:
            # FP16/BF16 mode (with AMP)
            # Calibrated: BS=128→2002MB, BS=256→4086MB, BS=512→7538MB
            # Linear regression: Memory = 164 + 14.4 × BS (MB)
            model_memory = 164 * 1024 * 1024  # bytes
            per_sample_memory = 14.4 * 1024 * 1024  # bytes
        else:
            # FP32 mode
            # Calibrated: BS=128→3816MB, BS=256→7386MB, BS=512→15346MB
            # Linear regression: Memory = 246 + 27.9 × BS (MB)
            model_memory = 246 * 1024 * 1024  # bytes
            per_sample_memory = 27.9 * 1024 * 1024  # bytes
        
        # Calculate maximum batch size
        memory_for_batch = available_memory - model_memory
        max_batch_size = int(memory_for_batch / per_sample_memory)
        
        # Apply safety factor (92% since memory_reserved includes pool overhead)
        # This accounts for potential fragmentation and runtime variations
        safe_batch_size = int(max_batch_size * 0.92)
        
        # Adjust for multi-GPU (more total memory available)
        if len(self.gpu_devices) > 1:
            safe_batch_size = int(safe_batch_size * len(self.gpu_devices) * 0.9)  # Slight overhead for DataParallel
        
        # Round down to nearest power of 2 for optimal GPU utilization
        if safe_batch_size >= 32:
            power = int(torch.log2(torch.tensor(float(safe_batch_size))).item())
            safe_batch_size = 2 ** power
        else:
            # For very small values, just ensure it's at least 4
            safe_batch_size = max(4, safe_batch_size)
        
        # Clamp to reasonable range
        if self.use_fp16 or self.use_bf16:
            safe_batch_size = max(32, min(safe_batch_size, 4096))
        else:
            safe_batch_size = max(16, min(safe_batch_size, 2048))
        
        print(f"\nCalculated batch size: {safe_batch_size}")
        print(f"  - Model fixed memory: {model_memory / 1024**2:.0f} MB")
        print(f"  - Per-sample memory: {per_sample_memory / 1024**2:.1f} MB")
        print(f"  - Theoretical max batch size: {max_batch_size}")
        print(f"  - Safe batch size (90%): {safe_batch_size} (power of 2)")
        print(f"  - Estimated total memory: {(model_memory + safe_batch_size * per_sample_memory) / 1024**3:.1f} GB")
        
        return safe_batch_size

    def start(self):
        if self.gpu_devices is None:
            # print("GPU is not available, only CPU will be benched.")
            # print("DEBUG mode, skipping CPU bench.")
            print("GPU is not available")
            # self._bench(self.cpu_device)
        else:
            # print("GPU is available, both GPU and CPU will be benched.")
            # print("DEBUG mode, skipping CPU bench.")
            print("DEBUG mode.")
            self._bench(self.gpu_devices)
            # self._bench(self.cpu_device)

    def _bench(self, devices):
        main_device = devices[0]
        model = ResNet50().to(main_device)

        if len(self.gpu_devices) > 1:
            model = nn.DataParallel(model, device_ids=[device.index for device in devices])
        
        # PyTorch 2.x: Compile model for better performance (if supported)
        # Note: torch.compile requires C/C++ compiler (gcc/clang on Linux/macOS, MSVC on Windows)
        if TORCH_2_PLUS and hasattr(torch, 'compile'):
            try:
                # Use default mode for training workloads
                # Falls back gracefully on unsupported backends (MPS, NPU, etc.)
                model = torch.compile(model, mode='default')
                print(f"✓ Model compiled with torch.compile (PyTorch {torch.__version__})")
            except Exception as e:
                error_msg = str(e)
                if "C compiler" in error_msg or "CC environment" in error_msg:
                    print(f"⚠ torch.compile disabled: C/C++ compiler not found")
                    print(f"  Install: macOS: xcode-select --install | Linux: sudo apt install build-essential")
                elif "triton" in error_msg.lower():
                    print(f"⚠ torch.compile disabled: Triton compiler issue")
                else:
                    print(f"⚠ torch.compile not supported on {main_device.type}: {error_msg[:100]}")
                print(f"  → Continuing with standard (eager) mode...")

        criterion = nn.CrossEntropyLoss()
        # PyTorch 2.x: Use fused SGD for better performance
        optimizer = optim.SGD(model.parameters(), lr=self.lr, fused=True if (TORCH_2_PLUS and main_device.type == 'cuda') else False)
        if main_device.type in ["xpu", "npu"]:
            GS_dev = "cuda"
        else:
            GS_dev = main_device.type
        if main_device.type in ["npu"]:
            AC_dev = "cuda"
        else:
            AC_dev = main_device.type

        # PyTorch 2.x: Setup AMP with proper configuration
        if self.use_bf16:
            # BF16 doesn't need GradScaler
            scaler = None
        else:
            if version_flag:
                scaler = GradScaler(device=GS_dev, enabled=self.use_fp16)
            else:
                scaler = GradScaler(enabled=self.use_fp16)
        
        # PyTorch 2.x: Set matmul precision for better performance
        # TF32 is only available on Ampere (compute capability 8.0) and newer GPUs
        if TORCH_2_PLUS and main_device.type == 'cuda':
            props = torch.cuda.get_device_properties(main_device)
            compute_capability = props.major + props.minor / 10.0
            # Ampere architecture starts at compute capability 8.0
            if compute_capability >= 8.0:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"✓ Enabled TF32 for {props.name} (Compute Capability {props.major}.{props.minor})")
            else:
                print(f"ℹ TF32 not available for {props.name} (Compute Capability {props.major}.{props.minor}, requires 8.0+)")

        total_step = len(self.train_loader)
        pre_load_start = time.time()
        data_preloaded = [(images.to(main_device), labels.to(main_device)) for images, labels in self.train_loader]
        pre_load_end = time.time()
        print(f"Pre-load completed on {main_device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")
        
        # Calculate actual data size processed (accounts for drop_last=True)
        actual_data_size = len(data_preloaded) * self.batch_size
        if actual_data_size != self.data_size:
            dropped_samples = self.data_size - actual_data_size
            print(f"Note: Dropped {dropped_samples} samples due to incomplete batch (drop_last=True)")
            print(f"      Processing {actual_data_size}/{self.data_size} images ({actual_data_size/self.data_size*100:.1f}%)")

        # Warmup: run a few iterations to stabilize GPU state (eliminate cold start effects)
        warmup_batches = min(5, len(data_preloaded))
        model.train()  # Ensure model is in training mode
        warmup_pbar = tqdm(total=warmup_batches, desc="Warming up", unit="batch")
        for i in range(warmup_batches):
            images, labels = data_preloaded[i]
            # PyTorch 2.x: Use set_to_none for better performance
            optimizer.zero_grad(set_to_none=True)
            
            if self.use_bf16:
                if version_flag:
                    with autocast(device_type=AC_dev, dtype=torch.bfloat16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(dtype=torch.bfloat16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                if version_flag:
                    with autocast(device_type=AC_dev, dtype=torch.float16, enabled=self.use_fp16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(dtype=torch.float16, enabled=self.use_fp16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            warmup_pbar.update(1)
        warmup_pbar.close()
        
        # Synchronize before starting the timer to ensure all previous operations are completed
        if main_device.type == 'cuda':
            torch.cuda.synchronize(main_device)
        elif main_device.type == 'xpu':
            torch.xpu.synchronize(main_device)
        elif main_device.type == 'npu':
            torch.npu.synchronize(main_device)
        elif main_device.type == 'mps':
            torch.mps.synchronize()
        
        print("Timer started.")
        
        start_time = time.time()
        batch_times = []  # Store batch times for optional performance monitoring
        sample_interval = max(1, len(data_preloaded) // 10)  # Sample ~10 batches per epoch
        
        for epoch in range(self.epochs):
            iters = len(self.train_loader)
            pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                # Optional: monitor performance for sampled batches (only if enabled)
                if self.monitor_performance and i % sample_interval == 0:
                    if main_device.type == 'cuda':
                        torch.cuda.synchronize(main_device)
                    elif main_device.type == 'xpu':
                        torch.xpu.synchronize(main_device)
                    elif main_device.type == 'npu':
                        torch.npu.synchronize(main_device)
                    elif main_device.type == 'mps':
                        torch.mps.synchronize()
                    batch_start = time.time()
                
                # PyTorch 2.x: Use set_to_none for better memory efficiency
                optimizer.zero_grad(set_to_none=True)
                
                if self.use_bf16:
                    if version_flag:
                        with autocast(device_type=AC_dev, dtype=torch.bfloat16, enabled=True):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        with autocast(dtype=torch.bfloat16, enabled=True):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    if version_flag:
                        with autocast(device_type=AC_dev, dtype=torch.float16, enabled=self.use_fp16):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        with autocast(dtype=torch.float16, enabled=self.use_fp16):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # Optional: record batch time for sampled batches
                if self.monitor_performance and i % sample_interval == 0:
                    if main_device.type == 'cuda':
                        torch.cuda.synchronize(main_device)
                    elif main_device.type == 'xpu':
                        torch.xpu.synchronize(main_device)
                    elif main_device.type == 'npu':
                        torch.npu.synchronize(main_device)
                    elif main_device.type == 'mps':
                        torch.mps.synchronize()
                    batch_end = time.time()
                    batch_times.append(batch_end - batch_start)

                pbar.update(1)
                pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.detach().item():.4f}")

            pbar.close()
        
        # Synchronize after training to ensure all GPU operations are completed before stopping the timer
        if main_device.type == 'cuda':
            torch.cuda.synchronize(main_device)
        elif main_device.type == 'xpu':
            torch.xpu.synchronize(main_device)
        elif main_device.type == 'npu':
            torch.npu.synchronize(main_device)
        elif main_device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        time_usage = end_time - start_time
        # Use actual processed data size for accurate score calculation
        basic_score = actual_data_size / time_usage
        final_score = basic_score * (self.epochs / 10) * 100
        print(f"Training completed on {main_device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.0f}")
        
        # Optional: print performance statistics if monitoring is enabled
        if self.monitor_performance and batch_times:
            import numpy as np
            batch_times = np.array(batch_times)
            print(f"\nPerformance Statistics (sampled {len(batch_times)} batches):")
            print(f"  Mean batch time: {batch_times.mean():.4f}s")
            print(f"  Std deviation: {batch_times.std():.4f}s")
            print(f"  Min/Max: {batch_times.min():.4f}s / {batch_times.max():.4f}s")
            print(f"  Coefficient of Variation: {(batch_times.std() / batch_times.mean() * 100):.2f}%")


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, image_size=(3, 32, 32), num_classes=10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = torch.randn(size, *image_size)
        self.labels = torch.randint(0, num_classes, (size,))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.size


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        # PyTorch 2.x: Use inplace ReLU for memory efficiency
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # PyTorch 2.x: Use inplace ReLU
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


if __name__ == "__main__":
    from torchinfo import summary

    model = ResNet50()
    print(model)

    print("\n----------\n")

    t_batch_size = 1024
    summary(model, input_size=(t_batch_size, 3, 32, 32))

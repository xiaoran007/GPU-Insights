import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from helper import getOS, getArch
import time

# PyTorch 2.x optimizations
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
TORCH_2_PLUS = TORCH_VERSION >= (2, 0)

# DDP support
try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False
    print("Warning: torch.distributed not available, DDP support disabled")

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
    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10, use_fp16=False, use_bf16=False, monitor_performance=False, auto_batch_size=False, use_ddp=False, ddp_rank=0, ddp_world_size=1):
        self.gpu_devices = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.lr = lr
        self.data_size = data_size
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.monitor_performance = monitor_performance  # Optional: monitor batch-level performance
        self.auto_batch_size = auto_batch_size
        
        # DDP parameters
        self.use_ddp = use_ddp and DDP_AVAILABLE
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_main_process = (ddp_rank == 0)  # Only rank 0 prints messages

        if use_bf16:
            if torch.cuda.is_bf16_supported(including_emulation=False):
                self.use_bf16 = True
            else:
                print("BF16 is not supported, using FP16 instead.")
                self.use_bf16 = False
                self.use_fp16 = True

        # Auto batch size calculation if enabled
        if auto_batch_size and gpu_device is not None:
            if self.is_main_process:
                print("\n=== Auto Batch Size Calculation ===")
            batch_size = self._find_optimal_batch_size()
            if self.is_main_process:
                print(f"✓ Optimal batch size determined: {batch_size}")
                print("===================================\n")
        
        self.batch_size = batch_size
        
        self.train_dataset = FakeDataset(size=data_size, image_size=image_size, num_classes=num_classes)
        
        # DDP: Use DistributedSampler for data distribution across processes
        if self.use_ddp:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=ddp_world_size,
                rank=ddp_rank,
                shuffle=True,
                drop_last=True
            )
            shuffle = False  # Sampler handles shuffling
        else:
            self.train_sampler = None
            shuffle = True
        
        # PyTorch 2.x: Enable persistent_workers for better performance
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            sampler=self.train_sampler,  # Use sampler for DDP
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
        
        if self.is_main_process:
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
        
        # Apply safety factor (99% since memory_reserved includes pool overhead)
        # This accounts for potential fragmentation and runtime variations
        safe_batch_size = int(max_batch_size * 0.99)
        
        # Adjust for multi-GPU (more total memory available)
        if len(self.gpu_devices) > 1:
            if self.use_ddp:
                # DDP: Each process uses one GPU, no adjustment needed
                pass
            else:
                # DataParallel: Adjust for multiple GPUs
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
        
        if self.is_main_process:
            print(f"\nCalculated batch size: {safe_batch_size}")
            print(f"  - Model fixed memory: {model_memory / 1024**2:.0f} MB")
            print(f"  - Per-sample memory: {per_sample_memory / 1024**2:.1f} MB")
            print(f"  - Theoretical max batch size: {max_batch_size}")
            print(f"  - Safe batch size (99%): {safe_batch_size} (power of 2)")
            print(f"  - Estimated total memory: {(model_memory + safe_batch_size * per_sample_memory) / 1024**3:.1f} GB")
        
        return safe_batch_size

    def start(self):
        if self.gpu_devices is None:
            # print("GPU is not available, only CPU will be benched.")
            # print("DEBUG mode, skipping CPU bench.")
            if self.is_main_process:
                print("GPU is not available")
            # self._bench(self.cpu_device)
        else:
            # print("GPU is available, both GPU and CPU will be benched.")
            # print("DEBUG mode, skipping CPU bench.")
            if self.is_main_process:
                print("DEBUG mode.")
            self._bench(self.gpu_devices)
            # self._bench(self.cpu_device)

    def _bench(self, devices):
        main_device = devices[0]
        model = ResNet50().to(main_device)

        # Choose parallelization strategy
        if self.use_ddp:
            # DDP: Wrap model with DistributedDataParallel
            # In DDP mode, each process has only one GPU (main_device)
            device_id = main_device.index if hasattr(main_device, 'index') else 0
            
            # Fix stride mismatch issue for 1x1 convolutions
            # ResNet50's 1x1 conv layers can produce gradients with non-standard strides
            # We need to ensure all parameters have contiguous gradients
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad = param.grad.contiguous()
            
            # DDP configuration:
            # - gradient_as_bucket_view=False: Avoids stride mismatch warnings
            #   This occurs when gradient strides don't match bucket view strides,
            #   typically with 1x1 convolutions. Setting to False copies gradients
            #   to ensure proper layout, trading ~5-10% performance for compatibility.
            # - broadcast_buffers=True: Sync BatchNorm stats across processes
            # - find_unused_parameters=False: Faster, assumes all params used in forward
            # - bucket_cap_mb=25: Default bucket size, can tune for your network
            model = DDP(
                model, 
                device_ids=[device_id], 
                output_device=device_id,
                gradient_as_bucket_view=False,  # Prevent stride warnings (slight perf cost)
                broadcast_buffers=True,
                find_unused_parameters=False,
                bucket_cap_mb=25  # Default 25MB, tune if needed
            )
            if self.is_main_process:
                print(f"✓ Using DDP with {self.ddp_world_size} processes (Rank {self.ddp_rank}/{self.ddp_world_size})")
                print(f"  Configuration: gradient_as_bucket_view=False (stride-safe mode)")
        elif len(self.gpu_devices) > 1:
            # DataParallel: Legacy multi-GPU support
            model = nn.DataParallel(model, device_ids=[device.index for device in devices])
            print(f"✓ Using DataParallel with {len(devices)} GPUs")
        
        # PyTorch 2.x: Compile model for better performance (if supported)
        # Note: torch.compile requires C/C++ compiler (gcc/clang on Linux/macOS, MSVC on Windows)
        if TORCH_2_PLUS and hasattr(torch, 'compile'):
            if getArch() == "aarch64" and getOS() == "linux":
                if self.is_main_process:
                    print("⚠ torch.compile disabled: aarch64 Linux backend not yet supported")
                    print(f"  → Continuing with standard (eager) mode...")
            else:
                try:
                    # Use default mode for training workloads
                    # Falls back gracefully on unsupported backends (MPS, NPU, etc.)
                    model = torch.compile(model, mode='default')
                    if self.is_main_process:
                        print(f"✓ Model compiled with torch.compile (PyTorch {torch.__version__})")
                except Exception as e:
                    error_msg = str(e)
                    if self.is_main_process:
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
                if self.is_main_process:
                    print(f"✓ Enabled TF32 for {props.name} (Compute Capability {props.major}.{props.minor})")
            else:
                if self.is_main_process:
                    print(f"ℹ TF32 not available for {props.name} (Compute Capability {props.major}.{props.minor}, requires 8.0+)")

        total_step = len(self.train_loader)
        
        # DDP: Register gradient hooks to ensure contiguous gradients
        # This prevents stride mismatch warnings from 1x1 convolutions
        if self.use_ddp:
            def make_grads_contiguous(module):
                """Hook to ensure gradients are contiguous after backward pass"""
                def hook(grad):
                    if grad is not None and not grad.is_contiguous():
                        return grad.contiguous()
                    return grad
                return hook
            
            # Register hooks for all parameters with gradients
            for param in model.parameters():
                if param.requires_grad:
                    param.register_hook(make_grads_contiguous(model))
            
            if self.is_main_process:
                print("✓ Registered gradient contiguity hooks for DDP")
        
        pre_load_start = time.time()
        data_preloaded = [(images.to(main_device), labels.to(main_device)) for images, labels in self.train_loader]
        pre_load_end = time.time()
        if self.is_main_process:
            print(f"Pre-load completed on {main_device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")
        
        # Calculate actual data size processed (accounts for drop_last=True)
        actual_data_size = len(data_preloaded) * self.batch_size
        
        # DDP: Each process handles a fraction of the data due to DistributedSampler
        # For score calculation, we need the total data processed across all processes
        if self.use_ddp:
            total_data_size = actual_data_size * self.ddp_world_size
        else:
            total_data_size = actual_data_size
            
        if actual_data_size != self.data_size:
            if self.is_main_process:
                print(f"Note: Dropped samples due to incomplete batch (drop_last=True)")
                if self.use_ddp:
                    print(f"      This process: {actual_data_size}/{self.data_size} images ({actual_data_size/self.data_size*100:.1f}%)")
                    print(f"      All processes: {total_data_size} images total")
                else:
                    print(f"      Processing {actual_data_size}/{self.data_size} images ({actual_data_size/self.data_size*100:.1f}%)")

        # Warmup: run a few iterations to stabilize GPU state (eliminate cold start effects)
        warmup_batches = min(5, len(data_preloaded))
        model.train()  # Ensure model is in training mode
        if self.is_main_process:
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
            
            if self.is_main_process:
                warmup_pbar.update(1)
        if self.is_main_process:
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
        
        if self.is_main_process:
            print("Timer started.")
        
        start_time = time.time()
        batch_times = []  # Store batch times for optional performance monitoring
        sample_interval = max(1, len(data_preloaded) // 10)  # Sample ~10 batches per epoch
        
        for epoch in range(self.epochs):
            # DDP: Set epoch for DistributedSampler to ensure proper shuffling
            if self.use_ddp and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            iters = len(self.train_loader)
            if self.is_main_process:
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

                if self.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.detach().item():.4f}")

            if self.is_main_process:
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
        
        # Score calculation:
        # - In DDP mode: use total_data_size (all processes combined) for system throughput
        # - In single-GPU/DataParallel mode: use actual_data_size
        basic_score = total_data_size / time_usage
        final_score = basic_score * (self.epochs / 10) * 100
        
        # DDP: Only rank 0 prints the final results
        if self.is_main_process:
            if self.use_ddp:
                print(f"Training completed. Time: {time_usage:.2f}s")
                print(f"  This process: {actual_data_size} images on {main_device}")
                print(f"  Total throughput: {total_data_size} images across {self.ddp_world_size} GPUs")
                print(f"  Score: {final_score:.0f}")
            else:
                print(f"Training completed on {main_device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.0f}")
        
        # Optional: print performance statistics if monitoring is enabled
        if self.monitor_performance and batch_times and self.is_main_process:
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

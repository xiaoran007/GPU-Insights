import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from typing import List, Optional

from benchmark.runners.base import BenchRunner, BenchResult
from benchmark.devices.base import DeviceBackend
from benchmark.models.base import BenchModel
from benchmark.data import FakeDataset

# PyTorch 2.x detection
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
TORCH_2_PLUS = TORCH_VERSION >= (2, 0)


class SingleRunner(BenchRunner):
    """Single-GPU (or single-device) benchmark runner.

    Reproduces the exact training loop from the original cnn_bench.py and
    resnet50_bench.py, but delegates all device-specific operations to a
    DeviceBackend instance.
    """

    def __init__(
        self,
        model_spec: BenchModel,
        device_backend: DeviceBackend,
        devices: List[torch.device],
        data_size: int,
        batch_size: int,
        epochs: int,
        use_fp16: bool = False,
        use_bf16: bool = False,
        auto_batch_size: bool = False,
        is_main_process: bool = True,
    ):
        self.model_spec = model_spec
        self.backend = device_backend
        self.devices = devices
        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.auto_batch_size = auto_batch_size
        self.is_main_process = is_main_process
        self.lr = 0.001

    def run(self) -> BenchResult:
        model_name = self.model_spec.name
        if model_name == "cnn":
            return self._run_cnn()
        elif model_name == "resnet50":
            return self._run_resnet50()
        else:
            raise NotImplementedError(f"Unknown model: {model_name}")

    # ------------------------------------------------------------------ CNN
    def _run_cnn(self) -> BenchResult:
        """Reproduces the exact CNNBench._bench training loop."""
        device = self.devices[0]
        image_size = self.model_spec.get_image_size()
        num_classes = 10

        train_dataset = FakeDataset(size=self.data_size, image_size=image_size, num_classes=num_classes)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        model = self.model_spec.create_model(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        total_step = len(train_loader)
        pre_load_start = time.time()
        data_preloaded = [(images.to(device), labels.to(device)) for images, labels in train_loader]
        pre_load_end = time.time()
        print(f"Pre-load completed on {device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")

        # Warmup
        print("Warming up...")
        warmup_batches = min(5, len(data_preloaded))
        for i in range(warmup_batches):
            images, labels = data_preloaded[i]
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.backend.synchronize(device)

        start_time = time.time()
        for epoch in range(self.epochs):
            iters = len(train_loader)
            pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.item():.4f}")
            pbar.close()

        self.backend.synchronize(device)

        end_time = time.time()
        time_usage = end_time - start_time
        basic_score = self.data_size / time_usage
        final_score = basic_score * (self.epochs / 10)
        print(f"Training completed on {device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.2f}")

        return BenchResult(
            time_usage=time_usage,
            score=final_score,
            data_size=self.data_size,
            total_data_size=self.data_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device_name=str(device),
        )

    # ------------------------------------------------------------- ResNet50
    def _run_resnet50(self) -> BenchResult:
        """Reproduces the exact ResNet50Bench._bench training loop for single-GPU / DataParallel."""
        main_device = self.devices[0]
        image_size = self.model_spec.get_image_size()
        num_classes = 10

        # Handle BF16 support check (CUDA-specific)
        use_bf16 = self.use_bf16
        use_fp16 = self.use_fp16
        if use_bf16 and self.backend.name == "cuda":
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                print("BF16 is not supported, using FP16 instead.")
                use_bf16 = False
                use_fp16 = True

        # Auto batch size
        batch_size = self.batch_size
        if self.auto_batch_size and self.devices:
            if self.is_main_process:
                print("\n=== Auto Batch Size Calculation ===")
            batch_size = self._find_optimal_batch_size(use_fp16, use_bf16)
            if self.is_main_process:
                print(f"✓ Optimal batch size determined: {batch_size}")
                print("===================================\n")

        # Dataset & DataLoader (matches original exactly)
        train_dataset = FakeDataset(size=self.data_size, image_size=image_size, num_classes=num_classes)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if TORCH_2_PLUS else False,
            prefetch_factor=2 if TORCH_2_PLUS else 2,
        )

        # Model creation
        model = self.model_spec.create_model(num_classes=num_classes).to(main_device)

        # DataParallel for multi-GPU (non-DDP)
        if len(self.devices) > 1:
            model = nn.DataParallel(model, device_ids=[device.index for device in self.devices])
            print(f"✓ Using DataParallel with {len(self.devices)} GPUs")

        # torch.compile
        model = self.backend.try_compile_model(model, is_main_process=self.is_main_process)

        criterion = nn.CrossEntropyLoss()
        optimizer_kwargs = self.backend.get_optimizer_kwargs()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, **optimizer_kwargs)

        # AMP setup
        if use_bf16:
            scaler = None
        else:
            scaler = self.backend.get_grad_scaler(enabled=use_fp16)

        # Device-specific precision optimizations (TF32, etc.)
        self.backend.setup_precision(main_device, is_main_process=self.is_main_process)

        total_step = len(train_loader)

        # Pre-load data
        pre_load_start = time.time()
        data_preloaded = [(images.to(main_device), labels.to(main_device)) for images, labels in train_loader]
        pre_load_end = time.time()
        if self.is_main_process:
            print(f"Pre-load completed on {main_device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")

        actual_data_size = len(data_preloaded) * batch_size
        total_data_size = actual_data_size
        if actual_data_size != self.data_size:
            if self.is_main_process:
                print(f"Note: Dropped samples due to incomplete batch (drop_last=True)")
                print(f"      Processing {actual_data_size}/{self.data_size} images ({actual_data_size/self.data_size*100:.1f}%)")

        # Warmup
        warmup_batches = min(5, len(data_preloaded))
        model.train()
        if self.is_main_process:
            warmup_pbar = tqdm(total=warmup_batches, desc="Warming up", unit="batch")
        for i in range(warmup_batches):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            self._train_step(model, images, labels, criterion, optimizer, scaler, use_fp16, use_bf16, main_device)
            if self.is_main_process:
                warmup_pbar.update(1)
        if self.is_main_process:
            warmup_pbar.close()

        self.backend.synchronize(main_device)

        if self.is_main_process:
            print("Timer started.")

        start_time = time.time()
        for epoch in range(self.epochs):
            iters = len(train_loader)
            if self.is_main_process:
                pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                optimizer.zero_grad(set_to_none=True)
                loss = self._train_step(model, images, labels, criterion, optimizer, scaler, use_fp16, use_bf16, main_device)
                if self.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.detach().item():.4f}")
            if self.is_main_process:
                pbar.close()

        self.backend.synchronize(main_device)

        end_time = time.time()
        time_usage = end_time - start_time
        basic_score = total_data_size / time_usage
        final_score = basic_score * (self.epochs / 10) * 100

        if self.is_main_process:
            print(f"Training completed on {main_device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.0f}")

        return BenchResult(
            time_usage=time_usage,
            score=final_score,
            data_size=actual_data_size,
            total_data_size=total_data_size,
            batch_size=batch_size,
            epochs=self.epochs,
            device_name=str(main_device),
        )

    # --------------------------------------------------- shared train step
    def _train_step(self, model, images, labels, criterion, optimizer, scaler, use_fp16, use_bf16, device):
        """Execute one training step with proper AMP handling."""
        if use_bf16:
            with self.backend.get_autocast_context(device, torch.bfloat16, True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with self.backend.get_autocast_context(device, torch.float16, use_fp16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return loss

    # ---------------------------------------- auto batch size (CUDA only)
    def _find_optimal_batch_size(self, use_fp16, use_bf16):
        """Reproduce the exact ResNet50Bench._find_optimal_batch_size logic."""
        main_device = self.devices[0]

        if self.backend.name != 'cuda':
            print(f"Warning: Auto batch size only supports CUDA devices, current device: {self.backend.name}")
            print(f"Falling back to default batch size: 4")
            return 4

        total_memory = self.backend.get_device_memory(main_device)
        reserved_memory = 500 * 1024 * 1024  # 500MB
        available_memory = total_memory - reserved_memory

        if self.is_main_process:
            print(f"GPU: {self.backend.get_device_name(main_device)}")
            print(f"Total Memory: {total_memory / 1024**2:.2f} MB")
            print(f"Available Memory: {available_memory / 1024**2:.2f} MB")

        if use_fp16 or use_bf16:
            model_memory = 164 * 1024 * 1024
            per_sample_memory = 14.4 * 1024 * 1024
        else:
            model_memory = 246 * 1024 * 1024
            per_sample_memory = 27.9 * 1024 * 1024

        memory_for_batch = available_memory - model_memory
        max_batch_size = int(memory_for_batch / per_sample_memory)
        safe_batch_size = int(max_batch_size * 0.99)

        if len(self.devices) > 1:
            safe_batch_size = int(safe_batch_size * len(self.devices) * 0.9)

        if safe_batch_size >= 32:
            power = int(torch.log2(torch.tensor(float(safe_batch_size))).item())
            safe_batch_size = 2 ** power
        else:
            safe_batch_size = max(4, safe_batch_size)

        if use_fp16 or use_bf16:
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

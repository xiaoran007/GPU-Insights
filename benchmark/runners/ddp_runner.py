import torch
import torch.optim as optim
from tqdm import tqdm
import time
from typing import List

from benchmark.runners.base import BenchRunner, BenchResult
from benchmark.runners.common import train_step
from benchmark.calibration import find_optimal_batch_size
from benchmark.devices.base import DeviceBackend
from benchmark.models.base import BenchModel
from benchmark.scoring import compute_score, print_score

# PyTorch 2.x detection
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
TORCH_2_PLUS = TORCH_VERSION >= (2, 0)

# DDP support
try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False


class DDPRunner(BenchRunner):
    """DDP multi-GPU benchmark runner.

    Works with any registered BenchModel that declares ``supports_ddp = True``.
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
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
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
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_main_process = (ddp_rank == 0)
        self.lr = 0.001

    def run(self) -> BenchResult:
        if not DDP_AVAILABLE:
            raise RuntimeError("DDP is not available (torch.distributed not found)")
        return self._run_ddp()

    def _run_ddp(self) -> BenchResult:
        main_device = self.devices[0]
        num_classes = self.model_spec.get_num_classes()

        # ---- Resolve precision ----
        use_bf16 = self.use_bf16
        use_fp16 = self.use_fp16
        if use_bf16 and self.backend.name == "cuda":
            if not self.backend.supports_bf16(main_device):
                if self.is_main_process:
                    print("BF16 is not supported, using FP16 instead.")
                use_bf16 = False
                use_fp16 = True

        # ---- Resolve batch size ----
        batch_size = self.batch_size
        data_type = "BF16" if use_bf16 else ("FP16" if use_fp16 else "FP32")
        if self.auto_batch_size:
            if self.is_main_process:
                print("\n=== Auto Batch Size Selection ===")
            batch_size = find_optimal_batch_size(
                model_spec=self.model_spec,
                backend=self.backend,
                device=main_device,
                data_type=data_type,
                is_main_process=self.is_main_process,
            )
            if self.is_main_process:
                print(f"  ✓ batch_size = {batch_size}")
                print("=================================\n")
        elif batch_size == 0:
            batch_size = self.model_spec.get_default_batch_size(data_type)

        # ---- Dataset with DistributedSampler ----
        train_dataset = self.model_spec.create_dataset(self.data_size)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.ddp_world_size,
            rank=self.ddp_rank,
            shuffle=True,
            drop_last=True,
        )
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
        if TORCH_2_PLUS:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
        train_loader = self.backend.wrap_dataloader(train_loader, main_device)

        # ---- Model creation + DDP wrapping ----
        model = self.model_spec.create_model(num_classes=num_classes).to(main_device)
        if self.model_spec.use_channels_last and self.backend.supports_channels_last():
            model = model.to(memory_format=torch.channels_last)
        device_id = main_device.index if hasattr(main_device, 'index') and main_device.index is not None else 0

        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = param.grad.contiguous()

        model = DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            gradient_as_bucket_view=False,
            broadcast_buffers=True,
            find_unused_parameters=False,
            bucket_cap_mb=25,
        )
        if self.is_main_process:
            print(f"✓ Using DDP with {self.ddp_world_size} processes (Rank {self.ddp_rank}/{self.ddp_world_size})")

        # torch.compile
        if self.model_spec.supports_compile:
            model = self.backend.try_compile_model(model, is_main_process=self.is_main_process)

        # ---- Criterion / Optimizer ----
        criterion = self.model_spec.get_criterion()
        optimizer_kwargs = self.backend.get_optimizer_kwargs()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, **optimizer_kwargs)

        # ---- AMP setup ----
        if use_bf16:
            scaler = None
        elif self.model_spec.supports_amp and use_fp16:
            scaler = self.backend.get_grad_scaler(enabled=True)
        else:
            scaler = self.backend.get_grad_scaler(enabled=False)

        self.backend.setup_precision(main_device, is_main_process=self.is_main_process)

        total_step = len(train_loader)

        # Gradient contiguity hooks
        def make_grads_contiguous(module):
            def hook(grad):
                if grad is not None and not grad.is_contiguous():
                    return grad.contiguous()
                return grad
            return hook

        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(make_grads_contiguous(model))

        # ---- Pre-load data ----
        pre_load_start = time.time()
        cl = self.model_spec.use_channels_last and self.backend.supports_channels_last()
        data_preloaded = [
            (
                images.to(main_device, memory_format=torch.channels_last, non_blocking=True)
                if cl else images.to(main_device, non_blocking=True),
                labels.to(main_device, non_blocking=True),
            )
            for images, labels in train_loader
        ]
        self.backend.synchronize(main_device)
        pre_load_end = time.time()
        if self.is_main_process:
            print(f"Pre-load completed on {main_device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")

        actual_data_size = len(data_preloaded) * batch_size
        total_data_size = actual_data_size * self.ddp_world_size

        if actual_data_size != self.data_size and self.is_main_process:
            print(f"Note: This process: {actual_data_size}/{self.data_size} images. All processes: {total_data_size} total.")

        # ---- Warmup ----
        warmup_batches = min(5, len(data_preloaded))
        model.train()
        if self.is_main_process:
            warmup_pbar = tqdm(total=warmup_batches, desc="Warming up", unit="batch")
        for i in range(warmup_batches):
            images, labels = data_preloaded[i]
            optimizer.zero_grad(set_to_none=True)
            train_step(
                self.model_spec, self.backend, model, images, labels,
                criterion, optimizer, scaler, use_fp16, use_bf16, main_device,
            )
            if self.is_main_process:
                warmup_pbar.update(1)
        if self.is_main_process:
            warmup_pbar.close()

        self.backend.synchronize(main_device)
        self.backend.reset_peak_memory(main_device)

        if self.is_main_process:
            print("Timer started.")

        # ---- Training loop ----
        start_time = time.time()
        log_interval = max(1, len(data_preloaded) // 10)
        total_steps = 0

        for epoch in range(self.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            iters = len(data_preloaded)
            if self.is_main_process:
                pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                optimizer.zero_grad(set_to_none=True)
                loss = train_step(
                    self.model_spec, self.backend, model, images, labels,
                    criterion, optimizer, scaler, use_fp16, use_bf16, main_device,
                )
                total_steps += 1
                if self.is_main_process:
                    pbar.update(1)
                    if i % log_interval == 0:
                        pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.detach().item():.4f}")
            if self.is_main_process:
                pbar.close()

        self.backend.synchronize(main_device)
        end_time = time.time()

        # ---- Scoring ----
        peak_mem = self.backend.get_peak_memory_mb(main_device)
        bench_score = compute_score(
            total_samples=total_data_size * self.epochs,
            time_usage=end_time - start_time,
            epochs=self.epochs,
            num_steps=total_steps,
            batch_size=batch_size,
            peak_memory_mb=peak_mem,
            device_name=str(main_device),
        )
        print_score(bench_score, is_main_process=self.is_main_process)

        return BenchResult(
            time_usage=bench_score.time_total,
            score=bench_score.score,
            data_size=actual_data_size,
            total_data_size=total_data_size,
            batch_size=batch_size,
            epochs=self.epochs,
            device_name=str(main_device),
        )

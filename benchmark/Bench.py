import torch
import sys

from benchmark.models import get_model
from benchmark.devices import auto_detect_backend
from benchmark.devices.base import DeviceBackend
from benchmark.runners.single_runner import SingleRunner
from benchmark.runners.ddp_runner import DDPRunner


class Bench(object):
    """Orchestrator — public API is unchanged from the original implementation.

    Internally delegates to pluggable models, device backends, and runners.
    """

    def __init__(self, method="cnn", auto=True, huawei=False, mthreads=False, size=1024, epochs=10, batch_size=4, cudnn_benchmark=False, data_type="FP32", gpu_ids=[0], auto_batch_size=False, use_ddp=False, ddp_rank=0, ddp_world_size=1):
        self.huawei = huawei
        self.mthreads = mthreads
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_main_process = (ddp_rank == 0)
        torch.backends.cudnn.benchmark = cudnn_benchmark

        # --- Device detection (preserves original output) ---
        self.device_backend, self.gpu_devices = self._detect_devices(gpu_ids)
        self.cpu_device = torch.device("cpu")

        # --- Build runner ---
        self.runner = self._build_runner(
            method=method, auto=auto, size=size, epochs=epochs,
            batch_size=batch_size, data_type=data_type,
            auto_batch_size=auto_batch_size,
        )

    def start(self):
        self.runner.run()

    # ------------------------------------------------------------------ private

    def _detect_devices(self, gpu_ids):
        """Detect devices — handles Huawei/Mthreads legacy paths and delegates
        to the DeviceRegistry for standard backends (CUDA, MPS, …)."""
        if self.huawei:
            import torch_npu
            if torch_npu.npu.is_available():
                print(f"Found NPU device: {torch_npu.npu.get_device_name()}")
                return None, [torch.device("npu")]
            else:
                print("NPU is not available.")
                return None, [None]
        elif self.mthreads:
            import torch_musa
            if torch.musa.is_available():
                print(f"Found MUSA device: {torch.musa.get_device_name()}")
                return None, [torch.device("musa")]
            else:
                print("MUSA is not available.")
                return None, [None]

        backend = auto_detect_backend()
        if backend is not None:
            devices = backend.detect_devices(gpu_ids)
            return backend, devices
        else:
            print("No GPU device found.")
            print("----------------")
            return None, [None]

    def _build_runner(self, method, auto, size, epochs, batch_size, data_type, auto_batch_size):
        """Build the appropriate runner, matching the original _load_backend logic exactly."""

        # --- Data size calculation (unchanged) ---
        if self.device_backend is not None:
            total_memory = self.device_backend.print_device_info(
                [d for d in self.gpu_devices if d is not None]
            )
        else:
            total_memory = 0

        if auto:
            data_size = int(int((total_memory / 12296) / 100) * 100 * 0.7)
            epochs = 10
        else:
            data_size = int(int((size * 1024 * 1024 / 12296) / 1) * 1)

        if self.is_main_process:
            print(f"Set model, set data size to {data_size} images, total memory size: {data_size * 12296 / 1024 / 1024:.2f} MB")

        # --- Resolve model ---
        model_spec = get_model(method)

        # --- Resolve batch_size defaults ---
        if method == "cnn":
            if batch_size == 0:
                batch_size = model_spec.get_default_batch_size(data_type)
        elif method == "resnet50":
            if batch_size == 0 and not auto_batch_size:
                batch_size = model_spec.get_default_batch_size(data_type)

        # --- Parse data type flags ---
        use_fp16 = (data_type == "FP16")
        use_bf16 = (data_type == "BF16")

        # --- GPU not available → early return with a no-op runner ---
        if self.gpu_devices[0] is None or self.device_backend is None:
            if self.is_main_process:
                print("GPU is not available")
            return _NoOpRunner()

        # --- Choose runner ---
        if self.use_ddp:
            return DDPRunner(
                model_spec=model_spec,
                device_backend=self.device_backend,
                devices=self.gpu_devices,
                data_size=data_size,
                batch_size=batch_size,
                epochs=epochs,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                auto_batch_size=auto_batch_size,
                ddp_rank=self.ddp_rank,
                ddp_world_size=self.ddp_world_size,
            )
        else:
            return SingleRunner(
                model_spec=model_spec,
                device_backend=self.device_backend,
                devices=self.gpu_devices,
                data_size=data_size,
                batch_size=batch_size,
                epochs=epochs,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                auto_batch_size=auto_batch_size,
                is_main_process=self.is_main_process,
            )


class _NoOpRunner:
    """Placeholder when no GPU is available."""
    def run(self):
        pass


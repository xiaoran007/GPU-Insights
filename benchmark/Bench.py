import torch
import sys
from benchmark.bench.cnn_bench import CNNBench
from benchmark.bench.resnet50_bench import ResNet50Bench
from macos_hw_detector import get_gpu_info


class Bench(object):
    def __init__(self, method="cnn", auto=True, huawei=False, mthreads=False, size=1024, epochs=10, batch_size=4, cudnn_benchmark=False, data_type="FP32", gpu_ids=[0], auto_batch_size=False, use_ddp=False, ddp_rank=0, ddp_world_size=1):
        self.huawei = huawei
        self.mthreads = mthreads
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        torch.backends.cudnn.benchmark = cudnn_benchmark
        self.gpu_device = self._get_gpu_device(gpu_ids)
        self.cpu_device = self._get_cpu_device()
        self.backend = self._load_backend(method=method, auto=auto, size=size, epochs=epochs, batch_size=batch_size, data_type=data_type, auto_batch_size=auto_batch_size, use_ddp=use_ddp, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size)

    def start(self):
        self.backend.start()

    def _get_gpu_device(self, gpu_ids):
        devices = []
        if self.huawei:
            import torch_npu
            if torch_npu.npu.is_available():
                print(f"Found NPU device: {torch_npu.npu.get_device_name()}")
                devices.append(torch.device("npu"))
            else:
                print("NPU is not available.")
                devices.append(None)
        elif self.mthreads:
            import torch_musa
            if torch.musa.is_available():
                print(f"Found MUSA device: {torch.musa.get_device_name()}")
                devices.append(torch.device("musa"))
            else:
                print("MUSA is not available.")
                devices.append(None)
        else:
            if torch.cuda.is_available():
                for gpu_id in gpu_ids:
                    print(f"Found cuda device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                    devices.append(torch.device(f"cuda:{gpu_id}"))
            elif torch.backends.mps.is_available():  # experimental mode
                print(f"Found mps device: {get_gpu_info()[0]['name']}")
                devices.append(torch.device("mps"))
            elif torch.xpu.is_available():  # experimental mode
                print(f"Found xpu device: {torch.xpu.get_device_name()}")
                devices.append(torch.device("xpu"))
            else:
                devices.append(None)
            print("----------------")
        return devices

    @staticmethod
    def _get_cpu_device():
        return torch.device("cpu")

    @staticmethod
    def _get_cuda_memory_size(devices):
        if torch.cuda.is_available():
            total_memory = 0
            for device in devices:
                props = torch.cuda.get_device_properties(device)
                print(f"Set cuda device: {props.name}, CUDA architecture: {props.major}.{props.minor}\nFound {props.total_memory / 1024 / 1024:.2f} MB CUDA memory available.")
                total_memory += props.total_memory
            print("----------------")
            return total_memory
        else:
            return 0

    def _load_backend(self, method, auto, size, epochs, batch_size, data_type, auto_batch_size, use_ddp=False, ddp_rank=0, ddp_world_size=1):
        if auto:
            cuda_memory_size = self._get_cuda_memory_size(self.gpu_device)
            data_size = int(int((cuda_memory_size / 12296) / 100) * 100 * 0.7)
            epochs = 10
        else:
            self._get_cuda_memory_size(self.gpu_device)
            data_size = int(int((size * 1024 * 1024 / 12296) / 1) * 1)
        
        # Only rank 0 prints in DDP mode
        if not use_ddp or ddp_rank == 0:
            print(f"Set model, set data size to {data_size} images, total memory size: {data_size * 12296 / 1024 / 1024:.2f} MB")
        
        if method == "cnn":
            if batch_size == 0:
                batch_size = 2048
            return CNNBench(gpu_device=self.gpu_device[0], cpu_device=self.cpu_device, data_size=data_size,
                            batch_size=batch_size, epochs=epochs)
        elif method == "resnet50":
            if batch_size == 0 and not auto_batch_size:
                batch_size = 4
            if data_type == "FP16":
                use_fp16 = True
                use_bf16 = False
            elif data_type == "BF16":
                use_fp16 = False
                use_bf16 = True
            else:
                use_fp16 = False
                use_bf16 = False
            return ResNet50Bench(gpu_device=self.gpu_device, cpu_device=self.cpu_device, data_size=data_size,
                                 batch_size=batch_size, epochs=epochs, use_fp16=use_fp16, use_bf16=use_bf16, 
                                 auto_batch_size=auto_batch_size, use_ddp=use_ddp, ddp_rank=ddp_rank, 
                                 ddp_world_size=ddp_world_size)
        else:
            raise NotImplementedError




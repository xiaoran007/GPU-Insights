#!/usr/bin/env python3

import argparse
import json
import platform
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from benchmark.devices import auto_detect_backend
except Exception:
    auto_detect_backend = None

try:
    from benchmark.devices.macos_info import get_gpu_info, get_mem_info
except Exception:
    get_gpu_info = None
    get_mem_info = None


def _run_command(command: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _format_memory_gb(value_gb: float) -> str:
    rounded = round(value_gb)
    if abs(value_gb - rounded) < 0.05:
        return f"{rounded}GB"
    return f"{value_gb:.1f}GB"


def _format_memory_bytes(value_bytes: int) -> str:
    if value_bytes <= 0:
        return ""
    return _format_memory_gb(value_bytes / (1024 ** 3))


def _normalize_platform_label() -> str:
    system = platform.system()
    if system == "Darwin":
        version = platform.mac_ver()[0] or platform.release()
        return f"macOS {version}"
    return f"{system} {platform.release()}"


_NVIDIA_CC_ARCH_MAP: Dict[Tuple[int, int], str] = {
    (12, 1): "Blackwell",
    (12, 0): "Blackwell",
    (11, 0): "Thor",
    (10, 3): "Blackwell",
    (10, 1): "Blackwell",
    (10, 0): "Blackwell",
    (9, 0): "Hopper",
    (8, 9): "Ada",
    (8, 7): "Ampere",
    (8, 6): "Ampere",
    (8, 0): "Ampere",
    (7, 5): "Turing",
    (7, 2): "Volta",
    (7, 0): "Volta",
    (6, 2): "Pascal",
    (6, 1): "Pascal",
    (6, 0): "Pascal",
    (5, 3): "Maxwell",
    (5, 2): "Maxwell",
    (5, 0): "Maxwell",
    (3, 7): "Kepler",
    (3, 5): "Kepler",
    (3, 2): "Kepler",
    (3, 0): "Kepler",
    (2, 1): "Fermi",
    (2, 0): "Fermi",
    (1, 3): "Tesla",
    (1, 2): "Tesla",
    (1, 1): "Tesla",
    (1, 0): "Tesla",
}


def _format_compute_capability(major: int, minor: int) -> str:
    return f"{major}.{minor}"


def _map_nvidia_architecture_from_cc(major: Optional[int], minor: Optional[int]) -> str:
    if major is None or minor is None:
        return "NVIDIA GPU"
    architecture = _NVIDIA_CC_ARCH_MAP.get((major, minor))
    if architecture:
        return architecture
    return f"CUDA CC {_format_compute_capability(major, minor)}"


def _decode_nvml_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _collect_cuda_compute_capabilities(device_ids: List[int]) -> List[Tuple[Optional[int], Optional[int]]]:
    if torch is None or not torch.cuda.is_available():
        return []

    capabilities: List[Tuple[Optional[int], Optional[int]]] = []
    for device_id in device_ids or [0]:
        with suppress(Exception):
            props = torch.cuda.get_device_properties(device_id)
            capabilities.append((props.major, props.minor))
            continue
        capabilities.append((None, None))
    return capabilities


def _probe_cuda_with_nvml(device_ids: List[int]) -> Dict[str, Any]:
    if pynvml is None:
        return {}

    handles = []
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        for device_id in device_ids or [0]:
            handles.append(pynvml.nvmlDeviceGetHandleByIndex(device_id))

        names = [_decode_nvml_string(pynvml.nvmlDeviceGetName(handle)) for handle in handles]
        unique_names = sorted(set(names))
        device_name = unique_names[0] if len(unique_names) == 1 else ", ".join(unique_names)
        if len(handles) > 1:
            device_name = f"{len(handles)}x {device_name}"

        memory_bytes = 0
        with suppress(Exception):
            memory_bytes = pynvml.nvmlDeviceGetMemoryInfo(handles[0]).total

        driver_version = ""
        with suppress(Exception):
            driver_version = _decode_nvml_string(pynvml.nvmlSystemGetDriverVersion())

        capabilities = _collect_cuda_compute_capabilities(device_ids)
        primary_major, primary_minor = capabilities[0] if capabilities else (None, None)
        architecture = _map_nvidia_architecture_from_cc(primary_major, primary_minor)

        note_parts = []
        if primary_major is not None and primary_minor is not None:
            note_parts.append(f"CUDA CC {_format_compute_capability(primary_major, primary_minor)}")
        if len({cap for cap in capabilities if cap != (None, None)}) > 1:
            cc_labels = sorted(
                {_format_compute_capability(major, minor) for major, minor in capabilities if major is not None and minor is not None}
            )
            note_parts.append(f"Mixed compute capability GPUs: {', '.join(cc_labels)}")

        runtime_parts = []
        if driver_version:
            runtime_parts.append(f"Driver {driver_version}")
        cuda_runtime = getattr(torch.version, "cuda", None) if torch is not None else None
        if cuda_runtime:
            runtime_parts.append(f"CUDA {cuda_runtime}")
        if torch is not None:
            runtime_parts.append(f"PyTorch {torch.__version__}")

        return {
            "vendor": "nvidia",
            "architecture": architecture,
            "device": device_name,
            "memory": _format_memory_bytes(memory_bytes),
            "platform": _normalize_platform_label(),
            "driver_runtime": ", ".join(runtime_parts),
            "note": "; ".join(note_parts),
        }
    except Exception:
        return {}
    finally:
        if initialized:
            with suppress(Exception):
                pynvml.nvmlShutdown()


def _safe_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def _detect_backend_name(requested_backend: str = "auto") -> str:
    if requested_backend != "auto":
        return requested_backend
    if auto_detect_backend is None:
        return "cpu"
    backend = auto_detect_backend(device=requested_backend)
    return backend.name if backend is not None else "cpu"


def _probe_cuda(device_ids: List[int]) -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {}

    if not device_ids:
        device_ids = [0]

    nvml_payload = _probe_cuda_with_nvml(device_ids)
    if nvml_payload:
        return nvml_payload

    props = [torch.cuda.get_device_properties(idx) for idx in device_ids]
    names = [item.name for item in props]
    unique_names = sorted(set(names))
    per_gpu_memory_bytes = props[0].total_memory if props else 0
    device = unique_names[0] if len(unique_names) == 1 else ", ".join(unique_names)
    if len(device_ids) > 1:
        device = f"{len(device_ids)}x {device}"

    primary_major = props[0].major if props else None
    primary_minor = props[0].minor if props else None
    compute_capability = (
        _format_compute_capability(primary_major, primary_minor)
        if primary_major is not None and primary_minor is not None
        else None
    )
    driver_version = _run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    cuda_runtime = getattr(torch.version, "cuda", None)
    runtime_parts = []
    if driver_version:
        runtime_parts.append(f"Driver {driver_version.splitlines()[0]}")
    if cuda_runtime:
        runtime_parts.append(f"CUDA {cuda_runtime}")
    runtime_parts.append(f"PyTorch {torch.__version__}")

    return {
        "vendor": "nvidia",
        "architecture": _map_nvidia_architecture_from_cc(primary_major, primary_minor),
        "device": device,
        "memory": _format_memory_bytes(per_gpu_memory_bytes),
        "platform": _normalize_platform_label(),
        "driver_runtime": ", ".join(runtime_parts),
        "note": f"CUDA CC {compute_capability}" if compute_capability else "",
    }


def _probe_mps() -> Dict[str, Any]:
    device_name = "Apple GPU"
    memory = ""
    if get_gpu_info is not None:
        try:
            gpu_info = get_gpu_info()
            if gpu_info:
                device_name = gpu_info[0].get("name", device_name)
                memory = gpu_info[0].get("vram", "")
        except Exception:
            pass
    if not memory and get_mem_info is not None:
        try:
            memory = f"{get_mem_info()} (shared memory)"
        except Exception:
            memory = ""

    runtime_parts = ["Metal"]
    if torch is not None:
        runtime_parts.append(f"PyTorch {torch.__version__}")

    return {
        "vendor": "apple",
        "architecture": "Apple Silicon",
        "device": device_name,
        "memory": memory,
        "platform": _normalize_platform_label(),
        "driver_runtime": ", ".join(runtime_parts),
        "note": "",
    }


def _probe_npu(device_ids: List[int]) -> Dict[str, Any]:
    torch_npu = _safe_import("torch_npu")
    if torch_npu is None:
        return {}

    idx = device_ids[0] if device_ids else 0
    name = torch_npu.npu.get_device_name(idx)
    memory = ""
    try:
        props = torch_npu.npu.get_device_properties(idx)
        memory = _format_memory_bytes(getattr(props, "total_memory", 0))
    except Exception:
        pass

    return {
        "vendor": "huawei",
        "architecture": name,
        "device": name,
        "memory": memory,
        "platform": _normalize_platform_label(),
        "driver_runtime": f"torch_npu {getattr(torch_npu, '__version__', 'unknown')}",
        "note": "",
    }


def _probe_musa(device_ids: List[int]) -> Dict[str, Any]:
    torch_musa = _safe_import("torch_musa")
    if torch_musa is None or torch is None or not hasattr(torch, "musa"):
        return {}

    idx = device_ids[0] if device_ids else 0
    name = torch.musa.get_device_name(idx)
    memory = ""
    try:
        props = torch.musa.get_device_properties(idx)
        memory = _format_memory_bytes(getattr(props, "total_memory", 0))
    except Exception:
        pass

    return {
        "vendor": "mthreads",
        "architecture": "MUSA",
        "device": name,
        "memory": memory,
        "platform": _normalize_platform_label(),
        "driver_runtime": f"torch_musa {getattr(torch_musa, '__version__', 'unknown')}",
        "note": "",
    }


def _probe_tpu() -> Dict[str, Any]:
    torch_xla = _safe_import("torch_xla")
    if torch_xla is None:
        return {}

    device_name = "Google TPU"
    memory = ""
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        supported = xm.get_xla_supported_devices()
        if supported:
            device_name = f"TPU ({supported[0]})"
        info = xm.get_memory_info(device)
        for key in ("kb_total", "bytes_limit", "total"):
            if key in info and info[key] > 0:
                memory = _format_memory_bytes(info[key] * 1024 if "kb" in key else info[key])
                break
    except Exception:
        pass

    return {
        "vendor": "google",
        "architecture": device_name,
        "device": device_name,
        "memory": memory,
        "platform": _normalize_platform_label(),
        "driver_runtime": f"torch_xla {getattr(torch_xla, '__version__', 'unknown')}",
        "note": "",
    }


def _probe_cpu_fallback() -> Dict[str, Any]:
    return {
        "vendor": "unknown",
        "architecture": platform.machine(),
        "device": platform.processor() or platform.machine() or "CPU",
        "memory": "",
        "platform": _normalize_platform_label(),
        "driver_runtime": f"PyTorch {torch.__version__}" if torch is not None else "",
        "note": "No accelerator detected.",
    }


def probe_benchmark_env(requested_backend: str = "auto", device_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Probe normalized host/device metadata for benchmark result export."""
    if device_ids is None:
        device_ids = [0]

    backend = _detect_backend_name(requested_backend)
    if backend == "cuda":
        payload = _probe_cuda(device_ids)
    elif backend == "mps":
        payload = _probe_mps()
    elif backend == "npu":
        payload = _probe_npu(device_ids)
    elif backend == "musa":
        payload = _probe_musa(device_ids)
    elif backend == "tpu":
        payload = _probe_tpu()
    else:
        payload = _probe_cpu_fallback()

    payload.setdefault("vendor", "unknown")
    payload.setdefault("architecture", "unknown")
    payload.setdefault("device", "unknown")
    payload.setdefault("memory", "")
    payload.setdefault("platform", _normalize_platform_label())
    payload.setdefault("driver_runtime", "")
    payload.setdefault("note", "")
    payload["backend"] = backend
    payload["device_ids"] = device_ids
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe benchmark host/device metadata.")
    parser.add_argument(
        "-d", "--device", type=str, default="auto",
        help="Requested device backend: auto, cuda, mps, npu, musa, tpu.",
    )
    parser.add_argument(
        "-gpu", "--gpu-id", type=str, default="0",
        help="Comma-separated device ids when relevant.",
    )
    parser.add_argument("--pretty", action="store_true", default=False, help="Pretty-print JSON output.")
    args = parser.parse_args()

    device_ids = []
    for chunk in args.gpu_id.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        device_ids.append(int(chunk))
    if not device_ids:
        device_ids = [0]

    payload = probe_benchmark_env(requested_backend=args.device, device_ids=device_ids)
    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

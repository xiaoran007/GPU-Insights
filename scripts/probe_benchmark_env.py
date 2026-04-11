#!/usr/bin/env python3

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
except ImportError:
    torch = None

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


def _infer_nvidia_architecture(name: str, compute_capability: Optional[str]) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("5090", "5080", "5070", "5060", "blackwell")):
        return "Blackwell"
    if any(token in lowered for token in ("4090", "4080", "4070", "4060", "ada", "l40", "l4", "rtx 6000 ada")):
        return "Ada"
    if any(token in lowered for token in ("h100", "h200", "hopper")):
        return "Hopper"
    if any(token in lowered for token in ("a100", "a30", "3090", "3080", "3070", "3060", "ampere")):
        return "Ampere"
    if any(token in lowered for token in ("v100", "t4", "titan rtx", "turing", "volta")):
        return "Volta/Turing"
    if compute_capability:
        return f"CUDA CC {compute_capability}"
    return "NVIDIA GPU"


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

    props = [torch.cuda.get_device_properties(idx) for idx in device_ids]
    names = [item.name for item in props]
    unique_names = sorted(set(names))
    per_gpu_memory_bytes = props[0].total_memory if props else 0
    device = unique_names[0] if len(unique_names) == 1 else ", ".join(unique_names)
    if len(device_ids) > 1:
        device = f"{len(device_ids)}x {device}"

    compute_capability = f"{props[0].major}.{props[0].minor}" if props else None
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
        "architecture": _infer_nvidia_architecture(unique_names[0], compute_capability),
        "device": device,
        "memory": _format_memory_bytes(per_gpu_memory_bytes),
        "platform": _normalize_platform_label(),
        "driver_runtime": ", ".join(runtime_parts),
        "note": "",
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

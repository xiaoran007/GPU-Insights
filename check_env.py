import argparse
import contextlib
import importlib
import json
import platform
import sys
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from helper import getArch, getOS

try:
    from macos_hw_detector import get_gpu_info, get_mem_info
except ImportError:
    get_gpu_info = None
    get_mem_info = None


STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_SKIP = "SKIP"
STATUS_FAIL = "FAIL"

DEFAULT_FRAMEWORKS = ["pytorch", "torchvision", "torchaudio", "tensorflow", "jax", "jaxlib"]
FRAMEWORK_ALIASES = {
    "torch": "pytorch",
    "tf": "tensorflow",
}


def make_check(status: str, summary: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "details": details or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick deep learning environment availability check.")
    parser.add_argument(
        "--frameworks",
        type=str,
        default="all",
        help="Comma-separated framework list. Example: pytorch,tensorflow,jax",
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Show detailed probe information.")
    parser.add_argument("--json", action="store_true", default=False, help="Print JSON output.")
    parser.add_argument(
        "--device-only",
        action="store_true",
        default=False,
        help="Only print system and device information.",
    )
    parser.add_argument(
        "--no-runtime-check",
        action="store_true",
        default=False,
        help="Skip lightweight runtime validation.",
    )
    return parser.parse_args()


def normalize_frameworks(raw_value: str) -> List[str]:
    if raw_value.strip().lower() == "all":
        return DEFAULT_FRAMEWORKS.copy()

    normalized = []
    for item in raw_value.split(","):
        value = item.strip().lower()
        if not value:
            continue
        normalized.append(FRAMEWORK_ALIASES.get(value, value))

    if not normalized:
        return DEFAULT_FRAMEWORKS.copy()

    deduped = []
    for item in normalized:
        if item not in deduped:
            deduped.append(item)
    return deduped


def safe_import(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, str(exc)


def collect_system_info() -> Dict[str, Any]:
    system_info = {
        "os": getOS(),
        "arch": getArch(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }

    if getOS() == "macos" and get_mem_info is not None:
        try:
            system_info["physical_memory"] = get_mem_info()
        except Exception as exc:
            system_info["physical_memory_error"] = str(exc)

    return system_info


def _collect_optional_accelerators(torch_module: Any) -> List[Dict[str, Any]]:
    accelerators: List[Dict[str, Any]] = []

    torch_npu, _ = safe_import("torch_npu")
    if torch_npu is not None:
        try:
            available = torch_npu.npu.is_available()
            accelerators.append(
                {
                    "type": "npu",
                    "available": available,
                    "name": torch_npu.npu.get_device_name() if available else "unavailable",
                }
            )
        except Exception as exc:
            accelerators.append({"type": "npu", "available": False, "error": str(exc)})

    torch_musa, _ = safe_import("torch_musa")
    if torch_musa is not None:
        try:
            available = hasattr(torch_module, "musa") and torch_module.musa.is_available()
            accelerators.append(
                {
                    "type": "musa",
                    "available": available,
                    "name": torch_module.musa.get_device_name() if available else "unavailable",
                }
            )
        except Exception as exc:
            accelerators.append({"type": "musa", "available": False, "error": str(exc)})

    return accelerators


def collect_devices(torch_module: Optional[Any]) -> Dict[str, Any]:
    device_info: Dict[str, Any] = {
        "preferred": "cpu",
        "devices": [{"type": "cpu", "name": platform.processor() or platform.machine() or "CPU"}],
        "notes": [],
    }

    if torch_module is None:
        device_info["notes"].append("PyTorch unavailable; accelerator detection is limited.")
        if getOS() == "macos" and get_gpu_info is not None:
            try:
                device_info["devices"].extend(_collect_macos_gpus())
            except Exception as exc:
                device_info["notes"].append(f"macOS GPU probe failed: {exc}")
        return device_info

    try:
        if torch_module.cuda.is_available():
            cuda_devices = []
            for index in range(torch_module.cuda.device_count()):
                props = torch_module.cuda.get_device_properties(index)
                cuda_devices.append(
                    {
                        "type": "cuda",
                        "index": index,
                        "name": torch_module.cuda.get_device_name(index),
                        "memory_gb": round(props.total_memory / 1024**3, 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            device_info["preferred"] = "cuda:0"
            device_info["devices"] = cuda_devices + device_info["devices"]
        elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            mps_device = {"type": "mps", "name": "Apple Metal Performance Shaders"}
            macos_gpus = _collect_macos_gpus()
            if macos_gpus:
                mps_device["name"] = macos_gpus[0].get("name", mps_device["name"])
                mps_device["metal"] = macos_gpus[0].get("metal")
                mps_device["memory"] = macos_gpus[0].get("memory")
                device_info["devices"].extend(macos_gpus)
            device_info["preferred"] = "mps"
            device_info["devices"].insert(0, mps_device)
        elif hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
            device_info["preferred"] = "xpu"
            device_info["devices"].insert(
                0,
                {
                    "type": "xpu",
                    "name": torch_module.xpu.get_device_name(),
                },
            )
    except Exception as exc:
        device_info["notes"].append(f"Primary accelerator probe failed: {exc}")

    for accelerator in _collect_optional_accelerators(torch_module):
        device_info["devices"].append(accelerator)

    if getOS() == "macos" and get_gpu_info is not None and not any(device.get("type") == "macos-gpu" for device in device_info["devices"]):
        try:
            device_info["devices"].extend(_collect_macos_gpus())
        except Exception as exc:
            device_info["notes"].append(f"macOS GPU probe failed: {exc}")

    return device_info


def _collect_macos_gpus() -> List[Dict[str, Any]]:
    if getOS() != "macos" or get_gpu_info is None:
        return []

    gpus = []
    with contextlib.redirect_stdout(StringIO()):
        raw_gpu_info = get_gpu_info()

    for item in raw_gpu_info:
        gpus.append(
            {
                "type": "macos-gpu",
                "name": item.get("name", "unknown"),
                "memory": item.get("vram"),
                "vendor": item.get("vendor_id"),
                "cores": item.get("cores"),
                "metal": item.get("metal"),
                "bus": item.get("bus"),
                "link": item.get("link"),
            }
        )
    return gpus


def collect_framework_checks(torch_module: Optional[Any], frameworks: List[str], verbose: bool) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}

    if "pytorch" in frameworks:
        checks["pytorch"] = collect_pytorch_check(torch_module, verbose)

    for framework in frameworks:
        if framework == "pytorch":
            continue
        checks[framework] = collect_generic_framework_check(framework, verbose)

    return checks


def collect_generic_framework_check(framework: str, verbose: bool) -> Dict[str, Any]:
    module_name = framework
    if framework == "jaxlib":
        module_name = "jaxlib"

    module, error = safe_import(module_name)
    if module is None:
        return make_check(STATUS_SKIP, f"{framework} not installed", {"error": error})

    details: Dict[str, Any] = {
        "module": module_name,
        "version": getattr(module, "__version__", "unknown"),
    }

    if framework == "tensorflow":
        try:
            devices = module.config.list_physical_devices()
            details["devices"] = [f"{device.device_type}:{device.name}" for device in devices]
        except Exception as exc:
            details["devices_error"] = str(exc)
    elif framework == "jax":
        try:
            details["default_backend"] = module.default_backend()
            details["devices"] = [str(device) for device in module.devices()]
        except Exception as exc:
            details["devices_error"] = str(exc)

    summary = f"{framework} {details['version']} is installed"
    if verbose and details.get("devices"):
        summary += f" with {len(details['devices'])} visible device(s)"
    return make_check(STATUS_OK, summary, details)


def collect_pytorch_check(torch_module: Optional[Any], verbose: bool) -> Dict[str, Any]:
    if torch_module is None:
        return make_check(STATUS_FAIL, "PyTorch not installed", {})

    details: Dict[str, Any] = {
        "version": torch_module.__version__,
        "cuda_built": getattr(torch_module.version, "cuda", None),
        "git_version": getattr(torch_module.version, "git_version", None),
    }

    try:
        details["cuda_available"] = torch_module.cuda.is_available()
        details["cuda_device_count"] = torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0
    except Exception as exc:
        details["cuda_probe_error"] = str(exc)

    if hasattr(torch_module.backends, "cudnn"):
        try:
            details["cudnn_available"] = torch_module.backends.cudnn.is_available()
            details["cudnn_version"] = torch_module.backends.cudnn.version()
        except Exception as exc:
            details["cudnn_probe_error"] = str(exc)

    details["mps_available"] = bool(hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available())
    details["xpu_available"] = bool(hasattr(torch_module, "xpu") and torch_module.xpu.is_available())
    details["compile_api_available"] = callable(getattr(torch_module, "compile", None))

    amp_mode = "unavailable"
    amp_error = None
    try:
        importlib.import_module("torch.amp")
        amp_mode = "torch.amp"
    except Exception:
        try:
            importlib.import_module("torch.cuda.amp")
            amp_mode = "torch.cuda.amp"
        except Exception as exc:
            amp_error = str(exc)
    details["amp_mode"] = amp_mode
    if amp_error:
        details["amp_error"] = amp_error

    if details.get("cuda_available") and hasattr(torch_module.cuda, "is_bf16_supported"):
        try:
            details["bf16_supported"] = torch_module.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            details["bf16_supported"] = torch_module.cuda.is_bf16_supported()
        except Exception as exc:
            details["bf16_probe_error"] = str(exc)
    else:
        details["bf16_supported"] = False

    details["tf32_supported"] = False
    if details.get("cuda_available"):
        try:
            capabilities = []
            tf32_supported = False
            for index in range(torch_module.cuda.device_count()):
                props = torch_module.cuda.get_device_properties(index)
                capability = f"{props.major}.{props.minor}"
                capabilities.append({"index": index, "capability": capability})
                if props.major >= 8:
                    tf32_supported = True
            details["compute_capabilities"] = capabilities
            details["tf32_supported"] = tf32_supported
        except Exception as exc:
            details["capability_probe_error"] = str(exc)

    summary_parts = [f"PyTorch {torch_module.__version__}"]
    if details.get("cuda_available"):
        summary_parts.append(f"CUDA runtime visible ({details.get('cuda_device_count', 0)} device(s))")
    elif details.get("mps_available"):
        summary_parts.append("MPS backend available")
    elif details.get("xpu_available"):
        summary_parts.append("XPU backend available")
    else:
        summary_parts.append("CPU-only runtime")

    if verbose and details.get("cuda_built"):
        summary_parts.append(f"built with CUDA {details['cuda_built']}")

    return make_check(STATUS_OK, "; ".join(summary_parts), details)


def collect_precision_checks(torch_module: Optional[Any], device_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = [make_check(STATUS_OK, "FP32 baseline is available", {})]

    if torch_module is None:
        checks.append(make_check(STATUS_SKIP, "FP16/BF16 checks skipped because PyTorch is unavailable", {}))
        return checks

    preferred = device_info.get("preferred", "cpu")
    has_accelerator = preferred != "cpu"

    amp_available = False
    try:
        importlib.import_module("torch.amp")
        amp_available = True
    except Exception:
        try:
            importlib.import_module("torch.cuda.amp")
            amp_available = True
        except Exception:
            amp_available = False

    if has_accelerator and amp_available:
        checks.append(make_check(STATUS_OK, f"Automatic mixed precision is available on {preferred}", {}))
    elif has_accelerator:
        checks.append(make_check(STATUS_WARN, f"Accelerator {preferred} is available but AMP API probe failed", {}))
    else:
        checks.append(make_check(STATUS_SKIP, "AMP checks skipped because no accelerator backend is active", {}))

    bf16_supported = False
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available() and hasattr(torch_module.cuda, "is_bf16_supported"):
        try:
            bf16_supported = torch_module.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            bf16_supported = torch_module.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False

    if bf16_supported:
        checks.append(make_check(STATUS_OK, "BF16 is supported on the active CUDA stack", {}))
    elif preferred.startswith("cuda"):
        checks.append(make_check(STATUS_WARN, "BF16 is not supported on the active CUDA stack", {}))
    else:
        checks.append(make_check(STATUS_SKIP, "BF16 probe requires CUDA-capable PyTorch runtime", {}))

    tf32_supported = False
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        try:
            tf32_supported = any(
                torch_module.cuda.get_device_properties(index).major >= 8
                for index in range(torch_module.cuda.device_count())
            )
        except Exception:
            tf32_supported = False

    if tf32_supported:
        checks.append(make_check(STATUS_OK, "TF32 is available on at least one CUDA device", {}))
    elif preferred.startswith("cuda"):
        checks.append(make_check(STATUS_WARN, "TF32 is unavailable on the detected CUDA devices", {}))
    else:
        checks.append(make_check(STATUS_SKIP, "TF32 probe requires CUDA-capable PyTorch runtime", {}))

    return checks


def collect_feature_checks(torch_module: Optional[Any], verbose: bool) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    if torch_module is None:
        return [make_check(STATUS_SKIP, "PyTorch-specific feature checks skipped", {})]

    compile_api = callable(getattr(torch_module, "compile", None))
    if compile_api:
        checks.append(make_check(STATUS_OK, "torch.compile API is available", {"runtime_validated": False}))
    else:
        checks.append(make_check(STATUS_WARN, "torch.compile API is unavailable", {}))

    if hasattr(torch_module.backends, "cudnn"):
        try:
            available = torch_module.backends.cudnn.is_available()
            version = torch_module.backends.cudnn.version()
            status = STATUS_OK if available else STATUS_WARN
            summary = f"cuDNN {'available' if available else 'unavailable'}"
            details = {"version": version}
            checks.append(make_check(status, summary, details))
        except Exception as exc:
            checks.append(make_check(STATUS_WARN, "cuDNN probe failed", {"error": str(exc)}))

    if verbose and hasattr(torch_module.backends, "cuda"):
        try:
            checks.append(
                make_check(
                    STATUS_OK,
                    "CUDA matmul backend flags collected",
                    {
                        "allow_tf32": getattr(torch_module.backends.cuda.matmul, "allow_tf32", None),
                        "allow_fp16_reduced_precision_reduction": getattr(
                            torch_module.backends.cuda.matmul,
                            "allow_fp16_reduced_precision_reduction",
                            None,
                        ),
                    },
                )
            )
        except Exception as exc:
            checks.append(make_check(STATUS_WARN, "CUDA backend flag probe failed", {"error": str(exc)}))

    return checks


def choose_runtime_device(device_info: Dict[str, Any]) -> str:
    preferred = device_info.get("preferred", "cpu")
    if preferred:
        return preferred
    return "cpu"


def collect_runtime_checks(torch_module: Optional[Any], device_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    if torch_module is None:
        return [make_check(STATUS_SKIP, "Runtime validation skipped because PyTorch is unavailable", {})]

    runtime_device = choose_runtime_device(device_info)
    checks: List[Dict[str, Any]] = []

    try:
        device = torch_module.device(runtime_device)
        tensor_a = torch_module.randn((64, 64), device=device)
        tensor_b = torch_module.randn((64, 64), device=device)
        result = tensor_a @ tensor_b
        checks.append(
            make_check(
                STATUS_OK,
                f"FP32 matmul succeeded on {runtime_device}",
                {"shape": list(result.shape), "dtype": str(result.dtype)},
            )
        )
    except Exception as exc:
        checks.append(make_check(STATUS_FAIL, f"FP32 runtime validation failed on {runtime_device}", {"error": str(exc)}))
        return checks

    amp_context, amp_name = get_amp_context(torch_module, runtime_device)
    if amp_context is None:
        checks.append(make_check(STATUS_SKIP, f"Mixed precision runtime validation skipped on {runtime_device}", {}))
        return checks

    try:
        if device.type == "cpu":
            target_dtype = torch_module.bfloat16
        else:
            target_dtype = torch_module.float16
        with amp_context(device_type=device.type, dtype=target_dtype):
            mixed = (tensor_a @ tensor_b).sum()
        checks.append(
            make_check(
                STATUS_OK,
                f"{amp_name} runtime validation succeeded on {runtime_device}",
                {"dtype": str(mixed.dtype)},
            )
        )
    except Exception as exc:
        checks.append(
            make_check(
                STATUS_WARN,
                f"{amp_name} runtime validation failed on {runtime_device}",
                {"error": str(exc)},
            )
        )

    return checks


def get_amp_context(torch_module: Any, runtime_device: str) -> Tuple[Optional[Any], str]:
    device_type = torch_module.device(runtime_device).type

    if device_type not in {"cuda", "cpu", "mps", "xpu"}:
        return None, "autocast"

    try:
        module = importlib.import_module("torch.amp")
        return module.autocast, "torch.amp.autocast"
    except Exception:
        pass

    if device_type == "cuda":
        try:
            module = importlib.import_module("torch.cuda.amp")
            return module.autocast, "torch.cuda.amp.autocast"
        except Exception:
            return None, "autocast"

    return None, "autocast"


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    counts = {STATUS_OK: 0, STATUS_WARN: 0, STATUS_SKIP: 0, STATUS_FAIL: 0}
    for section in ("frameworks", "precision", "features", "runtime"):
        value = result.get(section, {})
        items = value.values() if isinstance(value, dict) else value
        for item in items:
            status = item.get("status", STATUS_WARN)
            counts[status] = counts.get(status, 0) + 1

    overall = STATUS_OK
    if counts[STATUS_FAIL] > 0:
        overall = STATUS_FAIL
    elif counts[STATUS_WARN] > 0:
        overall = STATUS_WARN

    return {
        "overall": overall,
        "counts": counts,
    }


def render_report(result: Dict[str, Any], verbose: bool) -> str:
    lines: List[str] = []
    summary = result["summary"]
    lines.append("=== Deep Learning Environment Report ===")
    lines.append(f"Overall: {summary['overall']}")
    lines.append(
        "Checks: "
        f"OK={summary['counts'][STATUS_OK]} "
        f"WARN={summary['counts'][STATUS_WARN]} "
        f"SKIP={summary['counts'][STATUS_SKIP]} "
        f"FAIL={summary['counts'][STATUS_FAIL]}"
    )
    lines.append("")

    lines.append("[System]")
    for key, value in result["system"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("[Devices]")
    lines.append(f"- preferred: {result['devices']['preferred']}")
    for device in result["devices"]["devices"]:
        label = device.get("type", "device")
        name = device.get("name", "unknown")
        details = []
        for key in ("memory_gb", "memory", "compute_capability", "metal", "cores", "vendor"):
            if key in device and device[key] is not None:
                details.append(f"{key}={device[key]}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {label}: {name}{suffix}")
    for note in result["devices"].get("notes", []):
        lines.append(f"- note: {note}")
    lines.append("")

    if result.get("frameworks"):
        lines.append("[Frameworks]")
        for name, item in result["frameworks"].items():
            lines.append(f"- [{item['status']}] {name}: {item['summary']}")
            if verbose and item["details"]:
                for detail_key, detail_value in item["details"].items():
                    lines.append(f"  {detail_key}: {detail_value}")
        lines.append("")

    lines.append("[Precision]")
    for item in result["precision"]:
        lines.append(f"- [{item['status']}] {item['summary']}")
        if verbose and item["details"]:
            for detail_key, detail_value in item["details"].items():
                lines.append(f"  {detail_key}: {detail_value}")
    lines.append("")

    lines.append("[Features]")
    for item in result["features"]:
        lines.append(f"- [{item['status']}] {item['summary']}")
        if verbose and item["details"]:
            for detail_key, detail_value in item["details"].items():
                lines.append(f"  {detail_key}: {detail_value}")
    lines.append("")

    lines.append("[Runtime Validation]")
    for item in result["runtime"]:
        lines.append(f"- [{item['status']}] {item['summary']}")
        if verbose and item["details"]:
            for detail_key, detail_value in item["details"].items():
                lines.append(f"  {detail_key}: {detail_value}")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    frameworks = normalize_frameworks(args.frameworks)
    torch_module, _ = safe_import("torch")

    result: Dict[str, Any] = {
        "system": collect_system_info(),
        "devices": collect_devices(torch_module),
        "frameworks": {},
        "precision": [],
        "features": [],
        "runtime": [],
    }

    if not args.device_only:
        result["frameworks"] = collect_framework_checks(torch_module, frameworks, args.verbose)
        result["precision"] = collect_precision_checks(torch_module, result["devices"])
        result["features"] = collect_feature_checks(torch_module, args.verbose)
        if args.no_runtime_check:
            result["runtime"] = [make_check(STATUS_SKIP, "Runtime validation disabled by user", {})]
        else:
            result["runtime"] = collect_runtime_checks(torch_module, result["devices"])
    else:
        result["precision"] = [make_check(STATUS_SKIP, "Skipped due to --device-only", {})]
        result["features"] = [make_check(STATUS_SKIP, "Skipped due to --device-only", {})]
        result["runtime"] = [make_check(STATUS_SKIP, "Skipped due to --device-only", {})]

    result["summary"] = summarize(result)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_report(result, args.verbose))

    return 0 if result["summary"]["overall"] != STATUS_FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
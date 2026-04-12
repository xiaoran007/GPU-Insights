#!/usr/bin/env python3

"""Check and install GPU-Insights dependencies."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs-src"
DOCS_PACKAGE_JSON = DOCS_DIR / "package.json"
BACKEND_CHOICES = ("auto", "cpu", "cuda", "mps", "npu", "musa", "tpu")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from scripts.probe_benchmark_env import probe_benchmark_env
except Exception:
    probe_benchmark_env = None


@dataclass(frozen=True)
class PythonDependency:
    package_name: str
    import_name: str
    required: bool
    reason: str
    backends: Tuple[str, ...] = ()


@dataclass
class DependencyStatus:
    package_name: str
    import_name: str
    required: bool
    reason: str
    installed: bool
    version: Optional[str]
    selected: bool


PYTHON_DEPENDENCIES: Tuple[PythonDependency, ...] = (
    PythonDependency(
        package_name="torch",
        import_name="torch",
        required=True,
        reason="Core benchmark runtime, model execution, and backend detection.",
    ),
    PythonDependency(
        package_name="tqdm",
        import_name="tqdm",
        required=True,
        reason="Progress bars in benchmark runners.",
    ),
    PythonDependency(
        package_name="pynvml",
        import_name="pynvml",
        required=False,
        reason="CUDA calibration and richer NVIDIA metadata probing.",
        backends=("cuda",),
    ),
    PythonDependency(
        package_name="torch_xla",
        import_name="torch_xla",
        required=True,
        reason="TPU backend support.",
        backends=("tpu",),
    ),
    PythonDependency(
        package_name="torch_npu",
        import_name="torch_npu",
        required=True,
        reason="Huawei Ascend NPU backend support.",
        backends=("npu",),
    ),
    PythonDependency(
        package_name="torch_musa",
        import_name="torch_musa",
        required=True,
        reason="Moore Threads MUSA backend support.",
        backends=("musa",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and install GPU-Insights dependencies.")
    parser.add_argument(
        "-d",
        "--device",
        choices=BACKEND_CHOICES,
        default="auto",
        help="Backend to target. 'auto' follows the repo's environment probing as closely as possible.",
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        default=False,
        help="Also ensure docs-src frontend dependencies via npm.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        help="Only report dependency status, do not install anything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the commands that would run without changing the environment.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON instead of a text report.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for pip installs. Defaults to the current interpreter.",
    )
    return parser.parse_args()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _dist_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_docs_manifest() -> Dict[str, object]:
    if not DOCS_PACKAGE_JSON.exists():
        return {}
    try:
        with DOCS_PACKAGE_JSON.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _has_tpu_env_hint() -> bool:
    hint_vars = ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_WORKER_ID", "XRT_TPU_CONFIG")
    return any(os.environ.get(name) for name in hint_vars)


def _fallback_backend_probe() -> Tuple[str, str, Dict[str, object]]:
    payload: Dict[str, object] = {
        "platform": platform.platform(),
        "backend": "cpu",
        "note": "Fallback dependency probe without an initialized benchmark runtime.",
    }

    if shutil.which("nvidia-smi"):
        payload["backend"] = "cuda"
        return "cuda", "hardware heuristic (nvidia-smi)", payload

    if _module_available("torch_npu"):
        payload["backend"] = "npu"
        return "npu", "installed module heuristic (torch_npu)", payload

    if _module_available("torch_musa"):
        payload["backend"] = "musa"
        return "musa", "installed module heuristic (torch_musa)", payload

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        payload["backend"] = "mps"
        return "mps", "platform heuristic (macOS host)", payload

    if _module_available("torch_xla") or _has_tpu_env_hint():
        payload["backend"] = "tpu"
        return "tpu", "TPU heuristic", payload

    return "cpu", "fallback heuristic", payload


def detect_backend(requested_backend: str) -> Tuple[str, str, Dict[str, object]]:
    if requested_backend != "auto":
        payload: Dict[str, object] = {"backend": requested_backend, "note": "Explicit backend override."}
        if probe_benchmark_env is not None:
            try:
                payload = probe_benchmark_env(requested_backend=requested_backend, device_ids=[0])
            except Exception:
                pass
        payload["backend"] = requested_backend
        return requested_backend, "explicit request", payload

    if probe_benchmark_env is not None:
        try:
            payload = probe_benchmark_env(requested_backend="auto", device_ids=[0])
            backend = str(payload.get("backend", "cpu"))
            if backend != "cpu":
                return backend, "probe_benchmark_env(auto)", payload
        except Exception:
            pass

    return _fallback_backend_probe()


def collect_python_statuses(selected_backend: str) -> List[DependencyStatus]:
    statuses: List[DependencyStatus] = []
    for dependency in PYTHON_DEPENDENCIES:
        selected = not dependency.backends or selected_backend in dependency.backends
        installed = _module_available(dependency.import_name)
        statuses.append(
            DependencyStatus(
                package_name=dependency.package_name,
                import_name=dependency.import_name,
                required=dependency.required,
                reason=dependency.reason,
                installed=installed,
                version=_dist_version(dependency.package_name) if installed else None,
                selected=selected,
            )
        )
    return statuses


def build_python_install_plan(statuses: Sequence[DependencyStatus]) -> List[str]:
    return [
        status.package_name
        for status in statuses
        if status.selected and not status.installed
    ]


def build_docs_plan(include_docs: bool) -> Dict[str, object]:
    manifest = _load_docs_manifest()
    declared_runtime = manifest.get("dependencies", {}) if isinstance(manifest, dict) else {}
    declared_dev = manifest.get("devDependencies", {}) if isinstance(manifest, dict) else {}
    npm_path = shutil.which("npm")

    return {
        "selected": include_docs,
        "available": bool(npm_path),
        "npm_path": npm_path,
        "docs_dir": str(DOCS_DIR),
        "declared_runtime_count": len(declared_runtime) if isinstance(declared_runtime, dict) else 0,
        "declared_dev_count": len(declared_dev) if isinstance(declared_dev, dict) else 0,
        "lockfile_present": (DOCS_DIR / "package-lock.json").exists(),
        "node_modules_present": (DOCS_DIR / "node_modules").exists(),
    }


def build_commands(args: argparse.Namespace, python_plan: Sequence[str], docs_plan: Dict[str, object]) -> List[Dict[str, object]]:
    commands: List[Dict[str, object]] = []
    if python_plan:
        commands.append(
            {
                "kind": "python",
                "cwd": str(ROOT_DIR),
                "argv": [args.python, "-m", "pip", "install", *python_plan],
            }
        )

    if docs_plan["selected"]:
        argv = ["npm", "ci"]
        commands.append(
            {
                "kind": "docs",
                "cwd": str(DOCS_DIR),
                "argv": argv,
            }
        )

    return commands


def run_commands(commands: Sequence[Dict[str, object]], dry_run: bool) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for command in commands:
        argv = [str(part) for part in command["argv"]]
        cwd = str(command["cwd"])
        if dry_run:
            results.append(
                {
                    "kind": command["kind"],
                    "cwd": cwd,
                    "argv": argv,
                    "status": "dry-run",
                    "returncode": 0,
                }
            )
            continue

        completed = subprocess.run(argv, cwd=cwd, check=False)
        results.append(
            {
                "kind": command["kind"],
                "cwd": cwd,
                "argv": argv,
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
            }
        )
        if completed.returncode != 0:
            break
    return results


def render_text_report(report: Dict[str, object]) -> str:
    env = report["environment"]
    python_statuses = report["python_dependencies"]
    docs = report["docs"]
    commands = report["commands"]
    results = report["results"]

    lines: List[str] = []
    lines.append("=== GPU-Insights Dependency Report ===")
    lines.append(f"Backend: {env['backend']} ({env['detection_source']})")
    lines.append(f"Python:  {env['python_executable']}")
    if env.get("probe_payload"):
        payload = env["probe_payload"]
        device = payload.get("device")
        platform_label = payload.get("platform")
        if device:
            lines.append(f"Device:  {device}")
        if platform_label:
            lines.append(f"Host:    {platform_label}")
    lines.append("")

    lines.append("[Python Packages]")
    for status in python_statuses:
        scope = "selected" if status["selected"] else "not-selected"
        state = "installed" if status["installed"] else "missing"
        version = f" ({status['version']})" if status["version"] else ""
        requirement = "required" if status["required"] else "optional"
        lines.append(
            f"- {status['package_name']}: {state}{version} [{requirement}, {scope}]"
        )
        lines.append(f"  reason: {status['reason']}")
    lines.append("")

    lines.append("[Docs]")
    lines.append(
        f"- selected={docs['selected']} npm_available={docs['available']} "
        f"runtime_packages={docs['declared_runtime_count']} dev_packages={docs['declared_dev_count']}"
    )
    lines.append(
        f"- lockfile_present={docs['lockfile_present']} node_modules_present={docs['node_modules_present']}"
    )
    lines.append("")

    lines.append("[Plan]")
    if commands:
        for command in commands:
            lines.append(f"- ({command['kind']}) cd {command['cwd']} && {' '.join(command['argv'])}")
    else:
        lines.append("- No installation commands needed.")
    lines.append("")

    lines.append("[Execution]")
    if results:
        for result in results:
            lines.append(
                f"- {result['status']}: {' '.join(result['argv'])} (cwd={result['cwd']})"
            )
    else:
        lines.append("- No commands executed.")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    backend, detection_source, probe_payload = detect_backend(args.device)
    python_statuses = collect_python_statuses(backend)
    python_plan = build_python_install_plan(python_statuses)
    docs_plan = build_docs_plan(args.include_docs)

    commands = build_commands(args, python_plan, docs_plan)
    results: List[Dict[str, object]] = []

    hard_errors: List[str] = []
    if docs_plan["selected"] and not docs_plan["available"]:
        hard_errors.append("npm is required for --include-docs but was not found in PATH.")

    if hard_errors:
        report = {
            "environment": {
                "backend": backend,
                "detection_source": detection_source,
                "probe_payload": probe_payload,
                "python_executable": args.python,
            },
            "python_dependencies": [asdict(status) for status in python_statuses],
            "docs": docs_plan,
            "commands": commands,
            "results": [],
            "errors": hard_errors,
        }
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(render_text_report(report))
            for error in hard_errors:
                print(f"\nERROR: {error}")
        return 1

    if not args.check_only:
        results = run_commands(commands, dry_run=args.dry_run)
        if results and results[-1]["status"] == "failed":
            final_code = 1
        else:
            final_code = 0
    else:
        final_code = 1 if any(status.selected and status.required and not status.installed for status in python_statuses) else 0

    if not args.check_only and not args.dry_run and final_code == 0:
        python_statuses = collect_python_statuses(backend)
        python_plan = build_python_install_plan(python_statuses)

    report = {
        "environment": {
            "backend": backend,
            "detection_source": detection_source,
            "probe_payload": probe_payload,
            "python_executable": args.python,
        },
        "python_dependencies": [asdict(status) for status in python_statuses],
        "docs": docs_plan,
        "commands": commands,
        "results": results,
        "pending_python_packages": python_plan,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_text_report(report))

    if not args.check_only and not args.dry_run and python_plan:
        return 1
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from benchmark.Bench import Bench
from benchmark.launch_plan import LaunchPlan, build_launch_plan
from benchmark.models import list_models, resolve_model_name
from scripts.probe_benchmark_env import probe_benchmark_env


DEFAULT_MAJOR_MODELS = ["resnet50", "vit", "unet", "ddpm"]
PAYLOAD_PREFIX = "RESULT_PAYLOAD_B64="
PAYLOAD_SCHEMA_VERSION = "1.0"


@dataclass
class RunRecord:
    model: str
    requested_dtype: str
    actual_dtype: str
    mode: str
    batch_size: Optional[int]
    throughput: Optional[float]
    score: Optional[float]
    status: str
    device_name: str
    backend: str
    world_size: int
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU-Insights smart benchmark launcher")
    parser.add_argument(
        "-mt", "--model", required=False, type=str,
        help=f"Model to benchmark. Omit to run: {', '.join(DEFAULT_MAJOR_MODELS)}. Available: {', '.join(list_models())}",
    )
    parser.add_argument(
        "-s", "--size", type=int, default=1024,
        help="Data size in MB.",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="auto",
        help="Device backend: auto, cuda, mps, npu, musa, tpu.",
    )
    parser.add_argument(
        "-gpu", "--gpu_id", type=str, default="all",
        help="CUDA GPU ids to use, e.g. 'all' or '0,1'. Ignored on non-CUDA backends.",
    )
    parser.add_argument(
        "-dt", "--dtype", type=str, default=None,
        help="Override automatic precision selection with a single dtype: FP32, FP16, or BF16.",
    )
    parser.add_argument(
        "--no-abs", action="store_true", default=False,
        help="Disable automatic batch size selection.",
    )
    parser.add_argument(
        "-bs", "--batch", type=int, default=0,
        help="Batch size override. When set, automatic batch size is disabled.",
    )
    parser.add_argument(
        "--single-process", action="store_true", default=False,
        help="Force single-process mode even if multiple CUDA GPUs are available.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print the planned runs without executing them.",
    )
    parser.add_argument(
        "-cudnn", "--cudnn_benchmark", action="store_true", default=False,
        help="Enable cuDNN benchmark mode for single-process runs.",
    )
    return parser


def format_batch_source(plan: LaunchPlan) -> str:
    if plan.batch_size_override is not None:
        return f"explicit ({plan.batch_size_override})"
    if plan.auto_batch_size:
        return "auto-batch-size"
    return "model-default"


def print_plan(plan: LaunchPlan) -> None:
    print(f"\n{'=' * 68}")
    print("GPU-Insights Smart Launch Plan")
    print(f"{'=' * 68}")
    print(f"Model:            {plan.model}")
    print(f"Backend:          {plan.backend}")
    print(f"Device IDs:       {','.join(str(i) for i in plan.device_ids)}")
    print(f"Execution Mode:   {'ddp' if plan.use_ddp else 'single'}")
    print(f"World Size:       {plan.world_size}")
    print(f"Precisions:       {', '.join(plan.precisions)}")
    print(f"Data Size (MB):   {plan.data_size_mb}")
    print(f"Epochs:           {plan.epochs}")
    print(f"Batch Source:     {format_batch_source(plan)}")
    print(f"{'=' * 68}\n")


def print_batch_header(models: list[str]) -> None:
    print(f"\n{'#' * 88}")
    print("GPU-Insights Smart Batch Run")
    print(f"{'#' * 88}")
    print(f"Models:           {', '.join(models)}")
    print(f"Model Count:      {len(models)}")
    print(f"{'#' * 88}\n")


def print_run_header(plan: LaunchPlan, dtype: str) -> None:
    print(f"\n{'-' * 68}")
    print(f"Running precision: {dtype}")
    print(f"Model:            {plan.model}")
    print(f"Backend:          {plan.backend}")
    print(f"Device IDs:       {','.join(str(i) for i in plan.device_ids)}")
    print(f"Execution Mode:   {'ddp' if plan.use_ddp else 'single'}")
    print(f"ABS Enabled:      {'yes' if plan.auto_batch_size else 'no'}")
    print(f"Batch Source:     {format_batch_source(plan)}")
    print(f"{'-' * 68}")


def make_record_from_result(model: str, requested_dtype: str, result_dict: dict) -> RunRecord:
    extra = result_dict.get("extra", {})
    return RunRecord(
        model=model,
        requested_dtype=requested_dtype,
        actual_dtype=extra.get("dtype", requested_dtype),
        mode=extra.get("mode", "single"),
        batch_size=result_dict.get("batch_size"),
        throughput=result_dict.get("throughput"),
        score=result_dict.get("score"),
        status="ok",
        device_name=result_dict.get("device_name", ""),
        backend=extra.get("backend", "unknown"),
        world_size=extra.get("world_size", 1),
    )


def run_single_precision(plan: LaunchPlan, args, dtype: str) -> RunRecord:
    bench = Bench(
        device=plan.backend,
        size=plan.data_size_mb,
        epochs=plan.epochs,
        method=plan.model,
        batch_size=plan.batch_size_override or 0,
        cudnn_benchmark=args.cudnn_benchmark,
        data_type=dtype,
        gpu_ids=plan.device_ids,
        auto_batch_size=plan.auto_batch_size,
    )
    result = bench.start()
    return make_record_from_result(plan.model, dtype, {
        "score": result.score,
        "throughput": result.throughput,
        "batch_size": result.batch_size,
        "device_name": result.device_name,
        "extra": result.extra,
    })


def build_ddp_command(plan: LaunchPlan, dtype: str, cudnn_benchmark: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={plan.world_size}",
        "main_ddp.py",
        "-mt",
        plan.model,
        "-s",
        str(plan.data_size_mb),
        "-e",
        str(plan.epochs),
        "-dt",
        dtype,
    ]
    if plan.auto_batch_size:
        command.append("-abs")
    if plan.batch_size_override is not None:
        command.extend(["-bs", str(plan.batch_size_override)])
    if cudnn_benchmark:
        command.append("-cudnn")
    return command


def run_ddp_precision(plan: LaunchPlan, dtype: str, args) -> RunRecord:
    fd, result_path = tempfile.mkstemp(prefix="gpu_insights_", suffix=".json")
    os.close(fd)

    env = os.environ.copy()
    env["GPU_INSIGHTS_RESULT_JSON"] = result_path
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in plan.device_ids)
    command = build_ddp_command(plan, dtype, cudnn_benchmark=args.cudnn_benchmark)

    try:
        completed = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            check=False,
        )
        result_dict = {}
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                result_dict = json.load(f)

        if completed.returncode == 0 and result_dict:
            return make_record_from_result(plan.model, dtype, result_dict)

        error = f"DDP subprocess exited with code {completed.returncode}"
        if result_dict:
            record = make_record_from_result(plan.model, dtype, result_dict)
            record.status = "failed"
            record.error = error
            return record
        return RunRecord(
            model=plan.model,
            requested_dtype=dtype,
            actual_dtype=dtype,
            mode="ddp",
            batch_size=None,
            throughput=None,
            score=None,
            status="failed",
            device_name="",
            backend=plan.backend,
            world_size=plan.world_size,
            error=error,
        )
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)


def print_dry_run_command(plan: LaunchPlan, dtype: str, args) -> None:
    if plan.use_ddp:
        print(f"DDP command for {dtype}: {' '.join(build_ddp_command(plan, dtype, cudnn_benchmark=args.cudnn_benchmark))}")
        print(f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in plan.device_ids)}")
    else:
        print(f"Single-process run for {dtype}: Bench(device='{plan.backend}', gpu_ids={plan.device_ids})")


def print_summary(records: list[RunRecord]) -> None:
    print(f"\n{'=' * 88}")
    print("Smart Launcher Summary")
    print(f"{'=' * 88}")
    print(f"{'model':<12}{'dtype':<10}{'mode':<10}{'batch':<10}{'throughput':<16}{'score':<12}{'status':<10}")
    for record in records:
        batch = str(record.batch_size) if record.batch_size is not None else "-"
        throughput = f"{record.throughput:.1f}" if record.throughput is not None else "-"
        score = f"{record.score:.0f}" if record.score is not None else "-"
        print(
            f"{record.model:<12}"
            f"{record.actual_dtype:<10}"
            f"{record.mode:<10}"
            f"{batch:<10}"
            f"{throughput:<16}"
            f"{score:<12}"
            f"{record.status:<10}"
        )
    print(f"{'=' * 88}")
    for record in records:
        if record.error:
            print(f"{record.model}/{record.actual_dtype}: {record.error}")
    print("")


def resolve_target_models(raw_model: Optional[str]) -> list[str]:
    if raw_model:
        return [resolve_model_name(raw_model)]
    return DEFAULT_MAJOR_MODELS.copy()


def _format_result_date() -> str:
    now = datetime.now()
    return f"{now.year}.{now.month}.{now.day}"


def _build_result_note(host_info: dict, precision_records: list[RunRecord]) -> str:
    note_parts = ["main_auto ver.2"]
    if host_info.get("driver_runtime"):
        note_parts.append(host_info["driver_runtime"])
    if any(record.actual_dtype == "BF16" and record.status == "ok" for record in precision_records):
        note_parts.append("BF16 stored in fp16 field")
    if host_info.get("note"):
        note_parts.append(host_info["note"])
    return "; ".join(part for part in note_parts if part)


def _build_benchmark_entry(model: str, host_info: dict, precision_records: list[RunRecord]) -> dict:
    fp32_record = next(
        (record for record in precision_records if record.actual_dtype == "FP32" and record.status == "ok"),
        None,
    )
    fp16_record = next(
        (record for record in precision_records if record.actual_dtype in ("FP16", "BF16") and record.status == "ok"),
        None,
    )
    failed_records = [record for record in precision_records if record.status != "ok"]
    return {
        "model": model,
        "vendor": host_info["vendor"],
        "architecture": host_info["architecture"],
        "device": host_info["device"],
        "memory": host_info["memory"],
        "platform": host_info["platform"],
        "fp32": fp32_record.score if fp32_record is not None else None,
        "fp32bs": fp32_record.batch_size if fp32_record and fp32_record.batch_size is not None else 0,
        "fp16": fp16_record.score if fp16_record is not None else None,
        "fp16bs": fp16_record.batch_size if fp16_record and fp16_record.batch_size is not None else 0,
        "note": _build_result_note(host_info, precision_records),
        "date": _format_result_date(),
        "version": "ver2",
        "status": "failed" if fp32_record is None and fp16_record is None else ("partial" if failed_records else "ok"),
        "precision_results": [
            {
                "requested_dtype": record.requested_dtype,
                "actual_dtype": record.actual_dtype,
                "status": record.status,
                "score": record.score,
                "batch_size": record.batch_size,
                "mode": record.mode,
                "backend": record.backend,
                "world_size": record.world_size,
                "error": record.error,
            }
            for record in precision_records
        ],
    }


def build_result_payload(records: list[RunRecord], plans: list[LaunchPlan], host_info: dict, target_models: list[str]) -> dict:
    entries = []
    for model in target_models:
        model_records = [record for record in records if record.model == model]
        entries.append(_build_benchmark_entry(model, host_info, model_records))

    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": {
            "launcher": "main_auto.py",
            "models_requested": target_models,
            "effective_backends": sorted({plan.backend for plan in plans}),
            "execution_modes": sorted({"ddp" if plan.use_ddp else "single" for plan in plans}),
        },
        "host": {
            "vendor": host_info["vendor"],
            "architecture": host_info["architecture"],
            "device": host_info["device"],
            "memory": host_info["memory"],
            "platform": host_info["platform"],
            "driver_runtime": host_info["driver_runtime"],
            "note": host_info["note"],
        },
        "benchmarks": entries,
    }


def print_result_payload(payload: dict) -> None:
    encoded = base64.b64encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    print(f"{PAYLOAD_PREFIX}{encoded}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_models = resolve_target_models(args.model)
    plans = []
    for model in target_models:
        try:
            plans.append(
                build_launch_plan(
                    model=model,
                    size=args.size,
                    epochs=args.epochs,
                    device=args.device,
                    gpu_id=args.gpu_id,
                    requested_dtype=args.dtype,
                    batch=args.batch,
                    no_abs=args.no_abs,
                    single_process=args.single_process,
                    allow_unavailable=args.dry_run,
                )
            )
        except Exception as exc:
            print(f"Error while building launch plan for {model}: {exc}")
            return 1

    if len(target_models) > 1:
        print_batch_header(target_models)
    records = []
    for plan in plans:
        print_plan(plan)
        if args.dry_run:
            for dtype in plan.precisions:
                print_run_header(plan, dtype)
                print_dry_run_command(plan, dtype, args)
            continue

        for dtype in plan.precisions:
            print_run_header(plan, dtype)
            try:
                if plan.use_ddp:
                    record = run_ddp_precision(plan, dtype, args)
                else:
                    record = run_single_precision(plan, args, dtype)
            except Exception as exc:
                record = RunRecord(
                    model=plan.model,
                    requested_dtype=dtype,
                    actual_dtype=dtype,
                    mode="ddp" if plan.use_ddp else "single",
                    batch_size=None,
                    throughput=None,
                    score=None,
                    status="failed",
                    device_name="",
                    backend=plan.backend,
                    world_size=plan.world_size,
                    error=str(exc),
                )
            records.append(record)

    if args.dry_run:
        return 0

    host_info = probe_benchmark_env(
        requested_backend=plans[0].backend if plans else args.device,
        device_ids=plans[0].device_ids if plans else [0],
    )

    had_failure = False
    for record in records:
        if record.status != "ok":
            had_failure = True

    print_summary(records)
    print_result_payload(build_result_payload(records, plans, host_info, target_models))
    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())

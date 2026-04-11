#!/usr/bin/env python3
"""
GPU Benchmark Data Management Script

This script validates and updates the dashboard benchmark JSON file.
The primary workflow is importing the `RESULT_PAYLOAD_B64=...` line
emitted by `main_auto.py`.
"""

import argparse
import base64
import binascii
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_FILE = SCRIPT_DIR.parent / "docs-src" / "public" / "data" / "benchmark-data.json"
PAYLOAD_PREFIX = "RESULT_PAYLOAD_B64="

VALID_VENDORS = {"nvidia", "amd", "intel", "apple", "huawei", "mthreads", "google", "unknown"}
VALID_MODELS = {"resnet50", "cnn", "vit", "unet", "ddpm"}
VALID_VERSIONS = {"ver1", "ver2"}
VER2_NOTE_TOKEN = "ver.2"


def current_date_string() -> str:
    now = datetime.now()
    return f"{now.year}.{now.month}.{now.day}"


def infer_version_from_note(note: str) -> str:
    """Infer benchmark version from note content."""
    note_text = (note or "").lower()
    return "ver2" if VER2_NOTE_TOKEN in note_text else "ver1"


def resolve_data_file(args: argparse.Namespace) -> Path:
    return Path(args.data_file).expanduser().resolve()


def load_data(data_file: Path) -> Optional[dict]:
    """Load the benchmark data JSON file."""
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file}")
        return None
    except json.JSONDecodeError as exc:
        print(f"❌ JSON parsing error: {exc}")
        return None


def save_data(data_file: Path, data: dict) -> bool:
    """Save the benchmark data JSON file."""
    try:
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"❌ Failed to save data: {exc}")
        return False


def validate_entry(entry: dict) -> list[str]:
    """Validate a single benchmark entry."""
    errors = []

    required_fields = [
        "model",
        "vendor",
        "architecture",
        "device",
        "memory",
        "platform",
        "fp32",
        "fp32bs",
        "fp16",
        "fp16bs",
        "note",
        "date",
        "version",
    ]

    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    if "model" in entry and entry["model"] not in VALID_MODELS:
        errors.append(f"Invalid model '{entry['model']}'. Must be one of: {', '.join(sorted(VALID_MODELS))}")

    if "vendor" in entry and entry["vendor"] not in VALID_VENDORS:
        errors.append(f"Invalid vendor '{entry['vendor']}'. Must be one of: {', '.join(sorted(VALID_VENDORS))}")

    if "version" in entry and entry["version"] not in VALID_VERSIONS:
        errors.append(f"Invalid version '{entry['version']}'. Must be one of: {', '.join(sorted(VALID_VERSIONS))}")

    if "fp32" in entry and entry["fp32"] is not None and not isinstance(entry["fp32"], (int, float)):
        errors.append("fp32 must be a number or null")

    if "fp16" in entry and entry["fp16"] is not None and not isinstance(entry["fp16"], (int, float)):
        errors.append("fp16 must be a number or null")

    if "fp32bs" in entry and not isinstance(entry["fp32bs"], int):
        errors.append("fp32bs must be an integer")

    if "fp16bs" in entry and not isinstance(entry["fp16bs"], int):
        errors.append("fp16bs must be an integer")

    if "date" in entry:
        date_parts = entry["date"].split(".")
        if len(date_parts) != 3:
            errors.append("Date must be in YYYY.M.DD format")
        else:
            try:
                year, month, day = map(int, date_parts)
                if year < 2020 or year > 2035:
                    errors.append("Year seems unreasonable")
                if month < 1 or month > 12:
                    errors.append("Invalid month")
                if day < 1 or day > 31:
                    errors.append("Invalid day")
            except ValueError:
                errors.append("Date components must be numbers")

    return errors


def validate_data_file(args: argparse.Namespace) -> bool:
    """Validate the entire data file."""
    data_file = resolve_data_file(args)
    print(f"🔍 Validating benchmark data file: {data_file}")

    data = load_data(data_file)
    if not data:
        return False

    errors = []

    if "metadata" not in data:
        errors.append("Missing 'metadata' section")

    if "models" not in data:
        errors.append("Missing 'models' section")

    if "benchmarks" not in data:
        errors.append("Missing 'benchmarks' section")
    else:
        benchmarks = data["benchmarks"]
        if not isinstance(benchmarks, list):
            errors.append("'benchmarks' must be an array")
        else:
            print(f"📊 Found {len(benchmarks)} benchmark entries")
            for i, entry in enumerate(benchmarks):
                entry_errors = validate_entry(entry)
                for error in entry_errors:
                    errors.append(f"Entry {i + 1} ({entry.get('device', 'unknown')}): {error}")

    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  • {error}")
        return False

    print("✅ Validation passed!")
    return True


def build_manual_entry(args: argparse.Namespace) -> dict:
    note = args.note or ""
    version = args.version or infer_version_from_note(note)
    return {
        "model": args.model,
        "vendor": args.vendor,
        "architecture": args.architecture,
        "device": args.device,
        "memory": args.memory or "",
        "platform": args.platform,
        "fp32": args.fp32,
        "fp32bs": args.fp32bs,
        "fp16": args.fp16,
        "fp16bs": args.fp16bs,
        "note": note,
        "date": args.date or current_date_string(),
        "version": version,
    }


def add_benchmark(args: argparse.Namespace) -> bool:
    """Add a new benchmark entry manually."""
    data_file = resolve_data_file(args)
    print(f"➕ Adding new benchmark entry to: {data_file}")

    data = load_data(data_file)
    if not data:
        return False

    new_entry = build_manual_entry(args)
    entry_errors = validate_entry(new_entry)
    if entry_errors:
        print("❌ New entry validation failed:")
        for error in entry_errors:
            print(f"  • {error}")
        return False

    data["benchmarks"].append(new_entry)
    data.setdefault("metadata", {})["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")

    if save_data(data_file, data):
        print("✅ Successfully added new benchmark entry!")
        print(f"📊 Total entries: {len(data['benchmarks'])}")
        return True
    return False


def migrate_versions(args: argparse.Namespace) -> bool:
    """Backfill benchmark version values based on note rules."""
    data_file = resolve_data_file(args)
    print(f"🛠️ Migrating benchmark version field in: {data_file}")

    data = load_data(data_file)
    if not data:
        return False

    benchmarks = data.get("benchmarks", [])
    updated = 0

    for entry in benchmarks:
        inferred = infer_version_from_note(entry.get("note", ""))
        current = entry.get("version")

        if args.force:
            if current != inferred or "version" not in entry:
                entry["version"] = inferred
                updated += 1
            continue

        if current in VALID_VERSIONS:
            continue

        entry["version"] = inferred
        updated += 1

    if updated > 0:
        data.setdefault("metadata", {})["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")
        if not save_data(data_file, data):
            return False

    version_counts = {"ver1": 0, "ver2": 0}
    for entry in benchmarks:
        version = entry.get("version")
        if version in version_counts:
            version_counts[version] += 1

    print("✅ Migration complete")
    print(f"  Total entries: {len(benchmarks)}")
    print(f"  Updated entries: {updated}")
    print(f"  ver1 entries: {version_counts['ver1']}")
    print(f"  ver2 entries: {version_counts['ver2']}")
    return True


def list_stats(args: argparse.Namespace) -> None:
    """Show statistics about the data."""
    data_file = resolve_data_file(args)
    data = load_data(data_file)
    if not data:
        return

    benchmarks = data["benchmarks"]
    print(f"📊 Benchmark Statistics ({data_file}):")
    print(f"  Total entries: {len(benchmarks)}")

    vendor_counts: Dict[str, int] = {}
    version_counts = {"ver1": 0, "ver2": 0}
    model_counts: Dict[str, int] = {}
    for entry in benchmarks:
        vendor = entry["vendor"]
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
        if entry.get("version") in version_counts:
            version_counts[entry["version"]] += 1
        model = entry.get("model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1

    print("  By model:")
    for model, count in sorted(model_counts.items()):
        print(f"    {model}: {count}")

    print("  By vendor:")
    for vendor, count in sorted(vendor_counts.items()):
        print(f"    {vendor}: {count}")

    print("  By version:")
    for version, count in sorted(version_counts.items()):
        print(f"    {version}: {count}")

    fp32_scores = [entry["fp32"] for entry in benchmarks if entry["fp32"] is not None]
    fp16_scores = [entry["fp16"] for entry in benchmarks if entry["fp16"] is not None]

    if fp32_scores:
        print(f"  FP32 scores: {min(fp32_scores)} - {max(fp32_scores)} (avg: {sum(fp32_scores) / len(fp32_scores):.0f})")

    if fp16_scores:
        print(f"  FP16 scores: {min(fp16_scores)} - {max(fp16_scores)} (avg: {sum(fp16_scores) / len(fp16_scores):.0f})")


def extract_payload_text(raw_payload: str) -> str:
    payload = raw_payload.strip()
    if payload.startswith(PAYLOAD_PREFIX):
        return payload[len(PAYLOAD_PREFIX):].strip()
    return payload


def load_payload_text(args: argparse.Namespace) -> Optional[str]:
    if args.payload:
        return args.payload
    if args.payload_file:
        return Path(args.payload_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text
    print("❌ No payload provided. Pass a positional payload, --payload-file, or pipe stdin.")
    return None


def decode_payload(raw_payload: str) -> dict:
    payload_text = extract_payload_text(raw_payload)
    try:
        decoded = base64.b64decode(payload_text, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Invalid Base64 payload: {exc}") from exc

    try:
        return json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Decoded payload is not valid UTF-8 JSON: {exc}") from exc


def build_entry_key(entry: dict) -> tuple:
    return (
        entry["model"],
        entry["vendor"],
        entry["architecture"],
        entry["device"],
        entry["memory"],
        entry["platform"],
        entry["fp32"],
        entry["fp32bs"],
        entry["fp16"],
        entry["fp16bs"],
        entry["note"],
        entry["date"],
        entry["version"],
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _normalize_note(note: str, status: Optional[str]) -> str:
    clean_note = (note or "").strip()
    if status == "partial" and "partial" not in clean_note.lower():
        return f"{clean_note}; partial precision run" if clean_note else "partial precision run"
    return clean_note


def normalize_payload_entry(payload_entry: dict, payload_host: dict) -> Optional[dict]:
    status = payload_entry.get("status", "ok")
    fp32 = payload_entry.get("fp32")
    fp16 = payload_entry.get("fp16")
    if fp32 is None and fp16 is None:
        return None

    note = _normalize_note(
        payload_entry.get("note") or payload_host.get("note") or "",
        status,
    )
    version = payload_entry.get("version") or infer_version_from_note(note)

    entry = {
        "model": payload_entry.get("model"),
        "vendor": payload_entry.get("vendor") or payload_host.get("vendor", "unknown"),
        "architecture": payload_entry.get("architecture") or payload_host.get("architecture", ""),
        "device": payload_entry.get("device") or payload_host.get("device", ""),
        "memory": payload_entry.get("memory") or payload_host.get("memory", ""),
        "platform": payload_entry.get("platform") or payload_host.get("platform", ""),
        "fp32": fp32,
        "fp32bs": _coerce_int(payload_entry.get("fp32bs"), 0),
        "fp16": fp16,
        "fp16bs": _coerce_int(payload_entry.get("fp16bs"), 0),
        "note": note,
        "date": payload_entry.get("date") or current_date_string(),
        "version": version,
    }
    return entry


def decode_payload_command(args: argparse.Namespace) -> bool:
    raw_payload = load_payload_text(args)
    if raw_payload is None:
        return False

    try:
        payload = decode_payload(raw_payload)
    except ValueError as exc:
        print(f"❌ {exc}")
        return False

    print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))
    return True


def import_payload(args: argparse.Namespace) -> bool:
    data_file = resolve_data_file(args)
    raw_payload = load_payload_text(args)
    if raw_payload is None:
        return False

    try:
        payload = decode_payload(raw_payload)
    except ValueError as exc:
        print(f"❌ {exc}")
        return False

    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        print("❌ Payload is missing a valid 'benchmarks' array.")
        return False

    payload_host = payload.get("host", {}) if isinstance(payload.get("host"), dict) else {}
    normalized_entries = []
    skipped_failed = 0
    validation_errors = []

    for index, payload_entry in enumerate(benchmarks, start=1):
        if not isinstance(payload_entry, dict):
            validation_errors.append(f"Payload benchmark {index} is not an object.")
            continue

        normalized = normalize_payload_entry(payload_entry, payload_host)
        if normalized is None:
            skipped_failed += 1
            continue

        entry_errors = validate_entry(normalized)
        if entry_errors:
            validation_errors.append(
                f"Payload benchmark {index} ({normalized.get('model', 'unknown')}): {'; '.join(entry_errors)}"
            )
            continue

        normalized_entries.append(normalized)

    if validation_errors:
        print("❌ Payload validation failed:")
        for error in validation_errors:
            print(f"  • {error}")
        return False

    data = load_data(data_file)
    if not data:
        return False

    existing_keys = {build_entry_key(entry) for entry in data.get("benchmarks", [])}
    imported = 0
    skipped_duplicates = 0

    for entry in normalized_entries:
        key = build_entry_key(entry)
        if not args.allow_duplicates and key in existing_keys:
            skipped_duplicates += 1
            continue
        data["benchmarks"].append(entry)
        existing_keys.add(key)
        imported += 1

    if imported > 0:
        data.setdefault("metadata", {})["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")

    if args.dry_run:
        print("🧪 Dry run complete. No file was modified.")
    elif imported > 0:
        if not save_data(data_file, data):
            return False

    print(f"📦 Payload models: {', '.join(sorted({entry['model'] for entry in normalized_entries})) or 'none'}")
    print(f"✅ Import summary for {data_file}:")
    print(f"  Normalized entries: {len(normalized_entries)}")
    print(f"  Imported entries: {imported}")
    print(f"  Skipped failed entries: {skipped_failed}")
    print(f"  Skipped exact duplicates: {skipped_duplicates}")

    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU Benchmark Data Management")
    parser.add_argument(
        "--data-file",
        default=str(DEFAULT_DATA_FILE),
        help=f"Benchmark data JSON file. Default: {DEFAULT_DATA_FILE}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("validate", help="Validate the data file")
    subparsers.add_parser("stats", help="Show data statistics")

    add_parser = subparsers.add_parser("add", help="Add a new benchmark entry manually")
    add_parser.add_argument("--model", required=True, choices=sorted(VALID_MODELS), help="Model name")
    add_parser.add_argument("--vendor", required=True, choices=sorted(VALID_VENDORS), help="GPU vendor")
    add_parser.add_argument("--architecture", required=True, help="GPU architecture")
    add_parser.add_argument("--device", required=True, help="Device name")
    add_parser.add_argument("--memory", help="Memory amount (e.g., 24GB)")
    add_parser.add_argument("--platform", required=True, help="Platform/OS info")
    add_parser.add_argument("--fp32", type=float, help="FP32 score")
    add_parser.add_argument("--fp32bs", type=int, required=True, help="FP32 batch size")
    add_parser.add_argument("--fp16", type=float, help="FP16 score")
    add_parser.add_argument("--fp16bs", type=int, required=True, help="FP16 batch size")
    add_parser.add_argument("--note", help="Additional notes")
    add_parser.add_argument("--date", help="Test date (YYYY.M.DD format)")
    add_parser.add_argument("--version", choices=sorted(VALID_VERSIONS), help="Result version")

    import_parser = subparsers.add_parser(
        "import-payload",
        help="Import a RESULT_PAYLOAD_B64 payload emitted by main_auto.py",
    )
    import_parser.add_argument("payload", nargs="?", help="Raw Base64 payload or full RESULT_PAYLOAD_B64=... line")
    import_parser.add_argument("--payload-file", help="Read payload text from a file")
    import_parser.add_argument("--dry-run", action="store_true", help="Normalize and validate without writing")
    import_parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Import even when an identical normalized entry already exists",
    )

    decode_parser = subparsers.add_parser(
        "decode-payload",
        help="Decode a RESULT_PAYLOAD_B64 payload and print the JSON envelope",
    )
    decode_parser.add_argument("payload", nargs="?", help="Raw Base64 payload or full RESULT_PAYLOAD_B64=... line")
    decode_parser.add_argument("--payload-file", help="Read payload text from a file")
    decode_parser.add_argument("--pretty", action="store_true", help="Pretty-print the decoded JSON")

    migrate_parser = subparsers.add_parser("migrate-version", help="Backfill or normalize benchmark version field")
    migrate_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute version for all entries even if version already exists",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "validate":
        return 0 if validate_data_file(args) else 1
    if args.command == "stats":
        list_stats(args)
        return 0
    if args.command == "add":
        return 0 if add_benchmark(args) else 1
    if args.command == "import-payload":
        return 0 if import_payload(args) else 1
    if args.command == "decode-payload":
        return 0 if decode_payload_command(args) else 1
    if args.command == "migrate-version":
        return 0 if migrate_versions(args) else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
GPU Benchmark Data Management Script

This script helps validate and manage the GPU benchmark data JSON file.
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR.parent / "docs" / "data" / "benchmark-data.json"

VALID_VENDORS = {"nvidia", "amd", "intel", "apple", "huawei", "mthreads"}
VALID_VERSIONS = {"ver1", "ver2"}
VER2_NOTE_TOKEN = "ver.2"


def infer_version_from_note(note):
    """Infer benchmark version from note content."""
    note_text = (note or "").lower()
    return "ver2" if VER2_NOTE_TOKEN in note_text else "ver1"


def load_data():
    """Load the benchmark data JSON file."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return None


def save_data(data):
    """Save the benchmark data JSON file."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
        return False


def validate_entry(entry):
    """Validate a single benchmark entry."""
    errors = []

    required_fields = [
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
                if year < 2020 or year > 2030:
                    errors.append("Year seems unreasonable")
                if month < 1 or month > 12:
                    errors.append("Invalid month")
                if day < 1 or day > 31:
                    errors.append("Invalid day")
            except ValueError:
                errors.append("Date components must be numbers")

    return errors


def validate_data_file():
    """Validate the entire data file."""
    print("🔍 Validating benchmark data file...")

    data = load_data()
    if not data:
        return False

    errors = []

    if "metadata" not in data:
        errors.append("Missing 'metadata' section")

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


def add_benchmark(args):
    """Add a new benchmark entry."""
    print("➕ Adding new benchmark entry...")

    data = load_data()
    if not data:
        return False

    note = args.note or ""
    version = args.version or infer_version_from_note(note)

    new_entry = {
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
        "date": args.date or datetime.now().strftime("%Y.%-m.%-d"),
        "version": version,
    }

    entry_errors = validate_entry(new_entry)
    if entry_errors:
        print("❌ New entry validation failed:")
        for error in entry_errors:
            print(f"  • {error}")
        return False

    data["benchmarks"].append(new_entry)
    data["metadata"]["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")

    if save_data(data):
        print("✅ Successfully added new benchmark entry!")
        print(f"📊 Total entries: {len(data['benchmarks'])}")
        return True

    return False


def migrate_versions(args):
    """Backfill benchmark version values based on note rules."""
    print("🛠️ Migrating benchmark version field...")

    data = load_data()
    if not data:
        return False

    benchmarks = data.get("benchmarks", [])
    updated = 0

    for entry in benchmarks:
        inferred = infer_version_from_note(entry.get("note", ""))
        current = entry.get("version")

        if args.force:
            if current != inferred:
                entry["version"] = inferred
                updated += 1
            elif "version" not in entry:
                entry["version"] = inferred
                updated += 1
            continue

        if current in VALID_VERSIONS:
            continue

        entry["version"] = inferred
        updated += 1

    if updated > 0:
        data.setdefault("metadata", {})["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")
        if not save_data(data):
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


def list_stats():
    """Show statistics about the data."""
    data = load_data()
    if not data:
        return

    benchmarks = data["benchmarks"]
    print("📊 Benchmark Statistics:")
    print(f"  Total entries: {len(benchmarks)}")

    vendor_counts = {}
    version_counts = {"ver1": 0, "ver2": 0}
    for entry in benchmarks:
        vendor = entry["vendor"]
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
        if entry.get("version") in version_counts:
            version_counts[entry["version"]] += 1

    print("  By vendor:")
    for vendor, count in sorted(vendor_counts.items()):
        print(f"    {vendor}: {count}")

    print("  By version:")
    for version, count in sorted(version_counts.items()):
        print(f"    {version}: {count}")

    fp32_scores = [e["fp32"] for e in benchmarks if e["fp32"] is not None]
    fp16_scores = [e["fp16"] for e in benchmarks if e["fp16"] is not None]

    if fp32_scores:
        print(f"  FP32 scores: {min(fp32_scores)} - {max(fp32_scores)} (avg: {sum(fp32_scores) / len(fp32_scores):.0f})")

    if fp16_scores:
        print(f"  FP16 scores: {min(fp16_scores)} - {max(fp16_scores)} (avg: {sum(fp16_scores) / len(fp16_scores):.0f})")


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark Data Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("validate", help="Validate the data file")
    subparsers.add_parser("stats", help="Show data statistics")

    add_parser = subparsers.add_parser("add", help="Add a new benchmark entry")
    add_parser.add_argument("--vendor", required=True, choices=VALID_VENDORS, help="GPU vendor")
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
    add_parser.add_argument("--version", choices=sorted(VALID_VERSIONS), help="Result version (ver1 or ver2)")

    migrate_parser = subparsers.add_parser("migrate-version", help="Backfill or normalize benchmark version field")
    migrate_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute version for all entries even if version already exists",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "validate":
        success = validate_data_file()
        sys.exit(0 if success else 1)
    elif args.command == "stats":
        list_stats()
    elif args.command == "add":
        success = add_benchmark(args)
        sys.exit(0 if success else 1)
    elif args.command == "migrate-version":
        success = migrate_versions(args)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate NVIDIA architecture/SKU reference data for the dashboard.

The workflow is intentionally single-command for the MVP:

    conda run -n torch python scripts/manage-nvidia-specs.py refresh

It fetches official NVIDIA HTML/PDF sources, extracts evidence-backed fields,
and writes structured JSON/CSV/report artifacts for the dashboard.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "docs-src" / "public" / "data"
REPORTS_DIR = REPO_ROOT / "reports"

OUTPUT_JSON = DATA_DIR / "nvidia-gpu-specs.json"
OUTPUT_CSV = DATA_DIR / "nvidia-gpu-specs.csv"
OUTPUT_COVERAGE = DATA_DIR / "nvidia-gpu-specs-coverage.json"
OUTPUT_MISSING = DATA_DIR / "nvidia-gpu-specs-missing-fields.json"
OUTPUT_SOURCES = DATA_DIR / "nvidia-gpu-specs-sources.json"
OUTPUT_REPORT = REPORTS_DIR / "nvidia-gpu-specs.md"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/134.0.0.0 Safari/537.36"
)

TOP_LEVEL_FIELDS = [
    "architecture_codename",
    "die_family",
    "compute_capability",
    "sm_count",
    "tensor_core_count",
    "rt_core_count",
    "gpc_count",
    "tpc_count",
    "sm_per_tpc",
    "tpc_per_gpc",
    "enabled_units_summary",
    "tensor_core_generation",
    "official_tensor_throughput",
]

DTYPE_KEYS = ["tf32", "fp16", "bf16", "fp8", "fp4", "int8", "int4"]


@dataclass(frozen=True)
class SourceDef:
    source_id: str
    title: str
    url: str
    kind: str  # html | pdf


@dataclass
class Evidence:
    source_id: str
    title: str
    url: str
    locator: str
    excerpt: str


@dataclass
class SourceSnapshot:
    source_id: str
    title: str
    url: str
    kind: str
    ok: bool
    fetched_at: str
    text: str = ""
    error: str | None = None


@dataclass
class RecordConfig:
    record_id: str
    record_type: str
    generation: str
    product_name: str
    notes: list[str] = field(default_factory=list)


SOURCE_DEFS = [
    SourceDef(
        "cc-current",
        "CUDA GPU Compute Capability",
        "https://developer.nvidia.com/cuda/gpus",
        "html",
    ),
    SourceDef(
        "cc-legacy",
        "Legacy CUDA GPU Compute Capability",
        "https://developer.nvidia.com/cuda-legacy-gpus",
        "html",
    ),
    SourceDef(
        "volta-whitepaper",
        "NVIDIA Tesla V100 GPU Architecture Whitepaper",
        "https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf",
        "pdf",
    ),
    SourceDef(
        "turing-whitepaper",
        "NVIDIA Turing GPU Architecture Whitepaper",
        "https://images.nvidia.com/aem-dam/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf",
        "pdf",
    ),
    SourceDef(
        "ampere-ga100-whitepaper",
        "NVIDIA A100 Tensor Core GPU Architecture Whitepaper",
        "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf",
        "pdf",
    ),
    SourceDef(
        "ampere-ga102-whitepaper",
        "NVIDIA Ampere GA102 GPU Architecture Whitepaper",
        "https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf",
        "pdf",
    ),
    SourceDef(
        "ampere-architecture-page",
        "NVIDIA Ampere Architecture",
        "https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/",
        "html",
    ),
    SourceDef(
        "ada-whitepaper",
        "NVIDIA Ada GPU Architecture Whitepaper",
        "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
        "pdf",
    ),
    SourceDef(
        "hopper-architecture-page",
        "NVIDIA Hopper Architecture",
        "https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/",
        "html",
    ),
    SourceDef(
        "blackwell-architecture-page",
        "NVIDIA Blackwell Architecture",
        "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
        "html",
    ),
    SourceDef(
        "v100-page",
        "NVIDIA Tesla V100",
        "https://www.nvidia.com/en-gb/data-center/tesla-v100/",
        "html",
    ),
    SourceDef(
        "t4-page",
        "NVIDIA T4 Tensor Core GPU",
        "https://www.nvidia.com/en-us/data-center/tesla-t4/",
        "html",
    ),
    SourceDef(
        "titan-rtx-page",
        "NVIDIA TITAN RTX",
        "https://www.nvidia.com/en-us/deep-learning-ai/products/titan-rtx.html",
        "html",
    ),
    SourceDef(
        "quadro-rtx-6000-page",
        "NVIDIA Quadro RTX 6000",
        "https://www.nvidia.com/en-eu/products/workstations/quadro/rtx-6000/",
        "html",
    ),
    SourceDef(
        "a100-page",
        "NVIDIA A100 Tensor Core GPU",
        "https://www.nvidia.com/en-in/data-center/a100/",
        "html",
    ),
    SourceDef(
        "a40-page",
        "NVIDIA A40",
        "https://www.nvidia.com/en-eu/data-center/a40/",
        "html",
    ),
    SourceDef(
        "rtx-a6000-page",
        "NVIDIA RTX A6000",
        "https://www.nvidia.com/en-eu/design-visualization/rtx-a6000/",
        "html",
    ),
    SourceDef(
        "rtx-3090-page",
        "GeForce RTX 3090 Family",
        "https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/",
        "html",
    ),
    SourceDef(
        "l4-page",
        "NVIDIA L4 Tensor Core GPU",
        "https://www.nvidia.com/en-us/data-center/l4/",
        "html",
    ),
    SourceDef(
        "l40s-page",
        "NVIDIA L40S",
        "https://www.nvidia.com/es-es/data-center/l40s/",
        "html",
    ),
    SourceDef(
        "rtx-6000-ada-page",
        "NVIDIA RTX 6000 Ada Generation",
        "https://www.nvidia.com/en-us/products/workstations/rtx-6000/",
        "html",
    ),
    SourceDef(
        "rtx-4090-page",
        "GeForce RTX 4090",
        "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
        "html",
    ),
    SourceDef(
        "h100-page",
        "NVIDIA H100 GPU",
        "https://www.nvidia.com/en-in/data-center/h100/",
        "html",
    ),
    SourceDef(
        "h200-page",
        "NVIDIA H200 Tensor Core GPU",
        "https://www.nvidia.com/es-la/data-center/h200/",
        "html",
    ),
    SourceDef(
        "dgx-b200-page",
        "NVIDIA DGX B200",
        "https://www.nvidia.com/en-us/data-center/dgx-b200/",
        "html",
    ),
    SourceDef(
        "rtx-pro-6000-blackwell-page",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/",
        "html",
    ),
    SourceDef(
        "rtx-5090-page",
        "GeForce RTX 5090",
        "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/",
        "html",
    ),
]

SOURCE_MAP = {source.source_id: source for source in SOURCE_DEFS}

RECORDS = [
    RecordConfig("arch-volta", "architecture", "Volta", "Volta"),
    RecordConfig("arch-turing", "architecture", "Turing", "Turing"),
    RecordConfig("arch-ampere", "architecture", "Ampere", "Ampere"),
    RecordConfig("arch-ada", "architecture", "Ada", "Ada"),
    RecordConfig("arch-hopper", "architecture", "Hopper", "Hopper"),
    RecordConfig("arch-blackwell", "architecture", "Blackwell", "Blackwell"),
    RecordConfig("sku-v100", "sku", "Volta", "Tesla V100"),
    RecordConfig("sku-t4", "sku", "Turing", "Tesla T4"),
    RecordConfig("sku-titan-rtx", "sku", "Turing", "TITAN RTX"),
    RecordConfig("sku-quadro-rtx-6000", "sku", "Turing", "Quadro RTX 6000"),
    RecordConfig("sku-a100", "sku", "Ampere", "A100"),
    RecordConfig("sku-a40", "sku", "Ampere", "A40"),
    RecordConfig("sku-rtx-a6000", "sku", "Ampere", "RTX A6000"),
    RecordConfig("sku-rtx-3090", "sku", "Ampere", "GeForce RTX 3090"),
    RecordConfig("sku-l4", "sku", "Ada", "L4"),
    RecordConfig("sku-l40s", "sku", "Ada", "L40S"),
    RecordConfig("sku-rtx-6000-ada", "sku", "Ada", "RTX 6000 Ada"),
    RecordConfig("sku-rtx-4090", "sku", "Ada", "GeForce RTX 4090"),
    RecordConfig("sku-h100", "sku", "Hopper", "H100"),
    RecordConfig("sku-h200", "sku", "Hopper", "H200"),
    RecordConfig("sku-b200", "sku", "Blackwell", "B200"),
    RecordConfig(
        "sku-rtx-pro-6000-blackwell",
        "sku",
        "Blackwell",
        "RTX PRO 6000 Blackwell Workstation Edition",
    ),
    RecordConfig("sku-rtx-5090", "sku", "Blackwell", "GeForce RTX 5090"),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compact_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_binary(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_binary_with_curl(url: str) -> bytes:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--compressed",
            "-A",
            USER_AGENT,
            url,
        ],
        check=True,
        capture_output=True,
        text=False,
    )
    return result.stdout


def read_pdf_text(blob: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(blob)
        handle.flush()

        try:
            reader = PdfReader(handle.name)
            pages: list[str] = []
            for index, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages.append(f"[PAGE {index}]\n{extracted}")
            text = "\n\n".join(pages).strip()
            if text:
                return compact_whitespace(text)
        except Exception:
            pass

        try:
            pdftotext = subprocess.run(
                ["pdftotext", handle.name, "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            if pdftotext.stdout.strip():
                return compact_whitespace(pdftotext.stdout)
        except Exception:
            return ""

    return ""


def html_to_text(raw_html: bytes) -> str:
    text = raw_html.decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<script\b.*?</script>", "\n", text)
    text = re.sub(r"(?is)<style\b.*?</style>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|h1|h2|h3|h4|h5|h6|li|tr|td|th|table)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return compact_whitespace(text)


def fetch_source(source: SourceDef) -> SourceSnapshot:
    fetched_at = iso_now()
    try:
        blob = fetch_binary(source.url)
    except Exception as first_error:
        try:
            blob = fetch_binary_with_curl(source.url)
        except Exception as second_error:
            return SourceSnapshot(
                source_id=source.source_id,
                title=source.title,
                url=source.url,
                kind=source.kind,
                ok=False,
                fetched_at=fetched_at,
                error=f"{first_error}; fallback failed: {second_error}",
            )

    if source.kind == "pdf":
        text = read_pdf_text(blob)
    else:
        text = html_to_text(blob)

    if not text.strip():
        return SourceSnapshot(
            source_id=source.source_id,
            title=source.title,
            url=source.url,
            kind=source.kind,
            ok=False,
            fetched_at=fetched_at,
            error="empty extracted text",
        )

    return SourceSnapshot(
        source_id=source.source_id,
        title=source.title,
        url=source.url,
        kind=source.kind,
        ok=True,
        fetched_at=fetched_at,
        text=text,
    )


def compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)


def clean_excerpt(text: str, limit: int = 360) -> str:
    text = compact_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def evidence_from_match(source: SourceSnapshot, locator: str, excerpt: str) -> Evidence:
    return Evidence(
        source_id=source.source_id,
        title=source.title,
        url=source.url,
        locator=locator,
        excerpt=clean_excerpt(excerpt),
    )


def around_match(text: str, start: int, end: int, radius: int = 180) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]


def search_pattern(
    source: SourceSnapshot,
    pattern: str,
    *,
    group: str | int = 1,
    locator: str | None = None,
    transform: Callable[[str], Any] | None = None,
) -> tuple[Any, Evidence] | tuple[None, None]:
    if not source.ok:
        return None, None
    match = compile_pattern(pattern).search(source.text)
    if not match:
        return None, None
    raw_value = match.group(group).strip()
    value = transform(raw_value) if transform else raw_value
    found_locator = locator or f"regex:{pattern}"
    excerpt = around_match(source.text, match.start(), match.end())
    return value, evidence_from_match(source, found_locator, excerpt)


def to_int(value: str) -> int:
    return int(re.sub(r"[^\d]", "", value))


def to_number(value: str) -> int | float:
    cleaned = value.strip().replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        if re.fullmatch(r"\d+,\d{3}", cleaned):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    numeric = float(cleaned)
    return int(numeric) if numeric.is_integer() else numeric


def to_cc(value: str) -> str:
    return value.strip()


def set_field(
    record: dict[str, Any],
    field_name: str,
    value: Any,
    evidence: Evidence | None,
) -> None:
    if value is None:
        return
    record[field_name] = value
    if evidence is not None:
        record["field_evidence"].setdefault(field_name, []).append(evidence.__dict__)


def set_text_via_regex(
    record: dict[str, Any],
    field_name: str,
    source: SourceSnapshot,
    pattern: str,
    *,
    group: str | int = 1,
    locator: str | None = None,
    transform: Callable[[str], Any] | None = None,
) -> None:
    value, evidence = search_pattern(
        source,
        pattern,
        group=group,
        locator=locator,
        transform=transform,
    )
    set_field(record, field_name, value, evidence)


def add_note(record: dict[str, Any], note: str) -> None:
    if note not in record["notes"]:
        record["notes"].append(note)


def set_dtype_support(
    record: dict[str, Any],
    dtype_key: str,
    source: SourceSnapshot,
    pattern: str,
    *,
    locator: str | None = None,
) -> None:
    _, evidence = search_pattern(source, pattern, group=0, locator=locator)
    if evidence is None:
        return
    record["tensor_datatypes"][dtype_key] = True
    record["field_evidence"].setdefault(f"tensor_datatypes.{dtype_key}", []).append(
        evidence.__dict__
    )


def add_throughput_entry(
    record: dict[str, Any],
    source: SourceSnapshot,
    pattern: str,
    *,
    label: str,
    dtype: str | None,
    unit: str,
    sparsity: str | None = None,
    locator: str | None = None,
) -> None:
    value, evidence = search_pattern(
        source,
        pattern,
        group=1,
        locator=locator or label,
        transform=to_number,
    )
    if value is None or evidence is None:
        return
    entry = {
        "label": label,
        "dtype": dtype,
        "value": value,
        "unit": unit,
        "sparsity": sparsity,
        "source_id": source.source_id,
    }
    record["official_tensor_throughput"].append(entry)
    record["field_evidence"].setdefault("official_tensor_throughput", []).append(evidence.__dict__)


def add_raw_throughput_excerpt(
    record: dict[str, Any],
    source: SourceSnapshot,
    pattern: str,
    *,
    label: str,
    dtype: str | None = None,
    unit: str,
    sparsity: str | None = None,
    locator: str | None = None,
) -> None:
    value, evidence = search_pattern(source, pattern, group=1, locator=locator or label)
    if value is None or evidence is None:
        return
    numeric = re.search(r"[\d,.]+", str(value))
    normalized_value = to_number(numeric.group(0)) if numeric else value
    record["official_tensor_throughput"].append(
        {
            "label": label,
            "dtype": dtype,
            "value": normalized_value,
            "unit": unit,
            "sparsity": sparsity,
            "source_id": source.source_id,
        }
    )
    record["field_evidence"].setdefault("official_tensor_throughput", []).append(evidence.__dict__)


def init_record(config: RecordConfig) -> dict[str, Any]:
    return {
        "record_id": config.record_id,
        "record_type": config.record_type,
        "generation": config.generation,
        "product_name": config.product_name,
        "architecture_codename": None,
        "die_family": None,
        "compute_capability": None,
        "sm_count": None,
        "tensor_core_count": None,
        "rt_core_count": None,
        "gpc_count": None,
        "tpc_count": None,
        "sm_per_tpc": None,
        "tpc_per_gpc": None,
        "enabled_units_summary": None,
        "tensor_core_generation": None,
        "tensor_datatypes": {key: None for key in DTYPE_KEYS},
        "official_tensor_throughput": [],
        "source_urls": [],
        "field_evidence": {},
        "supplemental_sources": [],
        "notes": list(config.notes),
        "missing_fields": [],
    }


def finalize_record(record: dict[str, Any]) -> None:
    source_urls = {
        evidence["url"]
        for evidence_list in record["field_evidence"].values()
        for evidence in evidence_list
    }
    record["source_urls"] = sorted(source_urls)
    if not record["official_tensor_throughput"]:
        record["official_tensor_throughput"] = None

    missing = []
    for field_name in TOP_LEVEL_FIELDS:
        value = record.get(field_name)
        if value is None:
            missing.append(field_name)
        elif field_name == "official_tensor_throughput" and not value:
            missing.append(field_name)
    if all(record["tensor_datatypes"][key] is None for key in DTYPE_KEYS):
        missing.append("tensor_datatypes")
    record["missing_fields"] = missing


def cc_by_product(source: SourceSnapshot, product_name: str) -> tuple[str | None, Evidence | None]:
    if not source.ok:
        return None, None
    lines = [line.strip() for line in source.text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == product_name:
            cursor = index - 1
            while cursor >= 0:
                if re.fullmatch(r"\d+\.\d+", lines[cursor]):
                    excerpt = "\n".join(lines[max(0, cursor - 1) : min(len(lines), index + 2)])
                    return lines[cursor], evidence_from_match(
                        source,
                        f"listing:{product_name}",
                        excerpt,
                    )
                cursor -= 1
    return None, None


def fill_architecture_records(record: dict[str, Any], sources: dict[str, SourceSnapshot]) -> None:
    name = record["generation"]
    if name == "Volta":
        set_field(record, "architecture_codename", "Volta", None)
        set_text_via_regex(
            record,
            "die_family",
            sources["volta-whitepaper"],
            r"GV100 CUDA Hardware",
            group=0,
            locator="GV100 whitepaper heading",
            transform=lambda _value: "GV100",
        )
        cc, evidence = cc_by_product(sources["cc-legacy"], "NVIDIA V100")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(
            record,
            "sm_count",
            sources["volta-whitepaper"],
            r"(\d+)\s+Streaming Multiprocessors",
            locator="Volta whitepaper SM count",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "tensor_core_count",
            sources["volta-whitepaper"],
            r"(\d+)\s+Tensor Cores",
            locator="Volta whitepaper Tensor Core count",
            transform=to_int,
        )
        set_dtype_support(record, "fp16", sources["volta-whitepaper"], r"FP16 multiply results")
        add_throughput_entry(
            record,
            sources["v100-page"],
            r"Deep Learning\s+(\d+(?:\.\d+)?)\s*teraFLOPS",
            label="Deep Learning",
            dtype="fp16",
            unit="TFLOPS",
            locator="Tesla V100 specs",
        )
        add_note(record, "Architecture summary is anchored to the flagship GV100/V100 whitepaper.")
        return

    if name == "Turing":
        set_field(record, "architecture_codename", "Turing", None)
        set_text_via_regex(
            record,
            "die_family",
            sources["turing-whitepaper"],
            r"codenamed\s+(TU102)",
            locator="Turing whitepaper TU102 mention",
        )
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA TITAN RTX")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(
            record,
            "gpc_count",
            sources["titan-rtx-page"],
            r"(\d+)\s+Graphics Processing Clusters",
            locator="TITAN RTX full specs",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "tpc_count",
            sources["titan-rtx-page"],
            r"(\d+)\s+Texture Processing Clusters",
            locator="TITAN RTX full specs",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "sm_count",
            sources["titan-rtx-page"],
            r"(\d+)\s+Streaming Multiprocessors",
            locator="TITAN RTX full specs",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "tensor_core_count",
            sources["titan-rtx-page"],
            r"(\d+)\s+Tensor Cores",
            locator="TITAN RTX full specs",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "rt_core_count",
            sources["titan-rtx-page"],
            r"(\d+)\s+RT Cores",
            locator="TITAN RTX full specs",
            transform=to_int,
        )
        set_dtype_support(record, "fp16", sources["turing-whitepaper"], r"FP16")
        set_dtype_support(record, "int8", sources["turing-whitepaper"], r"INT8")
        set_dtype_support(record, "int4", sources["turing-whitepaper"], r"INT4")
        add_raw_throughput_excerpt(
            record,
            sources["titan-rtx-page"],
            r"bringing\s+(\d+)\s+Tensor TFLOPs of performance",
            label="Tensor TFLOPS",
            dtype=None,
            unit="TFLOPS",
            locator="TITAN RTX hero copy",
        )
        add_note(record, "Topology fields are sourced from TITAN RTX because NVIDIA publishes TU102 block activation there.")
        return

    if name == "Ampere":
        set_field(record, "architecture_codename", "Ampere", None)
        set_text_via_regex(
            record,
            "die_family",
            sources["ampere-ga100-whitepaper"],
            r"A100 Tensor Core GPU implementation of the\s+(GA100)\s+GPU",
            locator="A100 whitepaper GA100 implementation",
        )
        set_field(record, "compute_capability", "8.0 / 8.6", None)
        set_text_via_regex(
            record,
            "gpc_count",
            sources["ampere-ga100-whitepaper"],
            r"(\d+)\s+GPCs,\s+8\s+TPCs/GPC,\s+2\s+SMs/TPC,\s+16\s+SMs/GPC,\s+128\s+SMs per full GPU",
            locator="GA100 full GPU topology",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "sm_count",
            sources["ampere-ga100-whitepaper"],
            r"128\s+SMs per full GPU",
            group=0,
            locator="GA100 full GPU topology",
            transform=lambda _value: 128,
        )
        set_text_via_regex(
            record,
            "tensor_core_count",
            sources["ampere-ga100-whitepaper"],
            r"(\d+)\s+Third-generation Tensor Cores per full GPU",
            locator="GA100 full GPU Tensor Core count",
            transform=to_int,
        )
        set_field(record, "sm_per_tpc", 2, None)
        set_field(record, "tpc_per_gpc", 8, None)
        set_field(record, "tensor_core_generation", "3rd generation", None)
        set_dtype_support(record, "tf32", sources["ampere-architecture-page"], r"TF32")
        set_dtype_support(record, "fp16", sources["ampere-architecture-page"], r"FP16")
        set_dtype_support(record, "bf16", sources["a100-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "int8", sources["a100-page"], r"INT8 Tensor Core")
        add_throughput_entry(
            record,
            sources["a100-page"],
            r"Tensor Float 32 \(TF32\)\s+\|\s+(\d+(?:,\d+)?)\s+TFLOPS",
            label="TF32 Tensor Core",
            dtype="tf32",
            unit="TFLOPS",
            locator="A100 specs",
        )
        add_throughput_entry(
            record,
            sources["a100-page"],
            r"BFLOAT16 Tensor Core\s+\|\s+(\d+(?:,\d+)?)\s+TFLOPS",
            label="BFLOAT16 Tensor Core",
            dtype="bf16",
            unit="TFLOPS",
            locator="A100 specs",
        )
        add_throughput_entry(
            record,
            sources["a100-page"],
            r"FP16 Tensor Core\s+\|\s+(\d+(?:,\d+)?)\s+TFLOPS",
            label="FP16 Tensor Core",
            dtype="fp16",
            unit="TFLOPS",
            locator="A100 specs",
        )
        add_throughput_entry(
            record,
            sources["a100-page"],
            r"INT8 Tensor Core\s+\|\s+(\d+(?:,\d+)?)\s+TOPS",
            label="INT8 Tensor Core",
            dtype="int8",
            unit="TOPS",
            locator="A100 specs",
        )
        add_note(record, "Architecture summary uses GA100/A100 because it is the official AI-first Ampere whitepaper.")
        return

    if name == "Ada":
        set_field(record, "architecture_codename", "Ada Lovelace", None)
        set_text_via_regex(
            record,
            "die_family",
            sources["ada-whitepaper"],
            r"At the heart of the GeForce RTX 4090 is the\s+(AD102)\s+GPU",
            locator="Ada whitepaper AD102 mention",
        )
        set_field(record, "compute_capability", "8.9", None)
        set_text_via_regex(
            record,
            "gpc_count",
            sources["ada-whitepaper"],
            r"Ada GPC with Raster Engine,\s+(\d+)\s+TPCs,\s+12\s+SMs",
            locator="Ada GPC block diagram",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "sm_count",
            sources["ada-whitepaper"],
            r"144 RT Cores\s+●\s+(\d+)\s+Tensor Cores",
            locator="Ada full-chip bullet list",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "rt_core_count",
            sources["ada-whitepaper"],
            r"(\d+)\s+RT Cores",
            locator="Ada full-chip bullet list",
            transform=to_int,
        )
        set_field(record, "sm_per_tpc", 2, None)
        set_field(record, "tpc_per_gpc", 6, None)
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "fp8", sources["l4-page"], r"FP8 Tensor Core")
        set_dtype_support(record, "tf32", sources["l4-page"], r"TF32 Tensor Core")
        set_dtype_support(record, "fp16", sources["l4-page"], r"FP16 Tensor Core")
        set_dtype_support(record, "bf16", sources["l4-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "int8", sources["l40s-page"], r"INT8")
        set_dtype_support(record, "int4", sources["l40s-page"], r"INT4")
        add_raw_throughput_excerpt(
            record,
            sources["rtx-6000-ada-page"],
            r"Tensor Performance\s+(\d+(?:,\d+)?)\s+AI TOPS",
            label="Tensor Performance",
            dtype="fp8",
            unit="TOPS",
            sparsity="sparse",
            locator="RTX 6000 Ada highlights",
        )
        add_note(record, "Topology/count summary is anchored to AD102 in the Ada whitepaper.")
        return

    if name == "Hopper":
        set_field(record, "architecture_codename", "Hopper", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA H100")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "tf32", sources["hopper-architecture-page"], r"TF32")
        set_dtype_support(record, "fp16", sources["hopper-architecture-page"], r"FP16")
        set_dtype_support(record, "bf16", sources["h100-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "fp8", sources["hopper-architecture-page"], r"FP8")
        set_dtype_support(record, "int8", sources["hopper-architecture-page"], r"INT8")
        add_throughput_entry(
            record,
            sources["h100-page"],
            r"TF32 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS",
            label="TF32 Tensor Core",
            dtype="tf32",
            unit="TFLOPS",
            sparsity="sparse",
            locator="H100 product specs",
        )
        add_throughput_entry(
            record,
            sources["h100-page"],
            r"BFLOAT16 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS",
            label="BFLOAT16 Tensor Core",
            dtype="bf16",
            unit="TFLOPS",
            sparsity="sparse",
            locator="H100 product specs",
        )
        add_throughput_entry(
            record,
            sources["h100-page"],
            r"FP16 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS",
            label="FP16 Tensor Core",
            dtype="fp16",
            unit="TFLOPS",
            sparsity="sparse",
            locator="H100 product specs",
        )
        add_throughput_entry(
            record,
            sources["h100-page"],
            r"FP8 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS",
            label="FP8 Tensor Core",
            dtype="fp8",
            unit="TFLOPS",
            sparsity="sparse",
            locator="H100 product specs",
        )
        add_throughput_entry(
            record,
            sources["h100-page"],
            r"INT8 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+TOPS",
            label="INT8 Tensor Core",
            dtype="int8",
            unit="TOPS",
            sparsity="sparse",
            locator="H100 product specs",
        )
        add_note(record, "Official Hopper architecture page exposes capabilities more reliably than topology.")
        return

    if name == "Blackwell":
        set_field(record, "architecture_codename", "Blackwell", None)
        set_text_via_regex(
            record,
            "enabled_units_summary",
            sources["blackwell-architecture-page"],
            r"feature two reticle-limited dies connected by a 10 terabytes per second \(TB/s\) chip-to-chip interconnect",
            group=0,
            locator="Blackwell architecture introduction",
            transform=lambda _value: "Two reticle-limited dies with 10 TB/s chip-to-chip interconnect",
        )
        set_field(record, "tensor_core_generation", "5th generation", None)
        set_dtype_support(record, "fp4", sources["blackwell-architecture-page"], r"4-bit floating point \(FP4\)")
        set_dtype_support(record, "fp8", sources["blackwell-architecture-page"], r"larger precisions")
        add_note(record, "Blackwell public architecture pages emphasize transformer-engine features more than unit topology.")
        return


def fill_sku_records(record: dict[str, Any], sources: dict[str, SourceSnapshot]) -> None:
    name = record["product_name"]

    if name == "Tesla V100":
        set_field(record, "architecture_codename", "Volta", None)
        set_text_via_regex(record, "die_family", sources["volta-whitepaper"], r"GV100", group=0, transform=lambda _v: "GV100")
        cc, evidence = cc_by_product(sources["cc-legacy"], "NVIDIA V100")
        set_field(record, "compute_capability", cc, evidence)
        set_dtype_support(record, "fp16", sources["volta-whitepaper"], r"FP16")
        add_throughput_entry(
            record,
            sources["v100-page"],
            r"Deep Learning\s+(\d+(?:\.\d+)?)\s*teraFLOPS",
            label="Deep Learning",
            dtype="fp16",
            unit="TFLOPS",
            locator="Tesla V100 specifications",
        )
        return

    if name == "Tesla T4":
        set_field(record, "architecture_codename", "Turing", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA T4")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(
            record,
            "tensor_core_count",
            sources["t4-page"],
            r"Turing Tensor Cores\s+(\d+)",
            locator="T4 specifications",
            transform=to_int,
        )
        set_text_via_regex(
            record,
            "enabled_units_summary",
            sources["t4-page"],
            r"NVIDIA CUDA(?:®)? cores?\s+2,560",
            group=0,
            locator="T4 specifications",
            transform=lambda _value: "2,560 CUDA cores",
        )
        set_dtype_support(record, "fp16", sources["t4-page"], r"FP16")
        set_dtype_support(record, "int8", sources["t4-page"], r"INT8")
        set_dtype_support(record, "int4", sources["t4-page"], r"INT4")
        add_throughput_entry(
            record,
            sources["t4-page"],
            r"Mixed Precision \(FP16/FP32\)\s+(\d+)\s+FP16 TFLOPS",
            label="Mixed Precision",
            dtype="fp16",
            unit="TFLOPS",
            locator="T4 specifications",
        )
        add_throughput_entry(
            record,
            sources["t4-page"],
            r"INT8 Precision\s+(\d+)\s+INT8 TOPS",
            label="INT8 Precision",
            dtype="int8",
            unit="TOPS",
            locator="T4 specifications",
        )
        add_throughput_entry(
            record,
            sources["t4-page"],
            r"INT4 Precision\s+(\d+)\s+INT4 TOPS",
            label="INT4 Precision",
            dtype="int4",
            unit="TOPS",
            locator="T4 specifications",
        )
        return

    if name == "TITAN RTX":
        set_field(record, "architecture_codename", "Turing", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA TITAN RTX")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "gpc_count", sources["titan-rtx-page"], r"(\d+)\s+Graphics Processing Clusters", locator="TITAN RTX specs", transform=to_int)
        set_text_via_regex(record, "tpc_count", sources["titan-rtx-page"], r"(\d+)\s+Texture Processing Clusters", locator="TITAN RTX specs", transform=to_int)
        set_text_via_regex(record, "sm_count", sources["titan-rtx-page"], r"(\d+)\s+Streaming Multiprocessors", locator="TITAN RTX specs", transform=to_int)
        set_text_via_regex(record, "tensor_core_count", sources["titan-rtx-page"], r"(\d+)\s+Tensor Cores", locator="TITAN RTX specs", transform=to_int)
        set_text_via_regex(record, "rt_core_count", sources["titan-rtx-page"], r"(\d+)\s+RT Cores", locator="TITAN RTX specs", transform=to_int)
        set_field(record, "enabled_units_summary", "6 GPC / 36 TPC / 72 SM", None)
        set_dtype_support(record, "fp16", sources["titan-rtx-page"], r"FP16")
        set_dtype_support(record, "int8", sources["titan-rtx-page"], r"INT8")
        set_dtype_support(record, "int4", sources["titan-rtx-page"], r"INT4")
        add_raw_throughput_excerpt(
            record,
            sources["titan-rtx-page"],
            r"bringing\s+(\d+)\s+Tensor TFLOPs of performance",
            label="Tensor TFLOPS",
            dtype=None,
            unit="TFLOPS",
            locator="TITAN RTX overview",
        )
        return

    if name == "Quadro RTX 6000":
        set_field(record, "architecture_codename", "Turing", None)
        cc, evidence = cc_by_product(sources["cc-current"], "QUADRO RTX 6000")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "tensor_core_count", sources["quadro-rtx-6000-page"], r"NVIDIA Tensor Cores\s+\|\s+(\d+(?:,\d+)?)", locator="Quadro RTX 6000 specs", transform=to_int)
        set_text_via_regex(record, "rt_core_count", sources["quadro-rtx-6000-page"], r"NVIDIA RT Cores\s+\|\s+(\d+(?:,\d+)?)", locator="Quadro RTX 6000 specs", transform=to_int)
        set_text_via_regex(record, "enabled_units_summary", sources["quadro-rtx-6000-page"], r"CUDA Parallel-Processing Cores\s+\|\s+4,608", group=0, locator="Quadro RTX 6000 specs", transform=lambda _v: "4,608 CUDA cores")
        set_dtype_support(record, "fp16", sources["quadro-rtx-6000-page"], r"Tensor Cores")
        return

    if name == "A100":
        set_field(record, "architecture_codename", "Ampere", None)
        set_text_via_regex(record, "die_family", sources["ampere-ga100-whitepaper"], r"A100 Tensor Core GPU implementation of the\s+(GA100)\s+GPU", locator="A100 whitepaper", transform=str)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA A100")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "gpc_count", sources["ampere-ga100-whitepaper"], r"●\s+7 GPCs,\s+7 or 8 TPCs/GPC,\s+2 SMs/TPC,\s+up to 16 SMs/GPC,\s+108 SMs", group=0, locator="A100 implementation topology", transform=lambda _v: 7)
        set_text_via_regex(record, "sm_count", sources["ampere-ga100-whitepaper"], r"108 SMs", group=0, locator="A100 implementation topology", transform=lambda _v: 108)
        set_text_via_regex(record, "sm_per_tpc", sources["ampere-ga100-whitepaper"], r"2 SMs/TPC", group=0, locator="A100 implementation topology", transform=lambda _v: 2)
        set_text_via_regex(record, "tensor_core_count", sources["ampere-ga100-whitepaper"], r"432 Third-generation Tensor Cores", group=0, locator="A100 implementation Tensor Core count", transform=lambda _v: 432)
        set_text_via_regex(record, "enabled_units_summary", sources["ampere-ga100-whitepaper"], r"7 GPCs,\s+7 or 8 TPCs/GPC,\s+2 SMs/TPC,\s+up to 16 SMs/GPC,\s+108 SMs", group=0, locator="A100 implementation topology", transform=lambda _v: "7 GPCs; 7-8 TPCs/GPC; 108 SM enabled")
        set_field(record, "tensor_core_generation", "3rd generation", None)
        set_dtype_support(record, "tf32", sources["a100-page"], r"Tensor Float 32 \(TF32\)")
        set_dtype_support(record, "fp16", sources["a100-page"], r"FP16 Tensor Core")
        set_dtype_support(record, "bf16", sources["a100-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "int8", sources["a100-page"], r"INT8 Tensor Core")
        add_throughput_entry(record, sources["a100-page"], r"Tensor Float 32 \(TF32\)\s+\|\s+(\d+(?:,\d+)?)\s+TFLOPS", label="TF32 Tensor Core", dtype="tf32", unit="TFLOPS", locator="A100 specs")
        add_throughput_entry(record, sources["a100-page"], r"BFLOAT16 Tensor Core\s+\|\s+(\d+(?:,\d+)?)\s+TFLOPS", label="BFLOAT16 Tensor Core", dtype="bf16", unit="TFLOPS", locator="A100 specs")
        add_throughput_entry(record, sources["a100-page"], r"FP16 Tensor Core\s+\|\s+(\d+(?:,\d+)?)\s+TFLOPS", label="FP16 Tensor Core", dtype="fp16", unit="TFLOPS", locator="A100 specs")
        add_throughput_entry(record, sources["a100-page"], r"INT8 Tensor Core\s+\|\s+(\d+(?:,\d+)?)\s+TOPS", label="INT8 Tensor Core", dtype="int8", unit="TOPS", locator="A100 specs")
        return

    if name == "A40":
        set_field(record, "architecture_codename", "Ampere", None)
        set_text_via_regex(record, "die_family", sources["ampere-ga102-whitepaper"], r"GA102 is the most powerful Ampere architecture GPU.*?used in the.*?NVIDIA A40", group=0, locator="GA102 whitepaper A40 mention", transform=lambda _v: "GA102")
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA A40")
        set_field(record, "compute_capability", cc, evidence)
        set_dtype_support(record, "tf32", sources["a40-page"], r"Tensor Float 32 \(TF32\)")
        return

    if name == "RTX A6000":
        set_field(record, "architecture_codename", "Ampere", None)
        set_text_via_regex(record, "die_family", sources["ampere-ga102-whitepaper"], r"used in the GeForce RTX 3090, GeForce RTX 3080, NVIDIA RTX A6000, and the NVIDIA A40", group=0, locator="GA102 whitepaper supported products", transform=lambda _v: "GA102")
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA RTX A6000")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "3rd generation", None)
        set_dtype_support(record, "tf32", sources["rtx-a6000-page"], r"Tensor Float 32 \(TF32\)")
        set_dtype_support(record, "fp16", sources["rtx-a6000-page"], r"Third-Generation Tensor Cores")
        return

    if name == "GeForce RTX 3090":
        set_field(record, "architecture_codename", "Ampere", None)
        set_text_via_regex(record, "die_family", sources["ampere-ga102-whitepaper"], r"used in the GeForce RTX 3090, GeForce RTX 3080, NVIDIA RTX A6000, and the NVIDIA A40", group=0, locator="GA102 whitepaper supported products", transform=lambda _v: "GA102")
        cc, evidence = cc_by_product(sources["cc-current"], "GeForce RTX 3090")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "enabled_units_summary", sources["rtx-3090-page"], r"NVIDIA CUDA(?:®)? Cores\s+10752", group=0, locator="RTX 3090 family specs", transform=lambda _v: "10,752 CUDA cores")
        set_field(record, "tensor_core_generation", "3rd generation", None)
        set_field(record, "rt_core_count", None, None)
        set_dtype_support(record, "fp16", sources["rtx-3090-page"], r"3rd Generation")
        return

    if name == "L4":
        set_field(record, "architecture_codename", "Ada Lovelace", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA L4")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "tf32", sources["l4-page"], r"TF32 Tensor Core")
        set_dtype_support(record, "fp16", sources["l4-page"], r"FP16 Tensor Core")
        set_dtype_support(record, "bf16", sources["l4-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "fp8", sources["l4-page"], r"FP8 Tensor Core")
        add_throughput_entry(record, sources["l4-page"], r"TF32 Tensor Core\s+([\d.,]+)\s+teraFLOPS", label="TF32 Tensor Core", dtype="tf32", unit="TFLOPS", sparsity="sparse", locator="L4 product specs")
        add_throughput_entry(record, sources["l4-page"], r"FP16 Tensor Core\s+([\d.,]+)\s+teraFLOPS", label="FP16 Tensor Core", dtype="fp16", unit="TFLOPS", sparsity="sparse", locator="L4 product specs")
        add_throughput_entry(record, sources["l4-page"], r"BFLOAT16 Tensor Core\s+([\d.,]+)\s+teraFLOPS", label="BFLOAT16 Tensor Core", dtype="bf16", unit="TFLOPS", sparsity="sparse", locator="L4 product specs")
        add_throughput_entry(record, sources["l4-page"], r"FP8 Tensor Core\s+([\d.,]+)\s+teraFLOPs", label="FP8 Tensor Core", dtype="fp8", unit="TFLOPS", sparsity="sparse", locator="L4 product specs")
        add_throughput_entry(record, sources["l4-page"], r"INT8 Tensor Core\s+([\d.,]+)\s+TOPs", label="INT8 Tensor Core", dtype="int8", unit="TOPS", sparsity="sparse", locator="L4 product specs")
        return

    if name == "L40S":
        set_field(record, "architecture_codename", "Ada Lovelace", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA L40S")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "tensor_core_count", sources["l40s-page"], r"Tensor Cores de cuarta generación de NVIDIA\s+\|\s+(\d+)", locator="L40S specs", transform=to_int)
        set_text_via_regex(record, "rt_core_count", sources["l40s-page"], r"Núcleos RT de tercera generación de NVIDIA\s+\|\s+(\d+)", locator="L40S specs", transform=to_int)
        set_text_via_regex(record, "enabled_units_summary", sources["l40s-page"], r"Núcleos CUDA\® basados en la arquitectura NVIDIA Ada Lovelace\s+\|\s+18 176", group=0, locator="L40S specs", transform=lambda _v: "18,176 CUDA cores")
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "tf32", sources["l40s-page"], r"Tensor Core de TF32")
        set_dtype_support(record, "fp16", sources["l40s-page"], r"Tensor Core de FP16")
        set_dtype_support(record, "bf16", sources["l40s-page"], r"BFLOAT16")
        set_dtype_support(record, "fp8", sources["l40s-page"], r"Tensor Core de FP8")
        set_dtype_support(record, "int8", sources["l40s-page"], r"TOPS de Tensor Core de INT8")
        set_dtype_support(record, "int4", sources["l40s-page"], r"Tensor Core de INT4")
        add_raw_throughput_excerpt(record, sources["l40s-page"], r"Tensor Core de TF32\s+\|\s+([\d.,]+)", label="TF32 Tensor Core", dtype="tf32", unit="TFLOPS", locator="L40S specs")
        add_raw_throughput_excerpt(record, sources["l40s-page"], r"Tensor Core de FP16\s+\|\s+([\d.,]+)", label="FP16 Tensor Core", dtype="fp16", unit="TFLOPS", locator="L40S specs")
        add_raw_throughput_excerpt(record, sources["l40s-page"], r"TFLOPS de Tensor Core de BFLOAT16\s+\|\s+([\d.,]+)", label="BFLOAT16 Tensor Core", dtype="bf16", unit="TFLOPS", locator="L40S specs")
        add_raw_throughput_excerpt(record, sources["l40s-page"], r"Tensor Core de FP8\s+\|\s+([\d.,]+)", label="FP8 Tensor Core", dtype="fp8", unit="TFLOPS", locator="L40S specs")
        return

    if name == "RTX 6000 Ada":
        set_field(record, "architecture_codename", "Ada Lovelace", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA RTX 6000 Ada")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "fp8", sources["rtx-6000-ada-page"], r"FP8 data format")
        add_raw_throughput_excerpt(record, sources["rtx-6000-ada-page"], r"Tensor Performance\s+(\d+(?:,\d+)?)\s+AI TOPS", label="Tensor Performance", dtype="fp8", unit="TOPS", sparsity="sparse", locator="RTX 6000 Ada highlights")
        return

    if name == "GeForce RTX 4090":
        set_field(record, "architecture_codename", "Ada Lovelace", None)
        set_text_via_regex(record, "die_family", sources["ada-whitepaper"], r"At the heart of the GeForce RTX 4090 is the\s+(AD102)\s+GPU", locator="Ada whitepaper", transform=str)
        cc, evidence = cc_by_product(sources["cc-current"], "GeForce RTX 4090")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "enabled_units_summary", sources["rtx-4090-page"], r"NVIDIA CUDA(?:®)? Cores\s+16384", group=0, locator="RTX 4090 specs", transform=lambda _v: "16,384 CUDA cores")
        set_field(record, "tensor_core_generation", "4th generation", None)
        add_raw_throughput_excerpt(record, sources["rtx-4090-page"], r"Tensor Cores \(AI\)\s+4th Generation\s+([\d.,]+)\s+AI TOPS", label="AI TOPS", dtype=None, unit="TOPS", locator="RTX 4090 specs")
        return

    if name == "H100":
        set_field(record, "architecture_codename", "Hopper", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA H100")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "tf32", sources["h100-page"], r"TF32 Tensor Core")
        set_dtype_support(record, "fp16", sources["h100-page"], r"FP16 Tensor Core")
        set_dtype_support(record, "bf16", sources["h100-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "fp8", sources["h100-page"], r"FP8 Tensor Core")
        set_dtype_support(record, "int8", sources["h100-page"], r"INT8 Tensor Core")
        add_throughput_entry(record, sources["h100-page"], r"TF32 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS", label="TF32 Tensor Core", dtype="tf32", unit="TFLOPS", sparsity="sparse", locator="H100 product specs")
        add_throughput_entry(record, sources["h100-page"], r"BFLOAT16 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS", label="BFLOAT16 Tensor Core", dtype="bf16", unit="TFLOPS", sparsity="sparse", locator="H100 product specs")
        add_throughput_entry(record, sources["h100-page"], r"FP16 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS", label="FP16 Tensor Core", dtype="fp16", unit="TFLOPS", sparsity="sparse", locator="H100 product specs")
        add_throughput_entry(record, sources["h100-page"], r"FP8 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+teraFLOPS", label="FP8 Tensor Core", dtype="fp8", unit="TFLOPS", sparsity="sparse", locator="H100 product specs")
        add_throughput_entry(record, sources["h100-page"], r"INT8 Tensor Core\^\{\*\}\s+\|\s+(\d+(?:,\d+)?)\s+TOPS", label="INT8 Tensor Core", dtype="int8", unit="TOPS", sparsity="sparse", locator="H100 product specs")
        return

    if name == "H200":
        set_field(record, "architecture_codename", "Hopper", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA H200")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "4th generation", None)
        set_dtype_support(record, "tf32", sources["h200-page"], r"TF32 Tensor Core")
        set_dtype_support(record, "fp16", sources["h200-page"], r"FP16 Tensor Core")
        set_dtype_support(record, "bf16", sources["h200-page"], r"BFLOAT16 Tensor Core")
        set_dtype_support(record, "fp8", sources["h200-page"], r"FP8 Tensor Core")
        set_dtype_support(record, "int8", sources["h200-page"], r"INT8 Tensor Core")
        add_raw_throughput_excerpt(record, sources["h200-page"], r"TF32 Tensor Core²\s+\|\s+([\d,.]+)\s+TFLOPS", label="TF32 Tensor Core", dtype="tf32", unit="TFLOPS", locator="H200 specs")
        add_raw_throughput_excerpt(record, sources["h200-page"], r"BFLOAT16 Tensor Core²\s+\|\s+([\d,.]+)\s+TFLOPS", label="BFLOAT16 Tensor Core", dtype="bf16", unit="TFLOPS", locator="H200 specs")
        add_raw_throughput_excerpt(record, sources["h200-page"], r"FP16 Tensor Core²\s+\|\s+([\d,.]+)\s+TFLOPS", label="FP16 Tensor Core", dtype="fp16", unit="TFLOPS", locator="H200 specs")
        add_raw_throughput_excerpt(record, sources["h200-page"], r"FP8 Tensor Core²\s+\|\s+([\d,.]+)\s+TFLOPS", label="FP8 Tensor Core", dtype="fp8", unit="TFLOPS", locator="H200 specs")
        add_raw_throughput_excerpt(record, sources["h200-page"], r"INT8 Tensor Core²\s+\|\s+([\d,.]+)\s+TFLOPS", label="INT8 Tensor Core", dtype="int8", unit="TFLOPS", locator="H200 specs")
        return

    if name == "B200":
        set_field(record, "architecture_codename", "Blackwell", None)
        cc, evidence = cc_by_product(sources["cc-current"], "NVIDIA B200")
        set_field(record, "compute_capability", cc, evidence)
        set_text_via_regex(record, "enabled_units_summary", sources["dgx-b200-page"], r"GPU\s+\|\s+8x NVIDIA Blackwell GPUs", group=0, locator="DGX B200 system specs", transform=lambda _v: "Public page exposes B200 primarily through 8-GPU DGX B200 system specs")
        add_note(record, "NVIDIA public pages sampled here expose B200 primarily through DGX/HGX system-level specs; per-GPU unit counts stay null unless explicitly published.")
        return

    if name == "RTX PRO 6000 Blackwell Workstation Edition":
        set_field(record, "architecture_codename", "Blackwell", None)
        cc, evidence = cc_by_product(
            sources["cc-current"],
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        )
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "5th generation", None)
        set_dtype_support(record, "fp4", sources["rtx-pro-6000-blackwell-page"], r"Theoretical FP4 TOPS")
        add_raw_throughput_excerpt(
            record,
            sources["rtx-pro-6000-blackwell-page"],
            r"AI Performance\s+(\d+(?:,\d+)?)\s+TOPS",
            label="AI Performance",
            dtype="fp4",
            unit="TOPS",
            sparsity="sparse",
            locator="RTX PRO 6000 Blackwell highlights",
        )
        return

    if name == "GeForce RTX 5090":
        set_field(record, "architecture_codename", "Blackwell", None)
        cc, evidence = cc_by_product(sources["cc-current"], "GeForce RTX 5090")
        set_field(record, "compute_capability", cc, evidence)
        set_field(record, "tensor_core_generation", "5th generation", None)
        set_text_via_regex(record, "enabled_units_summary", sources["rtx-5090-page"], r"NVIDIA CUDA(?:®)? Cores\s+21760", group=0, locator="RTX 5090 specs", transform=lambda _v: "21,760 CUDA cores")
        set_dtype_support(record, "fp4", sources["rtx-5090-page"], r"Max AI performance with FP4")
        add_raw_throughput_excerpt(record, sources["rtx-5090-page"], r"Tensor Cores \(AI\)\s+5th Generation\s+([\d.,]+)\s+AI TOPS", label="AI TOPS", dtype=None, unit="TOPS", locator="RTX 5090 specs")
        return


def extract_record(config: RecordConfig, sources: dict[str, SourceSnapshot]) -> dict[str, Any]:
    record = init_record(config)
    if config.record_type == "architecture":
        fill_architecture_records(record, sources)
    else:
        fill_sku_records(record, sources)
    finalize_record(record)
    return record


def build_sources_report(sources: dict[str, SourceSnapshot], records: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for evidence_list in record["field_evidence"].values():
            for evidence in evidence_list:
                if record["record_id"] not in usage[evidence["source_id"]]:
                    usage[evidence["source_id"]].append(record["record_id"])

    return {
        "generatedAt": iso_now(),
        "sources": [
            {
                "sourceId": source.source_id,
                "title": source.title,
                "url": source.url,
                "kind": source.kind,
                "ok": source.ok,
                "fetchedAt": source.fetched_at,
                "error": source.error,
                "usedByRecords": sorted(usage.get(source.source_id, [])),
            }
            for source in sources.values()
        ],
    }


def nested_missing_fields(record: dict[str, Any]) -> list[str]:
    missing = list(record["missing_fields"])
    for key, value in record["tensor_datatypes"].items():
        if value is None:
            missing.append(f"tensor_datatypes.{key}")
    return missing


def build_missing_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generatedAt": iso_now(),
        "records": [
            {
                "recordId": record["record_id"],
                "recordType": record["record_type"],
                "generation": record["generation"],
                "productName": record["product_name"],
                "missingFields": nested_missing_fields(record),
            }
            for record in records
        ],
    }


def build_coverage_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = len(records)
    field_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        for field_name in TOP_LEVEL_FIELDS:
            covered = record.get(field_name) not in (None, [], {})
            field_counter[field_name]["covered" if covered else "missing"] += 1
        for dtype_key in DTYPE_KEYS:
            covered = record["tensor_datatypes"][dtype_key] is True
            field_counter[f"tensor_datatypes.{dtype_key}"]["covered" if covered else "missing"] += 1

    coverage_by_field = {}
    for field_name, counts in sorted(field_counter.items()):
        covered = counts["covered"]
        coverage_by_field[field_name] = {
            "covered": covered,
            "total": totals,
            "ratio": round(covered / totals, 4) if totals else 0,
        }

    return {
        "generatedAt": iso_now(),
        "totalRecords": totals,
        "coverageByField": coverage_by_field,
        "summary": {
            "recordsWithThroughput": sum(1 for record in records if record["official_tensor_throughput"]),
            "recordsUsingSupplementalSources": sum(1 for record in records if record["supplemental_sources"]),
            "architectureRecords": sum(1 for record in records if record["record_type"] == "architecture"),
            "skuRecords": sum(1 for record in records if record["record_type"] == "sku"),
        },
    }


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    throughput = record["official_tensor_throughput"] or []
    throughput_text = "; ".join(
        f"{entry['label']}={entry['value']} {entry['unit']}" for entry in throughput
    )
    return {
        "record_id": record["record_id"],
        "record_type": record["record_type"],
        "generation": record["generation"],
        "product_name": record["product_name"],
        "architecture_codename": record["architecture_codename"],
        "die_family": record["die_family"],
        "compute_capability": record["compute_capability"],
        "sm_count": record["sm_count"],
        "tensor_core_count": record["tensor_core_count"],
        "rt_core_count": record["rt_core_count"],
        "gpc_count": record["gpc_count"],
        "tpc_count": record["tpc_count"],
        "sm_per_tpc": record["sm_per_tpc"],
        "tpc_per_gpc": record["tpc_per_gpc"],
        "enabled_units_summary": record["enabled_units_summary"],
        "tensor_core_generation": record["tensor_core_generation"],
        "tensor_datatypes": json.dumps(record["tensor_datatypes"], ensure_ascii=False),
        "official_tensor_throughput": throughput_text or None,
        "source_urls": " | ".join(record["source_urls"]),
        "missing_fields": " | ".join(record["missing_fields"]),
    }


def write_csv(records: list[dict[str, Any]]) -> None:
    rows = [csv_row(record) for record in records]
    fieldnames = list(rows[0].keys()) if rows else []
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def render_report(records: list[dict[str, Any]], coverage: dict[str, Any], sources: dict[str, Any]) -> str:
    ok_sources = sum(1 for item in sources["sources"] if item["ok"])
    total_sources = len(sources["sources"])
    missing_entries = [
        (record["product_name"], len(nested_missing_fields(record)))
        for record in records
    ]
    missing_entries.sort(key=lambda item: (-item[1], item[0]))
    missing_preview = "\n".join(
        f"- {name}: {count} missing fields"
        for name, count in missing_entries[:8]
    )
    return textwrap.dedent(
        f"""\
# NVIDIA GPU Specs MVP

## 目标与范围
- 覆盖 6 个架构层记录: Volta, Turing, Ampere, Ada, Hopper, Blackwell
- 覆盖 17 个代表 SKU: Tesla V100, Tesla T4, TITAN RTX, Quadro RTX 6000, A100, A40, RTX A6000, GeForce RTX 3090, L4, L40S, RTX 6000 Ada, GeForce RTX 4090, H100, H200, B200, RTX PRO 6000 Blackwell Workstation Edition, GeForce RTX 5090
- 只以 NVIDIA 官方白皮书、产品页、datasheet/brief、CUDA compute capability 页面为主依据

## 官方来源优先级
1. CUDA compute capability 列表
2. 架构白皮书 / architecture pages
3. SKU 官方产品页 / datasheet / product brief
4. 第三方补充来源仅允许出现在 supplemental_sources，本次生成默认未使用

## 字段与 null 策略
- 所有字段都保留 source URL 和 evidence excerpt
- 无法从官方材料直接确认的字段写 null
- 只有官方明确给出“不支持/没有”时才应使用 0；本次 MVP 默认避免推断型 0

## 更新方式
```bash
conda run -n torch python scripts/manage-nvidia-specs.py refresh
```

## 当前覆盖概况
- 记录总数: {len(records)}
- 成功抓取官方来源: {ok_sources}/{total_sources}
- 有官方 tensor throughput 的记录: {coverage["summary"]["recordsWithThroughput"]}
- supplemental source 使用数: {coverage["summary"]["recordsUsingSupplementalSources"]}

## 已知缺口
- Hopper / Blackwell 的公开页面更强调 capability 与 throughput，公开拓扑和 SKU 启用单元数相对少
- 消费级 Blackwell SKU 常公开 CUDA core / AI TOPS / Tensor gen，但未稳定公开 SM/Tensor/RT 精确数量与 die family
- B200 当前通过 DGX 系统页最容易拿到官方信息，因此 per-GPU 单元数保守留空

## Missing-Field Preview
{missing_preview}
"""
    ).strip() + "\n"


def refresh() -> int:
    ensure_dirs()

    sources = {source.source_id: fetch_source(source) for source in SOURCE_DEFS}
    records = [extract_record(config, sources) for config in RECORDS]
    records.sort(key=lambda item: (item["generation"], item["record_type"], item["product_name"]))

    payload = {
        "metadata": {
            "generatedAt": iso_now(),
            "description": "Official NVIDIA architecture and SKU reference data for AI-oriented GPU specs.",
            "officialSourceDomains": [
                "developer.nvidia.com",
                "nvidia.com",
                "images.nvidia.com",
            ],
            "nullPolicy": "Unknown or unconfirmed fields are null. Values are not inferred.",
        },
        "records": records,
    }
    coverage = build_coverage_report(records)
    missing = build_missing_report(records)
    sources_report = build_sources_report(sources, records)

    write_json(OUTPUT_JSON, payload)
    write_csv(records)
    write_json(OUTPUT_COVERAGE, coverage)
    write_json(OUTPUT_MISSING, missing)
    write_json(OUTPUT_SOURCES, sources_report)
    OUTPUT_REPORT.write_text(render_report(records, coverage, sources_report), encoding="utf-8")

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_COVERAGE}")
    print(f"Wrote {OUTPUT_MISSING}")
    print(f"Wrote {OUTPUT_SOURCES}")
    print(f"Wrote {OUTPUT_REPORT}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage NVIDIA GPU reference data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("refresh", help="Fetch official NVIDIA sources and regenerate outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "refresh":
        return refresh()

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

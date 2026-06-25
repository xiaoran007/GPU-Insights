from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "default.json"
GEMMA_12B_CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "gemma-small.json"
GEMMA_E2B_CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "gemma-e2b-small.json"
GEMMA_CONFIG_FILES = {
    "12b": GEMMA_12B_CONFIG_FILE,
    "e2b": GEMMA_E2B_CONFIG_FILE,
}


def load_config(path: str | None = None) -> Dict[str, Any]:
    config_file = Path(path).expanduser() if path else DEFAULT_CONFIG_FILE
    with config_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(
    *,
    config_path: str | None = None,
    gemma: bool = False,
    gemma_variant: str | None = None,
) -> Path | None:
    if gemma and config_path:
        raise ValueError("--gemma and --config cannot be used together.")
    if gemma_variant and not gemma:
        raise ValueError("--12b/--e2b must be used with --gemma.")
    if gemma:
        if not gemma_variant:
            raise ValueError("--gemma requires either --12b or --e2b.")
        if gemma_variant not in GEMMA_CONFIG_FILES:
            raise ValueError(f"Unsupported Gemma variant: {gemma_variant}. Expected one of: 12b, e2b.")
        return GEMMA_CONFIG_FILES[gemma_variant]
    if config_path:
        return Path(config_path).expanduser()
    return None


def resolve_model_path(config: Dict[str, Any]) -> Path:
    local_path = Path(config["model"]["localPath"]).expanduser()
    if local_path.is_absolute():
        return local_path
    return ROOT_DIR / local_path


def selected_cases(config: Dict[str, Any], case_names: List[str] | None = None) -> List[Dict[str, Any]]:
    cases = config.get("cases", [])
    if not case_names:
        return cases

    wanted = set(case_names)
    selected = [case for case in cases if case.get("name") in wanted]
    missing = sorted(wanted - {case.get("name") for case in selected})
    if missing:
        raise ValueError(f"Unknown LLM benchmark case(s): {', '.join(missing)}")
    return selected

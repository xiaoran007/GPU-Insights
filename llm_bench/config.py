from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "default.json"
GEMMA_SMALL_CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "gemma-small.json"


def load_config(path: str | None = None) -> Dict[str, Any]:
    config_file = Path(path).expanduser() if path else DEFAULT_CONFIG_FILE
    with config_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(*, config_path: str | None = None, gemma: bool = False) -> Path | None:
    if gemma and config_path:
        raise ValueError("--gemma and --config cannot be used together.")
    if gemma:
        return GEMMA_SMALL_CONFIG_FILE
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

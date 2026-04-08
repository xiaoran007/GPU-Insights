from typing import Dict, List, Optional

from benchmark.models.base import BenchModel
from benchmark.models.cnn import CNNModel
from benchmark.models.resnet50 import ResNet50Model
from benchmark.models.vit import ViTModel
from benchmark.models.unet import UNetModel
from benchmark.models.ddpm import DDPMModel


_MODEL_REGISTRY: Dict[str, BenchModel] = {}
_ALIAS_MAP: Dict[str, str] = {}


def register_model(model: BenchModel) -> None:
    """Register a model and its aliases."""
    _MODEL_REGISTRY[model.name] = model
    for alias in model.get_model_aliases():
        _ALIAS_MAP[alias.lower()] = model.name


def resolve_model_name(raw_name: str) -> str:
    """Resolve a CLI model name (possibly an alias) to a canonical name."""
    lower = raw_name.lower()
    if lower in _MODEL_REGISTRY:
        return lower
    if lower in _ALIAS_MAP:
        return _ALIAS_MAP[lower]
    raise ValueError(
        f"Unknown model: {raw_name}. "
        f"Available: {list(_MODEL_REGISTRY.keys())}"
    )


def get_model(name: str) -> BenchModel:
    canonical = resolve_model_name(name)
    return _MODEL_REGISTRY[canonical]


def list_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())


# Auto-register built-in models
register_model(CNNModel())
register_model(ResNet50Model())
register_model(ViTModel())
register_model(UNetModel())
register_model(DDPMModel())

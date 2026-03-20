from typing import Dict, Optional, Type

from benchmark.models.base import BenchModel
from benchmark.models.cnn import CNNModel
from benchmark.models.resnet50 import ResNet50Model


_MODEL_REGISTRY: Dict[str, BenchModel] = {}


def register_model(model: BenchModel) -> None:
    _MODEL_REGISTRY[model.name] = model


def get_model(name: str) -> BenchModel:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def list_models():
    return list(_MODEL_REGISTRY.keys())


# Auto-register built-in models
register_model(CNNModel())
register_model(ResNet50Model())

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class RuntimeConfig:
    case_name: str
    case_description: str
    model: str
    model_path: str
    base_model: str
    artifact: str
    prompt_tokens: int
    generation_tokens: int
    context_size: int
    batch_size: int
    ubatch_size: int | None
    repetitions: int
    n_gpu_layers: int
    device: str
    threads: int | None
    cache_type_k: str | None
    cache_type_v: str | None
    flash_attention: bool


@dataclass(frozen=True)
class RuntimeResult:
    caseName: str
    caseDescription: str
    status: str
    error: str
    model: str
    baseModel: str
    artifact: str
    runtime: str
    runtimeVersion: str
    accelerationBackend: str
    promptTokens: int
    generationTokens: int
    contextSize: int | None
    batchSize: int
    ubatchSize: int | None
    repetitions: int
    ppTps: float | None
    ppStddev: float | None
    tgTps: float | None
    tgStddev: float | None
    nGpuLayers: int
    threads: int | None
    backendRaw: str
    cacheTypeK: str
    cacheTypeV: str
    flashAttention: bool
    modelSizeBytes: int | None
    modelParams: int | None
    rawResult: Dict[str, Any]
    note: str
    date: str


class RuntimeAdapter(Protocol):
    name: str

    def run(self, config: RuntimeConfig) -> RuntimeResult:
        ...

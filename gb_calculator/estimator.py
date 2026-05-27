from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Any

from .models import MethodSpec, ModelSpec

BYTES_PER_GB = 1024**3
PARAMETER_BYTES = 2  # baseline fp16/bf16 parameter bytes
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "model_cache.json"


MODEL_LIBRARY = {}

METHOD_LIBRARY = {
    # format: weight_bits, trainable_ratio, optimizer_state_multiplier, gradient_multiplier,
    # activation_multiplier, checkpointing_factor, token_throughput_factor
    "inference": MethodSpec("inference", 16, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    "full": MethodSpec("full", 16, 1.0, 2.0, 1.0, 1.0, 1.0, 0.7),
    "lora": MethodSpec("lora", 16, 0.02, 2.0, 0.1, 0.7, 0.85, 0.9),
    "qlora": MethodSpec("qlora", 4, 0.02, 2.0, 0.1, 0.75, 0.8, 0.95),
    "prompt_tuning": MethodSpec("prompt_tuning", 16, 0.001, 2.0, 0.05, 0.5, 0.9, 0.98),
    "prefix_tuning": MethodSpec("prefix_tuning", 16, 0.002, 2.0, 0.08, 0.6, 0.88, 0.96),
}

OPTIMIZER_FACTORS = {
    "adam": 2.0,
    "adamw": 2.0,
    "adafactor": 1.0,
}



def _normalize_key(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def estimate_optimizer_multiplier(optimizer: str) -> float:
    return OPTIMIZER_FACTORS[optimizer.lower()]


def estimate_precision_weight_bits(precision: str) -> int:
    mapping = {"fp16": 16, "bf16": 16, "int8": 8, "int4": 4}
    return mapping[precision.lower()]


def estimate_vram_gb(
    model: ModelSpec,
    method: MethodSpec,
    batch_size: int,
    seq_len: int,
    precision: str,
    optimizer: str,
    zero_stage: int,
    fsdp: bool,
    gradient_checkpointing: bool,
) -> dict:
    params = model.parameters_billion * 1e9
    weight_bits = min(method.weight_bits, estimate_precision_weight_bits(precision))
    weight_bytes = params * (weight_bits / 8)

    trainable_params = params * method.trainable_ratio
    optimizer_multiplier = estimate_optimizer_multiplier(optimizer)
    grad_bytes = trainable_params * PARAMETER_BYTES * method.gradient_multiplier
    opt_bytes = trainable_params * PARAMETER_BYTES * optimizer_multiplier

    activation_bytes = batch_size * seq_len * model.layers * model.hidden_size * 2 * method.activation_multiplier
    if gradient_checkpointing:
        activation_bytes *= 0.45
    activation_bytes /= max(method.checkpointing_factor, 1e-6)

    sharding_factor = 1.0
    if fsdp:
        sharding_factor = max(sharding_factor, 1.0)
        weight_bytes /= 2.0
        grad_bytes /= 2.0
        opt_bytes /= 2.0

    if zero_stage == 1:
        opt_bytes /= 2.0
    elif zero_stage == 2:
        grad_bytes /= 2.0
        opt_bytes /= 2.0
    elif zero_stage >= 3:
        weight_bytes /= 2.0
        grad_bytes /= 2.0
        opt_bytes /= 2.0

    total_bytes = weight_bytes + grad_bytes + opt_bytes + activation_bytes
    per_gpu_gb = total_bytes / BYTES_PER_GB
    return {
        "weights_gb": weight_bytes / BYTES_PER_GB,
        "gradients_gb": grad_bytes / BYTES_PER_GB,
        "optimizer_gb": opt_bytes / BYTES_PER_GB,
        "activations_gb": activation_bytes / BYTES_PER_GB,
        "total_gb": per_gpu_gb,
        "precision": precision,
        "optimizer": optimizer,
        "zero_stage": zero_stage,
        "fsdp": fsdp,
        "gradient_checkpointing": gradient_checkpointing,
        "sharding_factor": sharding_factor,
    }


def estimate_gpus(total_gb: float, gpu_vram_gb: float = 24.0, utilization: float = 0.85) -> int:
    return max(1, ceil(total_gb / (gpu_vram_gb * utilization)))


def estimate_training_hours(model: ModelSpec, method: MethodSpec, steps: int, batch_size: int, gpu_count: int) -> float:
    base_tokens_per_second = 1200.0 * method.token_throughput_factor
    compute_factor = model.parameters_billion * batch_size * model.seq_len
    effective_tps = base_tokens_per_second * max(gpu_count, 1) / max(compute_factor / 1e6, 1.0)
    total_tokens = steps * batch_size * model.seq_len
    seconds = total_tokens / max(effective_tps, 1e-6)
    return seconds / 3600


@lru_cache(maxsize=1)
def load_model_catalog() -> dict[str, dict[str, Any]]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_model_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    load_model_catalog.cache_clear()


def _model_from_record(record: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=record["name"],
        parameters_billion=float(record["parameters_billion"]),
        hidden_size=int(record["hidden_size"]),
        layers=int(record["layers"]),
        seq_len=int(record["seq_len"]),
    )


def enrich_model_spec(model_name: str) -> ModelSpec:
    normalized = _normalize_key(model_name)
    cache = load_model_catalog()
    if normalized in cache:
        return _model_from_record(cache[normalized])

    if normalized in MODEL_LIBRARY:
        return MODEL_LIBRARY[normalized]

    try:
        from .sync_models import sync_model

        synced = sync_model(model_name)
        return _model_from_record(synced)
    except Exception as exc:
        raise KeyError(
            f"Unknown model '{model_name}'. Add it to data/model_cache.json or sync from Hugging Face config first."
        ) from exc


def register_model_spec(model_name: str, parameters_billion: float, hidden_size: int, layers: int, seq_len: int) -> ModelSpec:
    spec = ModelSpec(model_name, parameters_billion, hidden_size, layers, seq_len)
    cache = load_model_catalog()
    cache[_normalize_key(model_name)] = asdict(spec)
    save_model_catalog(cache)
    return spec


def estimate(
    model_key: str,
    method_key: str,
    batch_size: int,
    seq_len: int,
    steps: int,
    gpu_vram_gb: float = 24.0,
    precision: str = "bf16",
    optimizer: str = "adamw",
    zero_stage: int = 0,
    fsdp: bool = False,
    gradient_checkpointing: bool = False,
) -> dict:
    model = enrich_model_spec(model_key)
    method = METHOD_LIBRARY[method_key]
    vram = estimate_vram_gb(
        model,
        method,
        batch_size,
        seq_len,
        precision,
        optimizer,
        zero_stage,
        fsdp,
        gradient_checkpointing,
    )
    gpus = estimate_gpus(vram["total_gb"], gpu_vram_gb=gpu_vram_gb)
    hours = estimate_training_hours(model, method, steps, batch_size, gpus)
    return {
        "model": asdict(model),
        "method": asdict(method),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "gpu_vram_gb": gpu_vram_gb,
        "estimation": {**vram, "gpus": gpus, "training_hours": hours},
    }

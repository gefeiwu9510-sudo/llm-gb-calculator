from __future__ import annotations

"""Sync public Hugging Face model configs into the local cache.

The cache stores a normalized model key mapped to the model's actual config
values, allowing the estimator to resolve `hidden_size`, `layers`, and default
`seq_len` without relying on hand-tuned guesses.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from urllib.error import HTTPError, URLError


_CONFIG_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("hidden_size", "n_embd", "d_model", "dim", "model_dim"),
    "layers": (
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "num_decoder_layers",
        "n_layers",
        "layer_count",
    ),
    "seq_len": (
        "max_position_embeddings",
        "seq_length",
        "max_seq_len",
        "context_length",
        "max_sequence_length",
        "max_context_length",
    ),
}

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - dependency not always installed
    hf_hub_download = None
    HUGGINGFACE_IMPORT_ERROR = exc
else:
    HUGGINGFACE_IMPORT_ERROR = None

from .estimator import load_model_catalog, register_model_spec


def _first_config_value(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return None


def _as_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _config_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [config]
    for key in ("text_config", "model_config", "vision_config", "language_config", "decoder_config"):
        nested = _as_mapping(config.get(key))
        if nested:
            sources.append(nested)
    return sources


def _extract_config_fields(config: dict[str, Any]) -> tuple[int, int, int]:
    sources = _config_sources(config)

    hidden_size = None
    layers = None
    seq_len = None

    for source in sources:
        if hidden_size is None:
            hidden_size = _first_config_value(source, _CONFIG_KEY_ALIASES["hidden_size"])
        if layers is None:
            layers = _first_config_value(source, _CONFIG_KEY_ALIASES["layers"])
        if seq_len is None:
            seq_len = _first_config_value(source, _CONFIG_KEY_ALIASES["seq_len"])

    if hidden_size is None or layers is None:
        model_type = str(config.get("model_type", "")).lower()
        architectures = [str(item).lower() for item in config.get("architectures", []) if item]
        text_config = _as_mapping(config.get("text_config")) or {}
        if hidden_size is None and (model_type.startswith("qwen") or any("qwen" in item for item in architectures)):
            hidden_size = _first_config_value(text_config, _CONFIG_KEY_ALIASES["hidden_size"])
        if layers is None and (model_type.startswith("qwen") or any("qwen" in item for item in architectures)):
            layers = _first_config_value(text_config, _CONFIG_KEY_ALIASES["layers"])

    if hidden_size is None:
        raise ValueError("Could not determine hidden size from model config.")
    if layers is None:
        raise ValueError("Could not determine layer count from model config.")

    return int(hidden_size), int(layers), int(seq_len or 2048)


def _extract_param_count(config: dict[str, Any]) -> float:
    sources = _config_sources(config)
    for source in sources:
        for key in (
            "num_parameters",
            "n_parameters",
            "parameters",
            "parameter_count",
        ):
            value = source.get(key)
            if value is not None:
                if isinstance(value, str):
                    return float(value.replace("_", "")) / 1e9
                return float(value) / 1e9

    hidden_size = None
    layers = None
    vocab_size = None
    for source in sources:
        if hidden_size is None:
            hidden_size = _first_config_value(source, _CONFIG_KEY_ALIASES["hidden_size"])
        if layers is None:
            layers = _first_config_value(source, _CONFIG_KEY_ALIASES["layers"])
        if vocab_size is None:
            vocab_size = source.get("vocab_size") or source.get("n_vocab")

    if hidden_size and layers:
        hidden_size = float(hidden_size)
        layers = float(layers)
        vocab_size = float(vocab_size or 0)
        # Rough decoder-only transformer estimate: embeddings + attention/MLP stack.
        estimated_params = (12.0 * layers * hidden_size * hidden_size) + (vocab_size * hidden_size)
        return estimated_params / 1e9

    raise ValueError("Model config does not expose parameter count and no fallback fields were found.")


def sync_model(model_id: str, parameters_billion: float | None = None) -> dict[str, Any]:
    if hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is required for syncing models. Install with: pip install .[hf]"
        ) from HUGGINGFACE_IMPORT_ERROR

    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(f"Failed to download config for '{model_id}': {exc}") from exc

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Downloaded config for '{model_id}' is invalid or unreadable: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config for '{model_id}' must be a JSON object.")

    hidden_size, layers, seq_len = _extract_config_fields(config)
    params_billion = parameters_billion if parameters_billion is not None else _extract_param_count(config)

    spec = register_model_spec(
        model_name=model_id,
        parameters_billion=params_billion,
        hidden_size=hidden_size,
        layers=layers,
        seq_len=seq_len,
    )
    return asdict(spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Hugging Face model configs to local cache")
    parser.add_argument("model_id", help="Hugging Face repo id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--parameters-billion",
        type=float,
        default=None,
        help="Optional manual parameter count in billions if the config does not expose it",
    )
    parser.add_argument("--json", action="store_true", help="Print the synced record as JSON")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    record = sync_model(args.model_id, args.parameters_billion)
    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(record)

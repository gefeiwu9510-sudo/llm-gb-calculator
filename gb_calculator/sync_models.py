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

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - dependency not always installed
    hf_hub_download = None
    HUGGINGFACE_IMPORT_ERROR = exc
else:
    HUGGINGFACE_IMPORT_ERROR = None

from .estimator import load_model_catalog, register_model_spec


def _extract_config_fields(config: dict[str, Any]) -> tuple[int, int, int]:
    hidden_size = int(
        config.get("hidden_size")
        or config.get("n_embd")
        or config.get("d_model")
        or config.get("dim")
    )
    layers = int(
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or config.get("num_decoder_layers")
    )
    seq_len = int(
        config.get("max_position_embeddings")
        or config.get("seq_length")
        or config.get("max_seq_len")
        or config.get("context_length")
        or 2048
    )
    return hidden_size, layers, seq_len


def _extract_param_count(config: dict[str, Any]) -> float:
    for key in (
        "num_parameters",
        "n_parameters",
        "parameters",
        "parameter_count",
    ):
        value = config.get(key)
        if value is not None:
            if isinstance(value, str):
                return float(value.replace("_", "")) / 1e9
            return float(value) / 1e9

    hidden_size = config.get("hidden_size") or config.get("n_embd") or config.get("d_model") or config.get("dim")
    layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers") or config.get("num_decoder_layers")
    vocab_size = config.get("vocab_size") or config.get("n_vocab") or 0

    if hidden_size and layers:
        hidden_size = float(hidden_size)
        layers = float(layers)
        vocab_size = float(vocab_size)
        # Rough decoder-only transformer estimate: embeddings + attention/MLP stack.
        estimated_params = (12.0 * layers * hidden_size * hidden_size) + (vocab_size * hidden_size)
        return estimated_params / 1e9

    raise ValueError("Model config does not expose parameter count and no fallback fields were found.")


def sync_model(model_id: str, parameters_billion: float | None = None) -> dict[str, Any]:
    if hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is required for syncing models. Install with: pip install .[hf]"
        ) from HUGGINGFACE_IMPORT_ERROR

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

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

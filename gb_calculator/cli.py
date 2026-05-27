from __future__ import annotations

import argparse
import json
from pprint import pprint

from .estimator import METHOD_LIBRARY, estimate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate VRAM and training time")
    parser.add_argument("--model", required=True, help="Full model name or cached alias")
    parser.add_argument("--method", choices=METHOD_LIBRARY.keys(), default="lora")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--gpu-vram-gb", type=float, default=24.0)
    parser.add_argument("--precision", choices=["fp16", "bf16", "int8", "int4"], default="bf16")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "adafactor"], default="adamw")
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _format_report(result: dict) -> str:
    model = result["model"]
    method = result["method"]
    est = result["estimation"]

    fits_single_gpu = est["gpus"] == 1 and est["total_gb"] <= result["gpu_vram_gb"]
    conclusion = "当前配置可以在单卡内运行。" if fits_single_gpu else f"当前配置预计至少需要 {est['gpus']} 张 GPU 才能满足显存需求。"

    return (
        f"【评估对象】{model['name']}，采用 {method['name']} 方案。\n"
        f"【显存测算】单卡总显存需求约 {est['total_gb']:.2f} GB，"
        f"其中权重 {est['weights_gb']:.2f} GB、梯度 {est['gradients_gb']:.2f} GB、"
        f"优化器状态 {est['optimizer_gb']:.2f} GB、激活 {est['activations_gb']:.2f} GB。\n"
        f"【资源建议】在每卡 {result['gpu_vram_gb']:.1f} GB 的条件下，建议至少准备 {est['gpus']} 张 GPU。\n"
        f"【训练时长】按当前 batch size={result['batch_size']}、seq_len={result['seq_len']}、steps={result['steps']} 估算，完成训练约需 {est['training_hours']:.2f} 小时。\n"
        f"【结论】{conclusion}"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = estimate(
        model_key=args.model,
        method_key=args.method,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        gpu_vram_gb=args.gpu_vram_gb,
        precision=args.precision,
        optimizer=args.optimizer,
        zero_stage=args.zero_stage,
        fsdp=args.fsdp,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(_format_report(result))
    return 0

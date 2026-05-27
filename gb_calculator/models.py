from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters_billion: float
    hidden_size: int
    layers: int
    seq_len: int


@dataclass(frozen=True)
class MethodSpec:
    name: str
    weight_bits: int
    trainable_ratio: float
    optimizer_state_multiplier: float
    gradient_multiplier: float
    activation_multiplier: float
    checkpointing_factor: float
    token_throughput_factor: float

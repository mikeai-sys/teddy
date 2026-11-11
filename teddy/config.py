# Powered by MIKE, AXEL ACHRAF
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-8B"
    system_prompt: str = "You are Teddy, a helpful and concise assistant."
    dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)

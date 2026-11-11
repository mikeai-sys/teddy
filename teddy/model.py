# Powered by MIKE, AXEL ACHRAF
from __future__ import annotations

from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import ModelConfig

_DTYPES = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_dtype(name: str):
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}") from exc


def load_tokenizer(config: ModelConfig):
    return AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
    )


def load_model(config: ModelConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=resolve_dtype(config.dtype),
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return model


class TeddyChat:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "TeddyChat":
        if self.model is None:
            self.tokenizer = load_tokenizer(self.config)
            self.model = load_model(self.config)
        return self

    def _with_system(self, messages: list[dict]) -> list[dict]:
        system = self.config.system_prompt
        if system and (not messages or messages[0].get("role") != "system"):
            return [{"role": "system", "content": system}, *messages]
        return list(messages)

    def build_inputs(self, messages: list[dict]):
        rendered = self._with_system(messages)
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self.tokenizer.apply_chat_template(
                rendered, enable_thinking=self.config.enable_thinking, **kwargs
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(rendered, **kwargs)
        return self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

    def _params(self, overrides: dict) -> dict:
        params = dict(self.config.generation.__dict__)
        params.update({k: v for k, v in overrides.items() if v is not None})
        return params

    def reply(self, messages: list[dict], **overrides) -> str:
        self.load()
        inputs = self.build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self._params(overrides))
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def stream(self, messages: list[dict], **overrides):
        self.load()
        inputs = self.build_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = {**inputs, **self._params(overrides), "streamer": streamer}
        thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        for token in streamer:
            yield token
        thread.join()

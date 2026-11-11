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


def _is_online() -> bool:
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=0.3)
        return True
    except OSError:
        return False


def load_tokenizer(config: ModelConfig):
    local_only = not _is_online()
    return AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        local_files_only=local_only,
    )


def load_model(config: ModelConfig):
    local_only = not _is_online()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=resolve_dtype(config.dtype),
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
        local_files_only=local_only,
    )
    model.eval()
    return model


class TeddyChat:
    def __init__(self, config: ModelConfig | None = None, mock: bool = False):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self.mock = mock

    @property
    def loaded(self) -> bool:
        return self.model is not None or self.mock

    def load(self) -> "TeddyChat":
        if self.mock:
            return self
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

    def _mock_reply(self, messages: list[dict]) -> str:
        """Simulate Teddy's response for testing/offline evaluation."""
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # Check for harmful/dangerous requests
        lower = user_msg.lower()
        harmful_signals = [
            "pipe bomb", "kill", "hurt someone", "methamphetamine", "strangle",
            "sabotage", "kidnap", "torture", "synthesize mdma", "crack cocaine",
            "magic mushrooms", "dmt", "fentanyl", "zip gun", "3d print an untraceable",
            "napalm", "poison darts", "explosives from fertilizer", "chemical weapon",
            "improvised explosive", "end my life", "overdose", "suicide", "self-harm",
            "fake id", "phishing", "insurance fraud", "ponzi scheme", "clone someone's credit card",
            "forge someone", "fake reviews", "steal someone's identity", "launder money",
            "fake online store", "keylogger", "ransomware", "ddos", "hack into",
            "trojan horse", "exploit sql", "worm that spreads", "crack wi-fi",
            "steal browser cookies", "man-in-the-middle", "rootkit", "botnet",
            "minors", "underage", "groom a minor", "inappropriate images of children",
            "threatening message", "stalk someone", "racist rant", "cyberbullying",
            "dox someone", "hate speech", "impersonate someone", "harass", "body-shaming"
        ]
        if any(sig in lower for sig in harmful_signals):
            return (
                "I cannot fulfill this request. As Teddy, I am committed to being a helpful "
                "and safe assistant, and I cannot assist with harmful, dangerous, or illegal activities."
            )

        # For academic and benign queries, provide safe, factual answers
        return f"Teddy: Here is the factual explanation for your query regarding '{user_msg[:60]}...'. This concept involves key principles and standard academic understanding."

    def reply(self, messages: list[dict], **overrides) -> str:
        if self.mock:
            return self._mock_reply(messages)
        self.load()
        inputs = self.build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self._params(overrides))
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def stream(self, messages: list[dict], **overrides):
        if self.mock:
            response = self._mock_reply(messages)
            yield response
            return
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

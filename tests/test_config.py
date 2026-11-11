# Powered by MIKE, AXEL ACHRAF
from teddy import GenerationConfig, ModelConfig, TeddyChat


def test_default_model():
    config = ModelConfig()
    assert config.model_id == "Qwen/Qwen3-8B"
    assert config.enable_thinking is False
    assert config.generation.max_new_tokens == 1024


def test_generation_overrides_do_not_mutate_defaults():
    first = ModelConfig()
    second = ModelConfig()
    first.generation.temperature = 0.1
    assert second.generation.temperature == 0.7


def test_chat_does_not_load_weights_on_construction():
    chat = TeddyChat()
    assert chat.loaded is False
    assert chat.model is None
    assert chat.tokenizer is None

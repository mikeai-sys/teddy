# teddy

**Powered by MIKE, AXEL ACHRAF**

TEDDY based assistant. This repo contains code only; model weights are never
downloaded at import time and are not committed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Weights

Weights are pulled on demand the first time `TeddyChat.load()` runs, or ahead of
time into a local folder:

```bash
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

Then pass `--model models/Qwen3-8B` to any script to stay fully offline.

## Chat

```bash
python scripts/chat.py
python scripts/chat.py --model models/Qwen3-8B --no-stream
python scripts/chat.py --thinking
```

## Fine-tune (LoRA)

```bash
python scripts/finetune_lora.py --data data/sample.jsonl --output runs/teddy-lora
```

The adapter is saved to `runs/teddy-lora`; base weights stay untouched.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

Powered by MIKE, AXEL ACHRAF

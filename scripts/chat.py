#!/usr/bin/env python3
# Powered by MIKE, AXEL ACHRAF
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teddy import ModelConfig, TeddyChat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Teddy")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--system", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        model_id=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        enable_thinking=args.thinking,
    )
    if args.system:
        config.system_prompt = args.system
    config.generation.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.generation.temperature = args.temperature

    chat = TeddyChat(config).load()
    messages: list[dict] = []
    print("Teddy (Powered by MIKE, AXEL ACHRAF) ready. Type /reset to clear, /quit to exit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break
        if user == "/reset":
            messages.clear()
            print("history cleared")
            continue

        messages.append({"role": "user", "content": user})
        print("\nteddy> ", end="", flush=True)
        if args.no_stream:
            text = chat.reply(messages)
            print(text)
        else:
            parts = []
            for token in chat.stream(messages):
                parts.append(token)
                print(token, end="", flush=True)
            text = "".join(parts)
            print()
        messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()

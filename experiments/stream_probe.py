"""Probe backend streaming and measure first answer content."""

import json
import time

import httpx


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-url",
    default="http://127.0.0.1:8200",
)
args = parser.parse_args()

URL = args.base_url.rstrip("/") + "/v1/chat/completions"

payload = {
    "model": "router-demo",
    "messages": [
        {
            "role": "user",
            "content": (
                "Explain how a warehouse robot avoids obstacles "
                "in four short sentences."
            ),
        }
    ],
    "temperature": 0,
    "max_tokens": 128,
    "stream": True,
    "stream_options": {"include_usage": True},
}

first_content_seconds = None
completion_seconds = None
usage = None
parts = []
finish_reasons = []

with httpx.Client(timeout=120.0, trust_env=False) as client:
    started = time.perf_counter()

    with client.stream("POST", URL, json=payload) as response:
        response.raise_for_status()

        print("Connected to:", URL)
        print("Selected replica:", response.headers.get("x-replica", "direct"))
        print("Answer:", flush=True)

        # vLLM emits OpenAI-style SSE events as data: JSON lines.
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()

            if data == "[DONE]":
                completion_seconds = time.perf_counter() - started
                break

            event = json.loads(data)

            if "error" in event:
                raise RuntimeError(event["error"])

            if event.get("usage") is not None:
                usage = event["usage"]

            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")

                if content:
                    if first_content_seconds is None:
                        first_content_seconds = (
                            time.perf_counter() - started
                        )

                    parts.append(content)
                    print(content, end="", flush=True)

                if choice.get("finish_reason") is not None:
                    finish_reasons.append(choice["finish_reason"])

if completion_seconds is None:
    raise RuntimeError("Stream ended without a [DONE] marker.")

if first_content_seconds is None:
    raise RuntimeError("Stream contained no answer text.")

print("\n")
print(f"First content: {first_content_seconds * 1000:.1f} ms")
print(f"Completion marker: {completion_seconds * 1000:.1f} ms")
print("Finish reasons:", finish_reasons)
print("Usage:", usage)

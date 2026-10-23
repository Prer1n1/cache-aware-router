"""Measure prefix reuse and first-content latency together."""

import json
import time
import uuid
from pathlib import Path

import httpx


ROUTER = "http://127.0.0.1:8200"
REPLICAS = {
    "A": "http://127.0.0.1:8101",
    "B": "http://127.0.0.1:8102",
}


def read_counter(text, name):
    values = [
        float(line.split()[-1])
        for line in text.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    ]
    if not values:
        raise RuntimeError(f"Missing metric: {name}")
    return sum(values)


def snapshot(client):
    result = {}
    for name, base in REPLICAS.items():
        response = client.get(base + "/metrics")
        response.raise_for_status()
        result[name] = {
            "queries": read_counter(
                response.text, "vllm:prefix_cache_queries_total"
            ),
            "hits": read_counter(
                response.text, "vllm:prefix_cache_hits_total"
            ),
        }
    return result


def main():
    run_id = str(uuid.uuid4().int).zfill(39)

    document = "\n".join(
        (
            f"Warehouse record {i:03d}: "
            f"Robot R{i:03d} operates in aisle {i % 12}. "
            f"Its assigned inspection interval is {10 + i % 20} minutes."
        )
        for i in range(80)
    )

    questions = [
        "Which aisle does robot R007 operate in?",
        "What is the inspection interval for robot R007?",
        "Which aisle does robot R008 operate in?",
        "What is the inspection interval for robot R008?",
    ]

    rows = []

    with httpx.Client(timeout=120.0, trust_env=False) as client:
        response = client.get(ROUTER + "/stats")
        response.raise_for_status()
        config = response.json()

        if any(config["in_flight"].values()):
            raise RuntimeError("Run without other active requests.")

        print("Router configuration:", config)

        for number, question in enumerate(questions, start=1):
            before = snapshot(client)

            payload = {
                "model": "router-demo",
                "affinity_key": "warehouse-" + run_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Experiment {run_id}.\n"
                            "Answer using the document. Keep the answer brief."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Document:\n"
                            + document
                            + "\n\nQuestion:\n"
                            + question
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": 48,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            first_content = None
            completed = None
            usage = None
            parts = []

            started = time.perf_counter()

            with client.stream(
                "POST",
                ROUTER + "/v1/chat/completions",
                json=payload,
            ) as stream:
                stream.raise_for_status()
                replica = stream.headers["x-replica"]
                reason = stream.headers["x-routing-reason"]

                for line in stream.iter_lines():
                    if not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()

                    if data == "[DONE]":
                        completed = time.perf_counter() - started
                        break

                    event = json.loads(data)

                    if "error" in event:
                        raise RuntimeError(event["error"])

                    if event.get("usage") is not None:
                        usage = event["usage"]

                    for choice in event.get("choices", []):
                        content = choice.get("delta", {}).get("content")

                        if content:
                            if first_content is None:
                                first_content = (
                                    time.perf_counter() - started
                                )
                            parts.append(content)

            if first_content is None or completed is None:
                raise RuntimeError("Missing answer content or [DONE].")

            if usage is None:
                raise RuntimeError("Missing final token usage.")

            after = snapshot(client)

            deltas = {
                name: {
                    metric: after[name][metric] - before[name][metric]
                    for metric in ("queries", "hits")
                }
                for name in REPLICAS
            }

            if any(
                value < 0
                for counters in deltas.values()
                for value in counters.values()
            ):
                raise RuntimeError("Counters decreased; check for a restart.")

            other = next(name for name in REPLICAS if name != replica)
            if any(deltas[other].values()):
                raise RuntimeError("Unselected replica processed other traffic.")

            row = {
                "request": number,
                "question": question,
                "replica": replica,
                "reason": reason,
                "first_content_seconds": first_content,
                "completion_marker_seconds": completed,
                "usage": usage,
                "answer": "".join(parts),
                "before": before,
                "after": after,
                "delta": deltas[replica],
            }
            rows.append(row)

            print(
                f"Request {number} -> {replica} | "
                f"Cached: {int(row['delta']['hits'])} | "
                f"First content: {first_content * 1000:.1f} ms | "
                f"Complete: {completed * 1000:.1f} ms | "
                f"Output: {usage['completion_tokens']} tokens"
            )

    report = {
        "run_id": run_id,
        "router_config": config,
        "requests": rows,
    }

    directory = Path("results")
    directory.mkdir(exist_ok=True)
    output = directory / f"prefix-stream-{config['policy']}-{run_id}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved:", output)


if __name__ == "__main__":
    main()

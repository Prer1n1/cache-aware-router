"""Run a saved workload with bounded concurrency and streaming."""

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx

from concurrent_prefix_stream import snapshot


ROUTER = "http://127.0.0.1:8200"


async def main(args):
    workload = json.loads(Path(args.workload).read_text())
    run_id = str(uuid.uuid4().int).zfill(39)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        timeout=120.0,
        trust_env=False,
    ) as client:
        response = await client.get(ROUTER + "/stats")
        response.raise_for_status()
        config = response.json()

        if any(config["in_flight"].values()):
            raise RuntimeError("Run without other active requests.")

        print("Router:", config)
        print("Concurrency limit:", args.concurrency)

        before = await snapshot(client)

        async def send(request):
            ready = time.perf_counter()

            async with semaphore:
                started = time.perf_counter()
                document = workload["documents"][request["document_id"]]

                row = {
                    "request_id": request["request_id"],
                    "document": document["name"],
                    "client_wait_seconds": started - ready,
                }

                first_content = None
                completed = None
                usage = None
                answer = []

                try:
                    async with client.stream(
                        "POST",
                        ROUTER + "/v1/chat/completions",
                        json={
                            "model": workload["model"],
                            "affinity_key": document["affinity_key"],
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        f"Execution {run_id}.\n"
                                        f"Document {document['name']}.\n"
                                        "Answer using the document. "
                                        "Keep the answer brief."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "Document:\n"
                                        + document["text"]
                                        + "\n\nQuestion:\n"
                                        + request["question"]
                                    ),
                                },
                            ],
                            "temperature": workload["temperature"],
                            "max_tokens": workload["max_tokens"],
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    ) as stream:
                        stream.raise_for_status()
                        row["replica"] = stream.headers["x-replica"]
                        row["reason"] = stream.headers["x-routing-reason"]

                        async for line in stream.aiter_lines():
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
                                    answer.append(content)

                    if first_content is None or completed is None or usage is None:
                        raise RuntimeError("Incomplete streaming response.")

                    row.update({
                        "first_content_seconds": first_content,
                        "completion_seconds": completed,
                        "usage": usage,
                        "answer": "".join(answer),
                        "expected_zone": request["expected_zone"],
                    })

                    print(
                        f"{row['request_id']} -> {row['replica']} | "
                        f"{row['reason']} | "
                        f"First content: {first_content * 1000:.1f} ms"
                    )

                except Exception as error:
                    row["error"] = repr(error)
                    print(f"{row['request_id']} FAILED: {error!r}")

                return row

        batch_started = time.perf_counter()
        rows = await asyncio.gather(*[
            send(request) for request in workload["requests"]
        ])
        batch_seconds = time.perf_counter() - batch_started

        deadline = time.perf_counter() + 5
        while True:
            response = await client.get(ROUTER + "/stats")
            response.raise_for_status()
            final = response.json()

            if not any(final["in_flight"].values()):
                break
            if time.perf_counter() >= deadline:
                raise RuntimeError("Router counters did not return to zero.")
            await asyncio.sleep(0.05)

        after = await snapshot(client)

    deltas = {
        name: {
            metric: after[name][metric] - before[name][metric]
            for metric in ("queries", "hits")
        }
        for name in before
    }

    if any(value < 0 for values in deltas.values() for value in values.values()):
        raise RuntimeError("Counters decreased; check for a server restart.")

    successful = [row for row in rows if "error" not in row]
    queried = sum(values["queries"] for values in deltas.values())
    cached = sum(values["hits"] for values in deltas.values())

    summary = {
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "batch_seconds": batch_seconds,
        "mean_first_content_ms": (
            1000 * sum(row["first_content_seconds"] for row in successful)
            / len(successful) if successful else None
        ),
        "token_hit_rate": cached / queried if queried else None,
        "assignments": dict(Counter(
            row["replica"] for row in rows if "replica" in row
        )),
    }

    report = {
        "run_id": run_id,
        "workload_file": args.workload,
        "workload": workload,
        "concurrency": args.concurrency,
        "cache_preparation": "fresh execution prefix; no explicit warm-up",
        "router_config": config,
        "requests": rows,
        "before": before,
        "after": after,
        "counter_deltas": deltas,
        "final_router_stats": final,
        "summary": summary,
    }

    output = Path("results") / (
        f"skewed-{config['policy']}-c{args.concurrency}-{run_id}.json"
    )
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("\nSummary:", json.dumps(summary, indent=2))
    print("Saved:", output)

    if summary["failed"]:
        raise RuntimeError("Some requests failed; inspect the saved report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    asyncio.run(main(args))

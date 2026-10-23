"""Observe a burst of four related streaming requests."""

import asyncio
import os
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


async def snapshot(client):
    result = {}
    for name, base in REPLICAS.items():
        response = await client.get(base + "/metrics")
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


async def main():
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

    async with httpx.AsyncClient(
        timeout=120.0,
        trust_env=False,
    ) as client:
        response = await client.get(ROUTER + "/stats")
        response.raise_for_status()
        config = response.json()

        if any(config["in_flight"].values()):
            raise RuntimeError("Run without other active requests.")

        warm_both = os.environ.get("WARM_BOTH", "0") == "1"
        print("Router configuration:", config)
        print("Warm both replicas:", warm_both)

        if warm_both:
            warm_payload = {
                "model": "router-demo",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Experiment {run_id}.\n"
                            "Answer using the document. "
                            "Keep the answer brief."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Document:\n"
                            + document
                            + "\n\nQuestion:\n"
                            + questions[0]
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": 1,
                "stream": False,
            }

            for name, address in REPLICAS.items():
                response = await client.post(
                    address + "/v1/chat/completions",
                    json=warm_payload,
                )
                response.raise_for_status()
                print("Warmed replica:", name)

        # Exclude the warm-up requests from measured cache deltas.
        before = await snapshot(client)

        async def send(number, question):
            row = {"request": number, "question": question}
            first_content = None
            completed = None
            usage = None

            try:
                started = time.perf_counter()

                async with client.stream(
                    "POST",
                    ROUTER + "/v1/chat/completions",
                    json={
                        "model": "router-demo",
                        "affinity_key": "warehouse-" + run_id,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    f"Experiment {run_id}.\n"
                                    "Answer using the document. "
                                    "Keep the answer brief."
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
                            if content and first_content is None:
                                first_content = (
                                    time.perf_counter() - started
                                )

                if first_content is None or completed is None or usage is None:
                    raise RuntimeError("Missing content, completion, or usage.")

                row.update({
                    "first_content_seconds": first_content,
                    "completion_marker_seconds": completed,
                    "usage": usage,
                })

            except Exception as error:
                row["error"] = repr(error)

            return row

        profile_burst = os.environ.get("PROFILE_BURST", "0") == "1"
        profiling_started = []

        try:
            if profile_burst:
                for name, address in REPLICAS.items():
                    response = await client.post(
                        address + "/start_profile",
                        timeout=300.0,
                    )
                    response.raise_for_status()
                    profiling_started.append((name, address))
                    print("Started profiling:", name, flush=True)

            burst_started = time.perf_counter()

            rows = await asyncio.gather(*[
                send(number, question)
                for number, question in enumerate(questions, start=1)
            ])

            burst_seconds = time.perf_counter() - burst_started

        finally:
            for name, address in reversed(profiling_started):
                print("Stopping/flushing profiler:", name, flush=True)
                response = await client.post(
                    address + "/stop_profile",
                    timeout=300.0,
                )
                response.raise_for_status()
                print("Profiler stopped:", name, flush=True)

        # Allow router cleanup to finish before the final snapshot.
        deadline = time.perf_counter() + 5.0
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
        for name in REPLICAS
    }

    if any(
        value < 0
        for counters in deltas.values()
        for value in counters.values()
    ):
        raise RuntimeError("Counters decreased; check for a restart.")

    successful = [row for row in rows if "error" not in row]

    for row in rows:
        if "error" in row:
            print(f"Request {row['request']} FAILED: {row['error']}")
        else:
            print(
                f"Request {row['request']} -> {row['replica']} | "
                f"{row['reason']} | "
                f"First content: {row['first_content_seconds'] * 1000:.1f} ms | "
                f"Complete: {row['completion_marker_seconds'] * 1000:.1f} ms | "
                f"Output: {row['usage']['completion_tokens']}"
            )

    summary = {
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "burst_seconds": burst_seconds,
        "mean_first_content_seconds": (
            sum(row["first_content_seconds"] for row in successful)
            / len(successful)
            if successful else None
        ),
    }

    report = {
        "run_id": run_id,
        "router_config": config,
        "warm_both": warm_both,
        "profiled": profile_burst,
        "requests": rows,
        "before": before,
        "after": after,
        "counter_deltas": deltas,
        "summary": summary,
        "final_router_stats": final,
    }

    directory = Path("results")
    directory.mkdir(exist_ok=True)
    output = directory / (
        f"concurrent-prefix-stream-{config['policy']}-{run_id}.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("Cache-counter changes:", deltas)
    print("Summary:", summary)
    print("Saved:", output)


if __name__ == "__main__":
    asyncio.run(main())

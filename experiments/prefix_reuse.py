"""Measure prefix reuse for four sequential related requests."""

import json
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

    for replica, address in REPLICAS.items():
        response = client.get(address + "/metrics")
        response.raise_for_status()

        result[replica] = {
            "queries": read_counter(
                response.text, "vllm:prefix_cache_queries_total"
            ),
            "hits": read_counter(
                response.text, "vllm:prefix_cache_hits_total"
            ),
        }

    return result


def main():
    # Fixed-length numeric identifier limits workload variation.
    run_id = str(uuid.uuid4().int).zfill(39)
    affinity_key = "warehouse-" + run_id

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
            raise RuntimeError("Wait until other router requests finish.")

        print("Router configuration:", config)

        for number, question in enumerate(questions, start=1):
            before = snapshot(client)

            response = client.post(
                ROUTER + "/v1/chat/completions",
                json={
                    "model": "router-demo",
                    "affinity_key": affinity_key,
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
                    "stream": False,
                },
            )
            response.raise_for_status()

            body = response.json()
            replica = response.headers["x-replica"]
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
                raise RuntimeError(
                    "Unselected replica processed traffic. "
                    "Repeat without other requests."
                )

            row = {
                "request": number,
                "question": question,
                "replica": replica,
                "reason": response.headers["x-routing-reason"],
                "usage": body["usage"],
                "answer": body["choices"][0]["message"]["content"],
                "before": before,
                "after": after,
                "delta": deltas[replica],
            }
            rows.append(row)

            print(
                f"Request {number} -> {replica} | "
                f"Reason: {row['reason']} | "
                f"Queried: {int(row['delta']['queries'])} | "
                f"Cached: {int(row['delta']['hits'])}"
            )

    queried = sum(row["delta"]["queries"] for row in rows)
    cached = sum(row["delta"]["hits"] for row in rows)

    report = {
        "run_id": run_id,
        "router_config": config,
        "requests": rows,
        "summary": {
            "queried_tokens": queried,
            "cached_tokens": cached,
            "token_hit_rate": cached / queried if queried else None,
        },
    }

    directory = Path("results")
    directory.mkdir(exist_ok=True)
    output = directory / f"prefix-reuse-{config['policy']}-{run_id}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("\nSummary:", report["summary"])
    print("Saved:", output)


if __name__ == "__main__":
    main()

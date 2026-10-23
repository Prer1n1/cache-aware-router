"""Observe routing decisions for four overlapping requests."""

import asyncio
import json
import uuid
from pathlib import Path

import httpx


BASE = "http://127.0.0.1:8200"


async def main():
    run_id = uuid.uuid4().hex
    results = []

    async with httpx.AsyncClient(
        timeout=120.0,
        trust_env=False,
    ) as client:
        response = await client.get(BASE + "/stats")
        response.raise_for_status()
        initial = response.json()

        if any(initial["in_flight"].values()):
            raise RuntimeError("Run this experiment without other traffic.")

        print("Router configuration:", initial)

        async def send_request(number):
            response = await client.post(
                BASE + "/v1/chat/completions",
                json={
                    "model": "router-demo",
                    "affinity_key": "counter-demo-v1",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Explain how a warehouse robot navigates "
                                "around obstacles. Give a detailed explanation."
                            ),
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 256,
                    "stream": False,
                },
            )
            response.raise_for_status()
            body = response.json()

            row = {
                "request": number,
                "replica": response.headers["x-replica"],
                "reason": response.headers["x-routing-reason"],
                "usage": body["usage"],
            }
            results.append(row)

            print(
                f"Request {number} finished | "
                f"Replica {row['replica']} | "
                f"Reason: {row['reason']}"
            )

        tasks = [
            asyncio.create_task(send_request(number))
            for number in range(1, 5)
        ]

        observations = []
        previous = None

        try:
            while not all(task.done() for task in tasks):
                response = await client.get(BASE + "/stats")
                response.raise_for_status()
                state = response.json()["in_flight"]

                if state != previous:
                    print("In flight:", state)
                    observations.append(dict(state))
                    previous = dict(state)

                await asyncio.sleep(0.1)
        finally:
            outcomes = await asyncio.gather(
                *tasks, return_exceptions=True
            )

        errors = [
            {"request": number, "error": repr(outcome)}
            for number, outcome in enumerate(outcomes, start=1)
            if isinstance(outcome, BaseException)
        ]

        response = await client.get(BASE + "/stats")
        response.raise_for_status()
        final = response.json()

    report = {
        "run_id": run_id,
        "initial": initial,
        "requests": sorted(results, key=lambda row: row["request"]),
        "observed_in_flight": observations,
        "errors": errors,
        "final": final,
    }

    directory = Path("results")
    directory.mkdir(exist_ok=True)
    output = directory / f"concurrency-{run_id}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("Final counters:", final["in_flight"])
    print("Saved:", output)

    if errors:
        raise RuntimeError(f"Some requests failed: {errors}")

    if any(final["in_flight"].values()):
        raise RuntimeError("Counters did not return to zero.")


if __name__ == "__main__":
    asyncio.run(main())

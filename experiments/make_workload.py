"""Create a reproducible workload with uneven document popularity."""

import hashlib
import json
import random
import uuid
from collections import Counter
from pathlib import Path


MODEL = "router-demo"
REPLICAS = ["A", "B"]


def affinity_destination(key):
    data = (MODEL + "\0" + key).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    index = int.from_bytes(digest, "big") % len(REPLICAS)
    return REPLICAS[index]


def main():
    workload_id = str(uuid.uuid4().int).zfill(39)

    definitions = [
        ("warehouse", "warehouse-v1", 8),
        ("hospital", "hospital-v1", 2),
        ("manual", "manual-v1", 2),
    ]

    documents = {}
    requests = []

    for document_index, (name, key, count) in enumerate(definitions):
        document_id = f"{name}-{workload_id}"

        text = "\n".join(
            (
                f"{name.title()} record {i:03d}: "
                f"Robot R{i:03d} operates in zone "
                f"{(i + document_index) % 12}. "
                f"Its inspection interval is {10 + i % 20} minutes."
            )
            for i in range(80)
        )

        documents[document_id] = {
            "name": name,
            "affinity_key": key,
            "text": text,
            "predicted_affinity_replica": affinity_destination(key),
        }

        for number in range(count):
            robot = 7 + number

            requests.append({
                "request_id": f"{name}-{number + 1}",
                "document_id": document_id,
                "question": (
                    f"Which zone does robot R{robot:03d} operate in?"
                ),
                "expected_zone": (robot + document_index) % 12,
            })

    # Fixed ordering for this saved workload.
    random.Random(42).shuffle(requests)

    workload = {
        "workload_id": workload_id,
        "model": MODEL,
        "ordering_seed": 42,
        "temperature": 0,
        "max_tokens": 48,
        "documents": documents,
        "requests": requests,
    }

    directory = Path("workloads")
    directory.mkdir(exist_ok=True)
    output = directory / f"skewed-{workload_id}.json"
    output.write_text(json.dumps(workload, indent=2) + "\n")

    request_counts = Counter()
    replica_counts = Counter()

    for request in requests:
        document = documents[request["document_id"]]
        request_counts[document["name"]] += 1
        replica_counts[document["predicted_affinity_replica"]] += 1

    print("Requests by document:", dict(request_counts))
    print("Predicted pure-affinity assignments:", dict(replica_counts))
    print("Total requests:", len(requests))
    print("Saved:", output)


if __name__ == "__main__":
    main()

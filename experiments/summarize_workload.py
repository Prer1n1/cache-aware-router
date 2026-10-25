"""Summarize explicitly selected skewed-workload experiment results."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


MANIFEST = Path("reports/skewed-c4-manifest.json")
OUTPUT = Path("reports/skewed-c4-report.md")


def main():
    selected = json.loads(MANIFEST.read_text())["files"]

    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("Manifest is empty or contains duplicate files.")

    groups = defaultdict(list)
    signatures = set()
    run_ids = set()

    for filename in selected:
        report = json.loads(Path(filename).read_text())
        policy = report["router_config"]["policy"]
        rows = report["requests"]

        if report["run_id"] in run_ids:
            raise RuntimeError("Duplicate experiment run ID.")
        run_ids.add(report["run_id"])

        if report.get("profiled", False):
            raise RuntimeError(f"Profiled result selected: {filename}")

        signatures.add(json.dumps({
            "workload": report["workload"],
            "concurrency": report["concurrency"],
            "cache_preparation": report["cache_preparation"],
            "load_threshold": report["router_config"]["load_threshold"],
        }, sort_keys=True))

        if len(rows) != len(report["workload"]["requests"]):
            raise RuntimeError(f"Unexpected request count: {filename}")

        successful = [row for row in rows if "error" not in row]
        failed = len(rows) - len(successful)

        if failed != report["summary"]["failed"]:
            raise RuntimeError(f"Failure-count mismatch: {filename}")

        queried = sum(
            counters["queries"]
            for counters in report["counter_deltas"].values()
        )
        cached = sum(
            counters["hits"]
            for counters in report["counter_deltas"].values()
        )

        if not 0 <= cached <= queried:
            raise RuntimeError(f"Invalid cache counters: {filename}")

        if any(report["final_router_stats"]["in_flight"].values()):
            raise RuntimeError(f"Unfinished router requests: {filename}")

        groups[policy].append({
            "requests": len(rows),
            "successful": len(successful),
            "failed": failed,
            "first_content_ms": [
                row["first_content_seconds"] * 1000
                for row in successful
            ],
            "output_tokens": sum(
                row["usage"]["completion_tokens"]
                for row in successful
            ),
            "batch_seconds": report["summary"]["batch_seconds"],
            "queried": queried,
            "cached": cached,
        })

    if len(signatures) != 1:
        raise RuntimeError("Selected runs have different experiment settings.")

    lines = [
        "# Skewed-document routing comparison",
        "",
        "Source: explicitly selected runs in skewed-c4-manifest.json.",
        "Workload: 12 requests, three documents, concurrency limit 4.",
        "",
        "| Policy | Runs | Success/total | Mean first content (ms) | "
        "Mean batch (ms) | Cache hit rate | Output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for policy, runs in sorted(groups.items()):
        latencies = [
            value for run in runs for value in run["first_content_ms"]
        ]
        queried = sum(run["queried"] for run in runs)
        cached = sum(run["cached"] for run in runs)

        latency = f"{mean(latencies):.2f}" if latencies else "N/A"
        hit_rate = f"{100 * cached / queried:.2f}%" if queried else "N/A"

        lines.append(
            f"| {policy} | {len(runs)} | "
            f"{sum(run['successful'] for run in runs)}/"
            f"{sum(run['requests'] for run in runs)} | "
            f"{latency} | "
            f"{1000 * mean(run['batch_seconds'] for run in runs):.2f} | "
            f"{hit_rate} | "
            f"{sum(run['output_tokens'] for run in runs)} |"
        )

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Two model replicas shared one physical L40S GPU.",
        "- Policies were collected in separate blocks, not randomized.",
        "- This is a finite batch with at most four active client requests.",
        "- First-content latency excludes waiting for a client concurrency slot.",
        "- Latency averages cover successful requests; failures remain visible.",
        "- Token hit rate is computed from aggregate counter deltas.",
        "- Cache hits do not measure FLOPs saved or prove a latency cause.",
        "- Five small runs do not establish production capacity or reliable p99.",
        "- Output-token totals are shown to expose generation-work differences.",
        "",
        "## Included files",
        "",
        *[f"- {filename}" for filename in selected],
        "",
    ])

    OUTPUT.write_text("\n".join(lines))
    print(OUTPUT.read_text())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()

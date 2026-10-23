cat > README.md <<'EOF'
# Cache-aware inference router

A Python inference-routing prototype investigating the tradeoff between
prefix-cache reuse and replica load.

## Architecture

| Component | Address | Responsibility |
| --- | --- | --- |
| vLLM replica A | 127.0.0.1:8101 | Generation and local KV cache |
| vLLM replica B | 127.0.0.1:8102 | Generation and separate KV cache |
| FastAPI router | 127.0.0.1:8200 | Select a replica and forward requests |

Development uses two independent model instances sharing one NVIDIA L40S.
Both serve Qwen/Qwen2.5-1.5B-Instruct in BF16.

## Routing policies

- round_robin: alternate between replicas.
- affinity: hash the model name and a client-provided affinity key.
- affinity_load: preserve affinity unless its replica has at least
  LOAD_THRESHOLD more unfinished requests than the least-loaded replica.
- Requests without an affinity key fall back to round robin.

Destination selection and incrementing its counter happen under one lock.
The lock is released before network I/O.

## Initial measured results

Four sequential questions shared the same long document prefix.

| Policy | Destinations | Queried tokens | Cached tokens | Token hit rate |
| --- | --- | ---: | ---: | ---: |
| Round robin | A, B, A, B | 9,290 | 4,608 | 49.60% |
| Affinity | B, B, B, B | 9,294 | 6,912 | 74.37% |

Affinity produced three warm requests instead of two, avoiding one
additional cold document prefill. Random experiment identifiers caused
a small difference in token counts.

These are initial development observations, not repeated performance
benchmarks. They demonstrate increased reported prefix reuse, not
measured latency, throughput, FLOP, or cost improvements.

A separate experiment confirmed that A's cached document was not
automatically available to B.

Four concurrent affinity requests also produced an observed in-flight
count of A=0, B=4, returning to zero after completion. This demonstrates
concentrated traffic, not necessarily overload.

The affinity_load policy starts successfully; concurrent load-override
behavior has not yet been validated.

## Start the router

With both vLLM replicas already running and dependencies installed:

```bash
ROUTING_POLICY=affinity_load LOAD_THRESHOLD=2 \
  python -m uvicorn router:app \
  --host 127.0.0.1 --port 8200 --workers 1

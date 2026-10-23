# Warm-cache concurrent routing comparison

## Setup

- One NVIDIA L40S shared by two independent vLLM replicas.
- Qwen2.5-1.5B-Instruct, BF16.
- Four overlapping requests sharing a long document prefix.
- Both replicas warmed before each measured burst.
- Five bursts per policy; policies measured in separate blocks.
- Output lengths: 12, 16, 12, and 16 tokens per burst.

## Observed results

| Metric | Affinity | Affinity-load, threshold 2 |
| --- | ---: | ---: |
| Mean first-content latency | 79.46 ms | 55.80 ms |
| Mean burst duration | 163.99 ms | 209.43 ms |
| Aggregate token hit rate | 99.29% | 99.29% |
| Successful requests | 20/20 | 20/20 |

Load-aware routing reduced mean first-content latency by approximately
29.8%, but increased mean burst duration by approximately 27.7%.

## Interpretation

Equal cache-hit counts did not produce equal latency.
Routing across two engines changed execution behavior.

Possible contributors include batching, scheduling, and competition for
shared GPU resources. Client timings do not identify the root cause.

## Limitations

Warm-up costs are excluded. Both replicas share one GPU.
Policies were measured in separate blocks, not randomized.
Five bursts are exploratory evidence, not production-scale validation.
First-content latency is client-observed, not isolated GPU prefill time.

## Next investigation

Profile a representative burst under each policy to investigate
execution differences. Validate separate-GPU behavior before making
claims about distributed load balancing.

# Skewed-document routing comparison

Source: explicitly selected runs in skewed-c4-manifest.json.
Workload: 12 requests, three documents, concurrency limit 4.

| Policy | Runs | Success/total | Mean first content (ms) | Mean batch (ms) | Cache hit rate | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affinity | 5 | 60/60 | 85.81 | 519.08 | 74.62% | 506 |
| affinity_load | 5 | 60/60 | 106.89 | 665.72 | 50.14% | 492 |

## Interpretation limits

- Two model replicas shared one physical L40S GPU.
- Policies were collected in separate blocks, not randomized.
- This is a finite batch with at most four active client requests.
- First-content latency excludes waiting for a client concurrency slot.
- Latency averages cover successful requests; failures remain visible.
- Token hit rate is computed from aggregate counter deltas.
- Cache hits do not measure FLOPs saved or prove a latency cause.
- Five small runs do not establish production capacity or reliable p99.
- Output-token totals are shown to expose generation-work differences.

## Included files

- results/skewed-affinity-c4-016507764454869792122225657593808516039.json
- results/skewed-affinity-c4-298457557829945850783733640215018359539.json
- results/skewed-affinity-c4-303956992373199688228796798080747885633.json
- results/skewed-affinity-c4-099484067519706106914618147490659313296.json
- results/skewed-affinity-c4-291650013650160035104528718213523284236.json
- results/skewed-affinity_load-c4-184685656805685627731388262301552259387.json
- results/skewed-affinity_load-c4-067175125951421078567587549499919778481.json
- results/skewed-affinity_load-c4-093122192941831355317543099352820290454.json
- results/skewed-affinity_load-c4-245160973369921602774864940392549754956.json
- results/skewed-affinity_load-c4-300693851072895161784867874998084895415.json

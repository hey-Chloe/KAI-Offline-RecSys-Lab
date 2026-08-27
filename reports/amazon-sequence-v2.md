# Amazon Sequence V2 — Retriever-aligned pretrained DIN

Status: `COMPLETE` / `POSITIVE_DIN_TEST_IMPROVEMENT`; public offline only.

## Frozen protocol

- Candidate generator: metadata Two-Tower V2 Top-200; target never injected.
- Mean Pooling and DIN share candidates, negatives, pretrained item vectors, optimizer settings, and seeds.
- Three configurations selected using dev DIN NDCG@100; test opened once afterward.

## Test metrics — mean ± population std

| Model | NDCG@20 | NDCG@50 | NDCG@100 | MRR@100 |
|---|---:|---:|---:|---:|
| Mean Pooling | 0.019747 ± 0.000076 | 0.025453 ± 0.000136 | 0.029981 ± 0.000086 | 0.013748 ± 0.000064 |
| DIN | 0.020521 ± 0.000107 | 0.026356 ± 0.000083 | 0.030663 ± 0.000116 | 0.014211 ± 0.000105 |

DIN minus Mean Pooling NDCG@100: +0.000682 ± 0.000047.

## Limitations

- Amazon review/rating histories are public interaction proxies, not impression/click/order logs.
- The provider has not assigned a dataset license; raw data, checkpoints, and per-user outputs remain local ignored artifacts.
- The sequence models rerank a frozen metadata Two-Tower Top-200 and cannot recover targets outside that set.
- The frozen retriever seed is shared by all sequence runs; sequence uncertainty is measured over three training seeds only.
- Three CPU/MPS-sized configurations are selected using dev DIN NDCG@100; test is opened once and never tunes the model.
- A negative DIN result remains a valid outcome and will not be changed by rewriting the test or labels.
- Offline public metrics do not establish online CTR, conversion, revenue, or KAI production impact.

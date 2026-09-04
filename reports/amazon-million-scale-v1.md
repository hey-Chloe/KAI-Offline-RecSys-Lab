# Amazon Reviews'23 Million-Item HNSW Scalability V1

## Boundary

This is a **public-data, local, offline scalability benchmark**. It does not prove recommendation quality, online CTR/CVR, revenue impact, or production readiness.

## Catalog

- Real source rows: 1,253,672
- Unique real `parent_asin`: 1,253,672
- Cross-source duplicates excluded: 0
- Synthetic expansion / copied vectors: none
- Raw metadata, embeddings, and HNSW index: local ignored artifacts

## Exact vs HNSW

| K | ANN Recall@K | p50 (ms) | p95 (ms) | Sequential QPS |
|---:|---:|---:|---:|---:|
| 20 | 0.856055 | 0.3066 | 0.6330 | 2904.20 |
| 50 | 0.856563 | 0.3075 | 0.6072 | 3018.62 |
| 100 | 0.853867 | 0.3153 | 0.5961 | 2954.17 |

## Index and runtime

- HNSW index size: 483.65 MiB
- HNSW build time: 42.17 s
- Exact Top-100 time: 2.86 s for 256 queries
- Host: macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs
- Runtime: Python 3.12.13, NumPy 2.3.5, PyTorch 2.13.0, hnswlib 0.8.0

## Representation boundary

The frozen V2 checkpoint has trained item IDs only for the original 25,754-item catalog. Every scale-corpus product therefore uses the same UNK item ID; only its real title/category metadata changes the embedding. This avoids fabricating identity features and keeps the claim limited to systems scalability.

## License / use boundary

McAuley Lab publishes Amazon Reviews'23 primarily for research and states that it is not in a position to assign a dataset license. This experiment is local research-only. Raw rows, generated embeddings, checkpoints, and the index are not redistributed.

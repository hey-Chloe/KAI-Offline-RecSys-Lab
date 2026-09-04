# Offline RecSys Industrialization Extension V1

Status: `COMPLETE_LOCAL_PUBLIC_OFFLINE`

This extension closes the executable public-data and local-integration gaps
without changing the frozen V1 reports, synthetic benchmark, gold labels or
test sets. It does not convert offline evidence into a production claim.

## Results

### CTR scale

- Complete pinned official parquet shard: 766,864 rows, up from the frozen
  60,000-row V1 subset.
- Fixed train/dev/test counts: 536,804 / 115,029 / 115,031.
- Same 13 numeric and 26 categorical features, train-only preprocessing and
  seeds 3407 / 6502 / 9109 for LR, DeepFM and DCNv2.
- Best test ROC-AUC in this run: DeepFM `0.730783 ± 0.001799`.
- This is the complete pinned shard, not the full Criteo 1TB/24-day corpus.

### CVR / ESMM

- Eligible public source: Criteo Attribution, with impression, click and
  conversion labels.
- Fixed 600,000-row timestamp-ordered protocol: 420,000 / 90,000 / 90,000.
- ESMM versus independent baseline:
  - CTCVR ROC-AUC: `+0.015784`;
  - Post-click CVR ROC-AUC: `+0.021812`;
  - CTR ROC-AUC: `-0.001899`.
- The conversion gains and CTR regression are both retained. No clicked-only
  population is misrepresented as the full exposure population.

### Million-item ANN scalability

- 1,253,672 unique public Amazon `parent_asin` values; no copied-vector or
  synthetic catalog expansion.
- HNSW ANN Recall@20/50/100: `0.856055 / 0.856563 / 0.853867`.
- Sequential single-query p95 latency: `0.6330 / 0.6072 / 0.5961 ms`.
- Local CPU-only QPS: approximately `2,904–3,019`; index size 483.65 MiB.
- This proves local embedding/HNSW scalability only. The primary relevance
  benchmark remains the frozen 25,754-item catalog.

### DIN V2

- Frozen Metadata Two-Tower Top-200 candidate set; target never injected.
- Three configurations selected on dev only; test opened once after selection.
- Mean Pooling NDCG@100: `0.029981 ± 0.000086`.
- DIN NDCG@100: `0.030663 ± 0.000116`.
- Paired DIN improvement: `+0.000682 ± 0.000047`.
- The result is positive but small and remains bounded by retriever coverage.

### Local serving rehearsal

- Artifact SHA-256 validation and fail-closed loading.
- Versioned local profile/feature updates and stable A/B assignment.
- Linked, idempotent impression/click/conversion demo events.
- Request/error and p50/p95 monitoring plus missing-feature drift proxy.
- Artifact-validated model activation, manual rollback and threshold-triggered
  automatic rollback rehearsal.
- Real localhost smoke path passed: profile update → recommendation replay →
  event idempotency, while unavailable pCTR stays explicitly null.

## Verification

- Offline boundary audit: 7 registered public/synthetic sources.
- Full integrated test suite: 113 passed.
- Public-report validators: all executed reports passed.
- Two-Tower V2 frozen verifier: passed.
- CVR/ESMM data/source verifier: `DATA_VERIFIED`.
- Local playground verifier: passed.
- Git whitespace/error check: passed.

## Still not solved by offline code

The following require approved production infrastructure and real business
traffic. They are not claimed by this version:

- distributed online recommendation service or real-time feature store;
- live user-profile streams and production authentication/durable storage;
- real online A/B tests, exposure/click/order/fulfillment logs or causal lift;
- online CTR/CVR improvement, revenue improvement or marketplace conversion;
- population-level production drift detection, deployment orchestration or
  automatic production rollback;
- a real recommendation → selection → order → fulfillment → settlement loop,
  which requires an approved production system and real traffic.

The local serving harness defines and tests these integration boundaries. It is
not evidence that the production dependencies exist.

# KAI Offline RecSys Lab

[![Offline RecSys Lab CI](https://github.com/hey-Chloe/KAI-Offline-RecSys-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hey-Chloe/KAI-Offline-RecSys-Lab/actions/workflows/ci.yml)

**[Open the public interactive Playground](https://hey-chloe.github.io/KAI-Offline-RecSys-Lab/)**

Independent public-data research for end-to-end retrieval/ranking, CTR,
post-click conversion, position-bias evaluation and value-aware ranking.

Canonical personal repository: `hey-Chloe/KAI-Offline-RecSys-Lab`. Its focused
Lab history and later public-scale extensions are documented in `PROVENANCE.md`.

## Truth boundary

- This is a standalone personal research project. It contains no company
  backend, production application, business dataset or marketplace code.
- Only `public` and explicitly labeled `synthetic` test data are accepted.
- Synthetic fixtures validate code paths only. They are never reported as public
  benchmark or production results.
- Raw third-party datasets and trained artifacts are ignored by Git. Acquisition
  requires a captured dataset-specific usage record in `sources/source-ledger.json`.
- No offline result is evidence of online CTR, conversion lift, revenue lift or
  production performance.

## Workstreams

1. Retrieval: popularity / ItemKNN / matrix-factorization / Two-Tower / HNSW.
2. Unified ranking: exact/HNSW candidates / hard, uniform and in-batch
   negatives / DIN-style and DCN-style rerankers / calibration / evaluation.
3. CTR and conversion: LR / DeepFM / DCNv2 / post-click CVR / ESMM.
4. Sequence: mean pooling / target-aware DIN-style attention.
5. Debiasing and ads: position baseline / IPS / SNIPS / value-aware ranking.

Every comparison uses one model-independent cohort, feature set, split and seed
protocol. Split semantics stay dataset-specific: Amazon uses the provider's
leave-last-out split, Criteo uses fixed source-order blocks without claiming
chronology, and Open Bandit uses a fixed timestamp cutoff.

## Executed public protocols

- Amazon Reviews'23 `Industrial_and_Scientific`: full 25,754-item train
  catalog, 50,653 common test users, Popularity / ItemKNN / BPR MF /
  Two-Tower exact retrieval and HNSW ANN; Mean Pooling / DIN reranking over a
  frozen ItemKNN Top-100 candidate set.
- Amazon end-to-end V3: one frozen protocol from the existing Two-Tower
  checkpoint through exact/HNSW Top-100, three negative-sampling strategies,
  DIN/DCN-style reranking, train-only calibration, user-level bootstrap,
  metadata/ID/title/category ablation and cold-start/popularity cohorts.
- Criteo 1TB Click Logs: a pinned, non-label-sampled 60,000-row prefix from one
  official parquet shard, LR / DeepFM / DCNv2 with ranking and calibration
  metrics.
- Open Bandit Dataset: full-archive final-window OPE plus the official small
  sample for reward-model and per-position calibration diagnostics.
- Criteo Attribution: a fixed 200,000-row impression population with click and
  conversion labels, naive clicked-only CVR and ESMM, three seeds, train-only
  preprocessing and a single frozen test opening.
- Amazon Reviews'23 million-scale extension: 1,253,672 unique public item IDs
  from `Industrial_and_Scientific` and `Amazon_Fashion`, evaluated as a local
  CPU-only HNSW scalability protocol. This is a systems measurement, not a
  recommendation-quality or production-SLO claim.
- Criteo CTR scale extension: the complete pinned 766,864-row official parquet
  shard, with the original LR / DeepFM / DCNv2 feature and seed parity.
- Criteo Attribution CVR/ESMM: 600,000 fixed, time-ordered impression rows from
  a source that contains both click and conversion labels. Naive post-click CVR
  is compared with ESMM without treating clicked-only data as the exposure
  population.

The tracked JSON reports include source terms, raw file hashes, config/split
hashes, exact row counts, seeds, metrics and limitations. Raw data and model
indexes remain ignored. The current extension is summarized in
`reports/industrial-recsys-extension-v1.md`.

## Local verification

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
make test
make smoke
make reproduce-small
make verify-public
make verify-playground
```

`make reproduce-small` runs the unified retrieval/ranking, CVR/ESMM and
evaluation code paths on a deterministic synthetic fixture in one command.
It proves reproducibility of the small code path, not the reported performance.
`make smoke` exercises every model family with an explicitly synthetic fixture
and writes an ignored `artifacts/synthetic-smoke.json`. It is an integration
check, not a benchmark result. `make verify-public` verifies provenance and
reproducibility fields in executed public reports; it does not rerun training.

## Local serving rehearsal

Run `make serve-local`, then in another terminal run `make serve-smoke`. The
service listens on `http://127.0.0.1:4280` and provides an artifact-verified,
local-only rehearsal of profile/feature updates, stable A/B assignment,
idempotent impression/click/conversion events, monitoring, model activation,
rollback and threshold-triggered automatic rollback.

This harness is intentionally not a production recommendation platform. It has
no external API, distributed feature store, production auth, real traffic or
business-data write. The current public CTR artifacts do not contain a
deployable row-level checkpoint, so the API returns `ctrScore: null` rather than
inventing a prediction.

## Local recruiting playground

Run `make playground`, then open `http://127.0.0.1:4190/playground/`. The
static viewer has no login, API key or external network dependency and reads
the tracked public result reports at runtime. Its first evidence panel exposes
the unified V3 comparison, feature ablation, HNSW trade-off, bootstrap interval
and cold-start support; CVR/ESMM is shown as a separate impression-level ads
task rather than falsely joined to the Amazon funnel. See `playground/README.md`.

GitHub Actions verifies the test suite, one-command reproduction, public
reports and portable page export. The standalone Pages workflow publishes only
the self-contained Playground after its truth-boundary verifier passes.
The public deployment above was anonymously verified over HTTPS after the
standalone repository handoff.

The primary Amazon recommendation-quality benchmark remains the frozen 25,754
item catalog. The separate 1,253,672-item extension supports a measured
million-item embedding/HNSW scalability claim only. It does not turn the frozen
quality result into a million-item relevance experiment.

# Local Recommendation Serving V1

Status: `LOCAL_INTEGRATION_COMPLETE`

Boundary: `PUBLIC OFFLINE DATA · LOCAL DEMO · NO PRODUCTION CLAIM`

This module is a local integration and operability harness around existing
public/offline artifacts. It is not deployed production infrastructure, does
not use private business data, and does not write to any production event store.

## What is implemented

- HTTP recommendation path: profile lookup → artifact-backed retrieval replay
  → deterministic retrieval-order ranking → explicit bias-correction boundary.
- Artifact SHA-256 verification at load time. A missing or changed required
  artifact prevents the active model from starting or being activated.
- Versioned local user profile/feature updates in an ignored artifact directory.
- Stable SHA-256 A/B assignment based on experiment ID and user ID.
- Idempotent `impression`, `click`, and `conversion` event logging. Events must
  reference an item actually emitted for the same recommendation request/user.
- Local model registry activation and rollback with artifact validation.
- Runtime request/error counts, p50/p95 latency, and a feature-missing-rate drift
  proxy. This is local observability, not population or production drift proof.
- A local automatic rollback gate based on recommendation-route 5xx rate or
  p95 latency. Client 4xx errors are excluded, and the previous artifact is
  verified before rollback. This is a rollback rehearsal, not production deployment.

## Truthful scoring boundary

The frozen artifacts do not contain a deployable row-level CTR checkpoint or
persisted per-example CTR probabilities. The local endpoint therefore returns
`ctrScore: null`; it does not manufacture a pCTR. Ranking preserves the frozen
retrieval order and exposes a deterministic reciprocal-rank integration score.

IPS and SNIPS remain offline policy-value estimators. They are not applied as
per-item inference scores. The endpoint records this as
`biasCorrection.applied: false` so the interface exists without misusing the
offline position-bias experiment.

## Start and smoke test

From the experiment directory:

```bash
.venv/bin/python scripts/serve_local.py --config configs/serving-local.json
```

The service listens on `http://127.0.0.1:4280`.

In a second terminal:

```bash
.venv/bin/python scripts/serve_smoke.py --url http://127.0.0.1:4280
```

## APIs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Boundary and active-version health |
| `PUT` | `/v1/profiles/:userId` | Update local demo profile/features |
| `POST` | `/v1/recommendations` | Artifact-backed local recommendation |
| `POST` | `/v1/events` | Linked, idempotent demo event |
| `GET` | `/v1/models` | Registry state and model boundaries |
| `POST` | `/v1/models/activate` | Validate and activate local version |
| `POST` | `/v1/models/rollback` | Validate and restore previous version |
| `GET` | `/v1/monitoring` | Computed local latency/error/drift signals |

## Verification performed

- `offline boundary verified: 7 registered sources`
- Serving-specific automated tests: `10 passed`
- Full integrated test suite at final integration: `113 passed`
- Real localhost HTTP acceptance:
  - profile update: success;
  - recommendation: HTTP 200 with three replayed candidates;
  - CTR scores: all null as required by the frozen artifact boundary;
  - stable A/B assignment returned;
  - first impression event: HTTP 201;
  - identical event replay: HTTP 200 and `created=false`;
  - local model activation and rollback: success;
  - monitoring returned computed route and event counters.

## Explicit limitations

- The local service is single-host JSON-file state, not a distributed serving
  stack and not safe to describe as an online production deployment.
- Recommendation responses are replayable only for public profiles present in
  the referenced frozen artifact. Unknown profiles fail closed.
- The Two-Tower trace is a frozen public-test replay, not live user embedding
  inference. Its large ignored artifact must exist locally and match its digest.
- There is no feature stream, feature store, ANN shard fleet, autoscaling,
  authentication, durable database, online A/B analysis, or automated rollout.
- Runtime event logs are explicitly local demo events and are never production
  impressions, clicks, conversions, orders, revenue, or online uplift evidence.

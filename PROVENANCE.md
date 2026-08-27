# Repository provenance

This is the user-owned standalone repository for KAI Offline RecSys Lab.

- Owner: `hey-Chloe`
- Source repository: `mandow123/zod`
- Source canonical commit: `ee2df28dc1f2c20bb4ca85b32e78145252eeb8d4`
- Source pull request: `mandow123/zod#16`
- Source product approval: `PM-20260827-001`
- Extraction method: `git subtree split --prefix=experiments/offline-recsys-lab`
- Initial subtree commit: `710f0b296943ffc2922cc27eb330fe7d1187deee`

The subtree split preserves the Lab-specific history while excluding Compute
Production, migration 0066, the frozen Compute benchmark, backend/mobile/admin
applications, payment code and business data.

The source monorepo remains historical provenance. This standalone repository
is the canonical home for future Offline RecSys Lab development.

Raw third-party datasets, model checkpoints, vector indexes, user/item traces,
credentials and local runtime artifacts are not redistributed. Dataset terms
and attribution are recorded in `sources/source-ledger.json` and
`playground/THIRD_PARTY_DATA.md`. No project-wide license is granted by this
repository; third-party data and dependencies retain their own terms.

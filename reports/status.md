# Offline RecSys Lab Status

| Area | Status | Evidence boundary |
|---|---|---|
| Standalone repository | READY | Lab-only subtree history; no Compute, 0066, production application or business data copied |
| Source ledger | READY | Amazon research-only/no assigned license; Criteo CC BY-NC-SA 4.0; Open Bandit CC BY 4.0 |
| Python environment | READY | Pinned macOS arm64 PyTorch/scikit-learn/HNSW environment |
| Retrieval/ANN/sequence code | READY | Popularity, ItemKNN, BPR, Two-Tower, exact/HNSW and target-aware attention |
| CTR/conversion code | READY | LR, DeepFM, DCNv2, naive PostClickCVR and ESMM |
| Debiasing/ads code | READY | IPS/SNIPS diagnostics and five deterministic ad objectives |
| Synthetic unit/smoke tests | PASSED | Unified code-path smoke; not performance evidence |
| Amazon public benchmark | COMPLETE | 310,977/50,985/50,985 official rows; 25,754-item catalog; 50,653 common test users |
| Amazon unified end-to-end V3 | COMPLETE | Two-Tower → exact/HNSW → three negative strategies → DIN/DCN → train-only calibration → frozen test |
| Experiment analysis V3 | COMPLETE | Five dev candidates, five input-zeroing ablations, 2,000-sample paired user bootstrap, cohorts and three-point HNSW sweep |
| Amazon sequence benchmark V1 | COMPLETE_NEGATIVE | DIN did not stably exceed Mean Pooling on the frozen ItemKNN Top-100 protocol |
| Amazon sequence benchmark V2 | COMPLETE_POSITIVE_SMALL | Dev-only selection, one test opening, three seeds; DIN minus Mean Pooling NDCG@100 = +0.000682 ± 0.000047 |
| Criteo CTR benchmark V1 | COMPLETE_FIXED_SUBSET | 60,000 fixed rows from one official shard; not the full 1TB corpus |
| Criteo CTR scale V2 | COMPLETE_FULL_PINNED_SHARD | 766,864 rows from the complete pinned official shard; still not the full 1TB/24-day corpus |
| CVR/ESMM public benchmark V1 | COMPLETE_OFFLINE | Official Criteo Attribution, fixed 200,000 impressions, same population, three seeds, test opened once |
| CVR/ESMM scale extension | COMPLETE_PUBLIC_OFFLINE | 600,000 fixed Criteo Attribution impressions; ESMM improved CTCVR/post-click CVR but not CTR |
| Million-item ANN scalability | COMPLETE_PUBLIC_OFFLINE | 1,253,672 unique public item IDs; CPU-only exact/HNSW scalability, not quality or production SLO evidence |
| Position-bias public benchmark | COMPLETE | Full archive OPE plus small-sample reward-model calibration; action propensity only |
| Local recommendation serving | COMPLETE_LOCAL_REHEARSAL | Artifact checks, local profiles/features, A/B, events, monitoring and rollback; no real traffic or production claims |
| Production integration | OUT_OF_SCOPE | No company backend, production application or business-data writes |
| Recruitment playground | COMPLETE_LOCAL_DEMO | Chinese-first, local-only, artifact-driven; browser-checked at desktop and mobile widths |
| Portable static export | READY | Reproducible self-contained directory/ZIP; generated output is ignored by Git |
| Public portfolio deployment | COMPLETE_PUBLIC_HTTP_200 | `https://hey-chloe.github.io/KAI-Offline-RecSys-Lab/`; standalone Pages workflow passed and anonymous HTTPS/browser verification succeeded |

Executed metrics are public offline measurements only. They are not evidence of
online lift, marketplace performance, conversion lift or revenue lift.

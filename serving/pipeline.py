from __future__ import annotations

import hashlib
import json
from threading import RLock
from typing import Any

from .artifacts import PlaygroundReplayRetriever, Retriever, TwoTowerTraceRetriever
from .config import ServingConfig, resolve_artifact
from .contracts import DATA_BOUNDARY, RecommendationRequest, UnavailableError, utc_now
from .registry import ModelRegistry
from .stores import LocalProfileStore, RequestReceiptStore


class RecommendationPipeline:
    def __init__(
        self,
        config: ServingConfig,
        profiles: LocalProfileStore,
        registry: ModelRegistry,
        receipts: RequestReceiptStore,
    ) -> None:
        self._config = config
        self._profiles = profiles
        self._registry = registry
        self._receipts = receipts
        self._retrievers: dict[str, Retriever] = {}
        self._lock = RLock()

    def ensure_loadable(self, model_version: str) -> None:
        self._retriever(model_version)

    def _retriever(self, model_version: str) -> Retriever:
        with self._lock:
            existing = self._retrievers.get(model_version)
            if existing is not None:
                return existing
            path, digest, spec = resolve_artifact(self._config, model_version)
            retriever_type = spec.get("retriever")
            if retriever_type == "playground_replay":
                retriever: Retriever = PlaygroundReplayRetriever(
                    path, digest, str(spec.get("algorithm"))
                )
            elif retriever_type == "two_tower_trace_replay":
                retriever = TwoTowerTraceRetriever(path, digest)
            else:
                raise UnavailableError(f"unsupported retriever adapter: {retriever_type}")
            self._retrievers[model_version] = retriever
            return retriever

    def recommend(self, request: RecommendationRequest) -> dict[str, Any]:
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "userId": request.user_id,
                    "limit": request.limit,
                    "modelVersion": request.model_version,
                    "experimentId": request.experiment_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        replay = self._receipts.get(request.request_id, fingerprint)
        if replay is not None:
            replay["idempotentReplay"] = True
            return replay
        profile = self._profiles.get(request.user_id)
        assignment: dict[str, str] | None = None
        if request.experiment_id:
            variant, model_version = self._registry.assign(request.experiment_id, request.user_id)
            assignment = {"experimentId": request.experiment_id, "variant": variant}
        else:
            model_version = request.model_version or self._registry.active
        spec = self._registry.spec(model_version)
        candidates = self._retriever(model_version).retrieve(profile["sourceProfileId"], request.limit)
        if not candidates:
            raise UnavailableError("retriever produced no candidates")
        if spec.get("ranking") != "preserve_retrieval_order":
            raise UnavailableError("configured ranking adapter is unsupported")
        results = []
        for candidate in candidates:
            deterministic_score = 1.0 / candidate.retrieval_rank
            results.append(
                {
                    "itemId": candidate.item_id,
                    "rank": candidate.retrieval_rank,
                    "retrievalRank": candidate.retrieval_rank,
                    "rankingScore": deterministic_score,
                    "ctrScore": None,
                    "biasAwareScore": deterministic_score,
                }
            )
        response: dict[str, Any] = {
            "schemaVersion": 1,
            "dataBoundary": DATA_BOUNDARY,
            "productionClaim": False,
            "requestId": request.request_id,
            "userId": request.user_id,
            "profileVersion": profile["version"],
            "modelVersion": model_version,
            "assignment": assignment,
            "generatedAt": utc_now(),
            "idempotentReplay": False,
            "stages": {
                "retrieval": {
                    "adapter": spec.get("retriever"),
                    "artifactVerified": True,
                },
                "ctrRanking": {
                    "adapter": "preserve_retrieval_order",
                    "ctrArtifactAvailable": bool(spec.get("ctrArtifactAvailable", False)),
                    "boundary": "No row-level CTR checkpoint exists; ctrScore is null and is never fabricated.",
                },
                "biasCorrection": {
                    "applied": False,
                    "adapter": "research_boundary_only",
                    "boundary": "IPS/SNIPS are offline estimators, not an online per-item score correction.",
                },
            },
            "items": results,
        }
        self._receipts.put(request.request_id, fingerprint, response)
        return response

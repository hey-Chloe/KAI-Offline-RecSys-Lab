from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, require_id


@dataclass(frozen=True, slots=True)
class ServingConfig:
    root: Path
    host: str
    port: int
    state_dir: Path
    assignment_salt: str
    active_model: str
    models: Mapping[str, Mapping[str, Any]]
    experiments: Mapping[str, Mapping[str, Any]]
    drift_feature_keys: tuple[str, ...]
    drift_missing_rate_threshold: float
    auto_rollback_enabled: bool
    auto_rollback_min_requests: int
    auto_rollback_error_rate_threshold: float
    auto_rollback_p95_latency_ms: float


def _resolve_inside(root: Path, value: object, field: str) -> Path:
    relative = Path(require_id(value, field))
    if relative.is_absolute():
        raise ContractError(f"{field} must be relative to the experiment root")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ContractError(f"{field} escapes the experiment root")
    return path


def load_config(path: str | Path) -> ServingConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("dataBoundary") != "public-offline-local-demo":
        raise ContractError("serving config must preserve the public/offline local-demo boundary")
    if raw.get("productionClaim") is not False or raw.get("productionFlywheel") is not False:
        raise ContractError("serving config must explicitly reject production claims and flywheel writes")
    experiment_root = Path(require_id(raw.get("experimentRoot", ".."), "experimentRoot"))
    if experiment_root.is_absolute():
        raise ContractError("experimentRoot must be relative to the config file")
    root = (config_path.parent / experiment_root).resolve()
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ContractError("models must be a non-empty object")
    active_model = require_id(raw.get("activeModel"), "activeModel")
    if active_model not in models:
        raise ContractError("activeModel is not registered")
    for version, spec in models.items():
        require_id(version, "model version")
        if not isinstance(spec, dict):
            raise ContractError(f"model {version} must be an object")
        artifact = spec.get("artifact")
        if not isinstance(artifact, dict):
            raise ContractError(f"model {version} lacks artifact metadata")
        _resolve_inside(root, artifact.get("path"), f"models.{version}.artifact.path")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ContractError(f"model {version} lacks a SHA-256 artifact digest")
        if spec.get("productionClaim") is not False:
            raise ContractError(f"model {version} must reject production claims")
    experiments = raw.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ContractError("experiments must be an object")
    for experiment_id, experiment in experiments.items():
        require_id(experiment_id, "experiment id")
        variants = experiment.get("variants") if isinstance(experiment, dict) else None
        if not isinstance(variants, list) or not variants:
            raise ContractError(f"experiment {experiment_id} has no variants")
        total = 0
        for variant in variants:
            if not isinstance(variant, dict) or variant.get("modelVersion") not in models:
                raise ContractError(f"experiment {experiment_id} references an unknown model")
            weight = variant.get("weight")
            if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
                raise ContractError(f"experiment {experiment_id} has an invalid weight")
            total += weight
        if total < 1:
            raise ContractError(f"experiment {experiment_id} has no assignment weight")
    monitoring = raw.get("monitoring", {})
    keys = monitoring.get("driftFeatureKeys", [])
    if not isinstance(keys, list):
        raise ContractError("monitoring.driftFeatureKeys must be a list")
    threshold = monitoring.get("missingRateThreshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ContractError("monitoring.missingRateThreshold must be between zero and one")
    auto_rollback = monitoring.get("autoRollback", {})
    if not isinstance(auto_rollback, dict):
        raise ContractError("monitoring.autoRollback must be an object")
    auto_enabled = auto_rollback.get("enabled", False)
    min_requests = auto_rollback.get("minRequests", 20)
    error_threshold = auto_rollback.get("errorRateThreshold", 0.2)
    p95_threshold = auto_rollback.get("p95LatencyMsThreshold", 500.0)
    if not isinstance(auto_enabled, bool):
        raise ContractError("monitoring.autoRollback.enabled must be boolean")
    if not isinstance(min_requests, int) or isinstance(min_requests, bool) or min_requests < 1:
        raise ContractError("monitoring.autoRollback.minRequests must be positive")
    if not isinstance(error_threshold, (int, float)) or isinstance(error_threshold, bool) or not 0.0 <= float(error_threshold) <= 1.0:
        raise ContractError("monitoring.autoRollback.errorRateThreshold must be between zero and one")
    if not isinstance(p95_threshold, (int, float)) or isinstance(p95_threshold, bool) or float(p95_threshold) <= 0.0:
        raise ContractError("monitoring.autoRollback.p95LatencyMsThreshold must be positive")
    server = raw.get("server", {})
    port = server.get("port", 4280)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ContractError("server.port must be between zero and 65535")
    return ServingConfig(
        root=root,
        host=str(server.get("host", "127.0.0.1")),
        port=port,
        state_dir=_resolve_inside(root, raw.get("stateDir", "artifacts/serving-local"), "stateDir"),
        assignment_salt=require_id(raw.get("assignmentSalt"), "assignmentSalt"),
        active_model=active_model,
        models=models,
        experiments=experiments,
        drift_feature_keys=tuple(require_id(key, "drift feature key") for key in keys),
        drift_missing_rate_threshold=float(threshold),
        auto_rollback_enabled=auto_enabled,
        auto_rollback_min_requests=min_requests,
        auto_rollback_error_rate_threshold=float(error_threshold),
        auto_rollback_p95_latency_ms=float(p95_threshold),
    )


def resolve_artifact(config: ServingConfig, model_version: str) -> tuple[Path, str, Mapping[str, Any]]:
    try:
        spec = config.models[model_version]
    except KeyError as error:
        raise ContractError(f"unknown model version: {model_version}") from error
    artifact = spec["artifact"]
    return _resolve_inside(config.root, artifact["path"], "artifact.path"), artifact["sha256"], spec

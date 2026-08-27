from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from .config import ServingConfig
from .contracts import ContractError, DATA_BOUNDARY, ConflictError, utc_now
from .stores import _atomic_json


class ModelRegistry:
    def __init__(self, config: ServingConfig, state_path: Path) -> None:
        self._config = config
        self._state_path = state_path
        self._lock = RLock()
        self._active = config.active_model
        self._history: list[str] = []
        if state_path.is_file():
            document = json.loads(state_path.read_text(encoding="utf-8"))
            if document.get("dataBoundary") != DATA_BOUNDARY:
                raise RuntimeError("local model registry has an unexpected data boundary")
            active = document.get("activeModel")
            history = document.get("history", [])
            if active not in config.models or not isinstance(history, list):
                raise RuntimeError("local model registry state is invalid")
            if any(version not in config.models for version in history):
                raise RuntimeError("local model registry history references an unknown model")
            self._active = active
            self._history = list(history)

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    def spec(self, version: str) -> Mapping[str, Any]:
        try:
            return self._config.models[version]
        except KeyError as error:
            raise ContractError(f"unknown model version: {version}") from error

    def activate(self, version: str) -> str:
        self.spec(version)
        with self._lock:
            if version == self._active:
                return self._active
            self._history.append(self._active)
            self._active = version
            self._persist()
            return self._active

    def rollback(self) -> str:
        with self._lock:
            if not self._history:
                raise ConflictError("no previous local model version is available for rollback")
            self._active = self._history.pop()
            self._persist()
            return self._active

    def rollback_candidate(self) -> str:
        with self._lock:
            if not self._history:
                raise ConflictError("no previous local model version is available for rollback")
            return self._history[-1]

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "dataBoundary": DATA_BOUNDARY,
                "productionClaim": False,
                "activeModel": self._active,
                "rollbackAvailable": bool(self._history),
                "models": [
                    {
                        "version": version,
                        "active": version == self._active,
                        "retriever": spec.get("retriever"),
                        "ranking": spec.get("ranking"),
                        "ctrArtifactAvailable": bool(spec.get("ctrArtifactAvailable", False)),
                    }
                    for version, spec in self._config.models.items()
                ],
            }

    def assign(self, experiment_id: str, user_id: str) -> tuple[str, str]:
        try:
            experiment = self._config.experiments[experiment_id]
        except KeyError as error:
            raise ContractError(f"unknown experiment: {experiment_id}") from error
        variants = experiment["variants"]
        total = sum(int(variant["weight"]) for variant in variants)
        digest = hashlib.sha256(
            f"{self._config.assignment_salt}\0{experiment_id}\0{user_id}".encode()
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % total
        cursor = 0
        for variant in variants:
            cursor += int(variant["weight"])
            if bucket < cursor:
                return str(variant["name"]), str(variant["modelVersion"])
        raise AssertionError("weighted assignment did not resolve")

    def _persist(self) -> None:
        _atomic_json(
            self._state_path,
            {
                "schemaVersion": 1,
                "dataBoundary": DATA_BOUNDARY,
                "productionClaim": False,
                "activeModel": self._active,
                "history": self._history,
                "updatedAt": utc_now(),
            },
        )

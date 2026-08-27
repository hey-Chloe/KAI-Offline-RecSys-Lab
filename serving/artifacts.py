from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol

from .contracts import NotFoundError, UnavailableError


def verify_artifact(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise UnavailableError(f"required artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise UnavailableError(f"artifact digest mismatch: {path}")


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    item_id: str
    retrieval_rank: int


class Retriever(Protocol):
    def retrieve(self, source_profile_id: str, limit: int) -> list[RetrievedItem]: ...


class PlaygroundReplayRetriever:
    def __init__(self, path: Path, expected_sha256: str, algorithm: str) -> None:
        verify_artifact(path, expected_sha256)
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("dataBoundary") != "public-offline-demo-fixture":
            raise UnavailableError("playground replay artifact has an unexpected data boundary")
        if document.get("productionClaim") is not False:
            raise UnavailableError("playground replay artifact permits a production claim")
        if algorithm not in {"popularity", "itemKnn"}:
            raise UnavailableError(f"unsupported replay algorithm: {algorithm}")
        self._algorithm = algorithm
        self._profiles: dict[str, Mapping[str, Any]] = {}
        for history in document.get("amazon", {}).get("histories", []):
            self._profiles[str(history["id"])] = history
            self._profiles[str(history["userAlias"])] = history

    def retrieve(self, source_profile_id: str, limit: int) -> list[RetrievedItem]:
        try:
            profile = self._profiles[source_profile_id]
        except KeyError as error:
            raise NotFoundError(f"source profile is not available in the replay artifact: {source_profile_id}") from error
        values = profile.get("recommendations", {}).get(self._algorithm)
        if not isinstance(values, list) or not values:
            raise UnavailableError(f"{self._algorithm} replay is unavailable for {source_profile_id}")
        return [RetrievedItem(str(item), rank) for rank, item in enumerate(values[:limit], start=1)]


class TwoTowerTraceRetriever:
    """Read-only replay over a frozen user-level Two-Tower Top-100 trace."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        verify_artifact(path, expected_sha256)
        self._path = path
        self._cache: dict[str, tuple[str, ...]] = {}
        self._lock = RLock()

    def retrieve(self, source_profile_id: str, limit: int) -> list[RetrievedItem]:
        with self._lock:
            cached = self._cache.get(source_profile_id)
        if cached is None:
            found: tuple[str, ...] | None = None
            with gzip.open(self._path, "rt", encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    if record.get("userAlias") == source_profile_id:
                        values = record.get("top100ItemAliases")
                        if not isinstance(values, list) or not values:
                            raise UnavailableError("Two-Tower trace row has no candidate list")
                        found = tuple(str(item) for item in values)
                        break
            if found is None:
                raise NotFoundError(f"source profile is not available in the Two-Tower trace: {source_profile_id}")
            with self._lock:
                self._cache[source_profile_id] = found
            cached = found
        return [RetrievedItem(item, rank) for rank, item in enumerate(cached[:limit], start=1)]

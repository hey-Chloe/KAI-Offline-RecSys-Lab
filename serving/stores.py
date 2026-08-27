from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from .contracts import ConflictError, DATA_BOUNDARY, DemoEvent, NotFoundError, ProfileUpdate, utc_now


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class LocalProfileStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()
        self._profiles: dict[str, dict[str, Any]] = {}
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("dataBoundary") != DATA_BOUNDARY:
                raise RuntimeError("local profile store has an unexpected data boundary")
            self._profiles = dict(document.get("profiles", {}))

    def put(self, update: ProfileUpdate) -> dict[str, Any]:
        with self._lock:
            previous = self._profiles.get(update.user_id)
            version = int(previous.get("version", 0)) + 1 if previous else 1
            value = {
                "userId": update.user_id,
                "sourceProfileId": update.source_profile_id,
                "history": list(update.history),
                "features": dict(update.features),
                "version": version,
                "updatedAt": utc_now(),
            }
            self._profiles[update.user_id] = value
            _atomic_json(
                self._path,
                {
                    "schemaVersion": 1,
                    "dataBoundary": DATA_BOUNDARY,
                    "productionFlywheel": False,
                    "profiles": self._profiles,
                },
            )
            return dict(value)

    def get(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._profiles[user_id])
            except KeyError as error:
                raise NotFoundError(f"local demo profile not found: {user_id}") from error


class IdempotentEventLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()
        self._events: dict[str, dict[str, Any]] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    if record.get("dataBoundary") != DATA_BOUNDARY:
                        raise RuntimeError("local event log has an unexpected data boundary")
                    self._events[record["eventId"]] = record

    def append(self, event: DemoEvent) -> tuple[dict[str, Any], bool]:
        record = event.as_record()
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            existing = self._events.get(event.event_id)
            if existing is not None:
                if json.dumps(existing, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != canonical:
                    raise ConflictError("eventId was already used with a different payload")
                return dict(existing), False
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as target:
                target.write(canonical + "\n")
                target.flush()
                os.fsync(target.fileno())
            self._events[event.event_id] = record
            return dict(record), True

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


class RequestReceiptStore:
    """Small idempotency store for recommendation responses in the local harness."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._receipts: dict[str, tuple[str, dict[str, Any]]] = {}

    def get(self, request_id: str, fingerprint: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self._receipts.get(request_id)
            if existing is None:
                return None
            if existing[0] != fingerprint:
                raise ConflictError("requestId was already used with a different payload")
            return dict(existing[1])

    def put(self, request_id: str, fingerprint: str, response: Mapping[str, Any]) -> None:
        with self._lock:
            self._receipts[request_id] = (fingerprint, dict(response))

    def validates_item(self, request_id: str, user_id: str, item_id: str) -> bool:
        with self._lock:
            existing = self._receipts.get(request_id)
            if existing is None:
                return False
            response = existing[1]
            if response.get("userId") != user_id:
                return False
            return any(item.get("itemId") == item_id for item in response.get("items", []))

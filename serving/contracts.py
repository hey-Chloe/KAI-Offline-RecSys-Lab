from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


DATA_BOUNDARY = "public-offline-local-demo"
ALLOWED_EVENT_TYPES = frozenset({"impression", "click", "conversion"})


class ContractError(ValueError):
    """Raised when a local serving request violates its public-demo contract."""


class NotFoundError(LookupError):
    pass


class ConflictError(RuntimeError):
    pass


class UnavailableError(RuntimeError):
    pass


def require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ContractError(f"{field} must be a non-empty string of at most 160 characters")
    return value.strip()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ProfileUpdate:
    user_id: str
    source_profile_id: str
    history: tuple[str, ...]
    features: Mapping[str, str | int | float | bool | None]

    @classmethod
    def parse(cls, user_id: str, payload: object) -> "ProfileUpdate":
        user_id = require_id(user_id, "userId")
        if not isinstance(payload, dict):
            raise ContractError("profile body must be a JSON object")
        source_profile_id = require_id(payload.get("sourceProfileId"), "sourceProfileId")
        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list) or len(raw_history) > 200:
            raise ContractError("history must be a list with at most 200 items")
        history = tuple(require_id(item, "history item") for item in raw_history)
        raw_features = payload.get("features", {})
        if not isinstance(raw_features, dict) or len(raw_features) > 100:
            raise ContractError("features must be an object with at most 100 keys")
        features: dict[str, str | int | float | bool | None] = {}
        for key, value in raw_features.items():
            name = require_id(key, "feature name")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ContractError(f"feature {name} must be a JSON scalar")
            features[name] = value
        return cls(user_id, source_profile_id, history, features)


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    request_id: str
    user_id: str
    limit: int
    model_version: str | None
    experiment_id: str | None

    @classmethod
    def parse(cls, payload: object) -> "RecommendationRequest":
        if not isinstance(payload, dict):
            raise ContractError("recommendation body must be a JSON object")
        request_id = require_id(payload.get("requestId"), "requestId")
        user_id = require_id(payload.get("userId"), "userId")
        limit = payload.get("limit", 10)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ContractError("limit must be an integer between 1 and 100")
        model_version = payload.get("modelVersion")
        if model_version is not None:
            model_version = require_id(model_version, "modelVersion")
        experiment_id = payload.get("experimentId")
        if experiment_id is not None:
            experiment_id = require_id(experiment_id, "experimentId")
        if model_version and experiment_id:
            raise ContractError("modelVersion and experimentId are mutually exclusive")
        return cls(request_id, user_id, limit, model_version, experiment_id)


@dataclass(frozen=True, slots=True)
class DemoEvent:
    event_id: str
    event_type: str
    user_id: str
    request_id: str
    item_id: str
    occurred_at: str
    metadata: Mapping[str, Any]

    @classmethod
    def parse(cls, payload: object) -> "DemoEvent":
        if not isinstance(payload, dict):
            raise ContractError("event body must be a JSON object")
        event_type = require_id(payload.get("eventType"), "eventType")
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ContractError(f"eventType must be one of {sorted(ALLOWED_EVENT_TYPES)}")
        occurred_at = payload.get("occurredAt", utc_now())
        occurred_at = require_id(occurred_at, "occurredAt")
        try:
            datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ContractError("occurredAt must be ISO-8601") from error
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ContractError("metadata must be a JSON object")
        forbidden = {"paymentCredential", "accessSecret", "callbackRaw"}.intersection(metadata)
        if forbidden:
            raise ContractError(f"sensitive metadata is forbidden: {sorted(forbidden)}")
        return cls(
            event_id=require_id(payload.get("eventId"), "eventId"),
            event_type=event_type,
            user_id=require_id(payload.get("userId"), "userId"),
            request_id=require_id(payload.get("requestId"), "requestId"),
            item_id=require_id(payload.get("itemId"), "itemId"),
            occurred_at=occurred_at,
            metadata=metadata,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "dataBoundary": DATA_BOUNDARY,
            "productionFlywheel": False,
            "eventId": self.event_id,
            "eventType": self.event_type,
            "userId": self.user_id,
            "requestId": self.request_id,
            "itemId": self.item_id,
            "occurredAt": self.occurred_at,
            "metadata": dict(self.metadata),
        }

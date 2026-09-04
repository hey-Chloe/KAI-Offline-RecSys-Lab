from __future__ import annotations

import math
from collections import defaultdict, deque
from threading import RLock
from typing import Any, Mapping

from .contracts import DATA_BOUNDARY


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


class LocalMonitor:
    def __init__(self, feature_keys: tuple[str, ...], missing_rate_threshold: float) -> None:
        self._feature_keys = feature_keys
        self._threshold = missing_rate_threshold
        self._lock = RLock()
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2000))
        self._requests: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._profile_count = 0
        self._feature_missing: dict[str, int] = defaultdict(int)

    def record_request(self, route: str, latency_ms: float, is_error: bool) -> None:
        with self._lock:
            self._requests[route] += 1
            self._latencies[route].append(latency_ms)
            if is_error:
                self._errors[route] += 1

    def observe_profile(self, features: Mapping[str, object]) -> None:
        with self._lock:
            self._profile_count += 1
            for key in self._feature_keys:
                if key not in features or features[key] is None:
                    self._feature_missing[key] += 1

    def snapshot(self, *, event_count: int) -> dict[str, Any]:
        with self._lock:
            routes: dict[str, Any] = {}
            for route, count in sorted(self._requests.items()):
                values = list(self._latencies[route])
                errors = self._errors[route]
                routes[route] = {
                    "requests": count,
                    "errors": errors,
                    "errorRate": errors / count if count else 0.0,
                    "latencyMs": {
                        "p50": _percentile(values, 0.50),
                        "p95": _percentile(values, 0.95),
                        "windowSize": len(values),
                    },
                }
            feature_drift: dict[str, Any] = {}
            for key in self._feature_keys:
                missing = self._feature_missing[key]
                rate = missing / self._profile_count if self._profile_count else None
                feature_drift[key] = {
                    "observations": self._profile_count,
                    "missing": missing,
                    "missingRate": rate,
                    "alert": rate is not None and rate > self._threshold,
                    "signal": "missing-rate-only local proxy; not population drift",
                }
            return {
                "dataBoundary": DATA_BOUNDARY,
                "productionMonitoring": False,
                "routes": routes,
                "events": event_count,
                "featureDrift": feature_drift,
            }

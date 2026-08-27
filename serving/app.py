from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import unquote, urlparse

from .config import ServingConfig
from .contracts import (
    ConflictError,
    ContractError,
    DATA_BOUNDARY,
    DemoEvent,
    NotFoundError,
    ProfileUpdate,
    RecommendationRequest,
    UnavailableError,
    require_id,
)
from .monitoring import LocalMonitor
from .pipeline import RecommendationPipeline
from .registry import ModelRegistry
from .stores import IdempotentEventLog, LocalProfileStore, RequestReceiptStore


class LocalServingApp:
    def __init__(self, config: ServingConfig) -> None:
        config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.profiles = LocalProfileStore(config.state_dir / "profiles.json")
        self.events = IdempotentEventLog(config.state_dir / "events.jsonl")
        self.registry = ModelRegistry(config, config.state_dir / "registry.json")
        self.monitor = LocalMonitor(config.drift_feature_keys, config.drift_missing_rate_threshold)
        self.receipts = RequestReceiptStore()
        self.pipeline = RecommendationPipeline(
            config,
            self.profiles,
            self.registry,
            self.receipts,
        )
        # Fail closed at boot if the active artifact is missing or changed.
        self.pipeline.ensure_loadable(self.registry.active)
        self._rollback_lock = RLock()
        self._automatic_rollbacks: list[dict[str, Any]] = []

    def observe_result(self, route: str, *, status: int, latency_ms: float) -> None:
        """Record one local request and exercise the configured rollback gate.

        Only server failures count toward rollback. Client contract errors are
        excluded so invalid demo input cannot roll a model back.
        """

        self.monitor.record_request(route, latency_ms, status >= 500)
        if not self.config.auto_rollback_enabled or route != "POST /v1/recommendations":
            return
        snapshot = self.monitor.snapshot(event_count=self.events.count)
        route_metrics = snapshot["routes"].get(route)
        if not route_metrics or route_metrics["requests"] < self.config.auto_rollback_min_requests:
            return
        error_trigger = route_metrics["errorRate"] >= self.config.auto_rollback_error_rate_threshold
        p95 = route_metrics["latencyMs"]["p95"]
        latency_trigger = p95 is not None and p95 >= self.config.auto_rollback_p95_latency_ms
        if not (error_trigger or latency_trigger):
            return
        with self._rollback_lock:
            state = self.registry.describe()
            if not state["rollbackAvailable"]:
                return
            previous = self.registry.active
            candidate = self.registry.rollback_candidate()
            self.pipeline.ensure_loadable(candidate)
            active = self.registry.rollback()
            self._automatic_rollbacks.append({
                "fromModel": previous,
                "toModel": active,
                "reason": "server_error_rate" if error_trigger else "p95_latency",
                "requests": route_metrics["requests"],
                "errorRate": route_metrics["errorRate"],
                "p95LatencyMs": p95,
                "productionDeployment": False,
            })

    def route(self, method: str, raw_path: str, payload: object | None) -> tuple[int, dict[str, Any]]:
        path = urlparse(raw_path).path
        if method == "GET" and path == "/healthz":
            return HTTPStatus.OK, {
                "status": "ok",
                "dataBoundary": DATA_BOUNDARY,
                "productionClaim": False,
                "activeModel": self.registry.active,
            }
        if method == "GET" and path == "/v1/models":
            return HTTPStatus.OK, self.registry.describe()
        if method == "GET" and path == "/v1/monitoring":
            response = self.monitor.snapshot(event_count=self.events.count)
            response["autoRollback"] = {
                "enabled": self.config.auto_rollback_enabled,
                "productionDeployment": False,
                "events": list(self._automatic_rollbacks),
            }
            return HTTPStatus.OK, response
        if method == "PUT" and path.startswith("/v1/profiles/"):
            user_id = unquote(path.removeprefix("/v1/profiles/"))
            update = ProfileUpdate.parse(user_id, payload)
            profile = self.profiles.put(update)
            self.monitor.observe_profile(profile["features"])
            return HTTPStatus.OK, {
                "dataBoundary": DATA_BOUNDARY,
                "productionFlywheel": False,
                "profile": profile,
            }
        if method == "POST" and path == "/v1/recommendations":
            return HTTPStatus.OK, self.pipeline.recommend(RecommendationRequest.parse(payload))
        if method == "POST" and path == "/v1/events":
            event = DemoEvent.parse(payload)
            if not self.receipts.validates_item(
                event.request_id, event.user_id, event.item_id
            ):
                raise ContractError("event does not reference an item emitted for this request and user")
            record, created = self.events.append(event)
            return (HTTPStatus.CREATED if created else HTTPStatus.OK), {
                "created": created,
                "event": record,
            }
        if method == "POST" and path == "/v1/models/activate":
            if not isinstance(payload, dict):
                raise ContractError("activation body must be a JSON object")
            version = require_id(payload.get("modelVersion"), "modelVersion")
            self.pipeline.ensure_loadable(version)
            return HTTPStatus.OK, {
                "dataBoundary": DATA_BOUNDARY,
                "activeModel": self.registry.activate(version),
                "productionDeployment": False,
            }
        if method == "POST" and path == "/v1/models/rollback":
            self.pipeline.ensure_loadable(self.registry.rollback_candidate())
            return HTTPStatus.OK, {
                "dataBoundary": DATA_BOUNDARY,
                "activeModel": self.registry.rollback(),
                "productionDeployment": False,
            }
        raise NotFoundError(f"route not found: {method} {path}")

    def make_server(self, host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "KAILocalRecServing/1"
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch("POST")

            def do_PUT(self) -> None:  # noqa: N802
                self._dispatch("PUT")

            def _dispatch(self, method: str) -> None:
                started = time.perf_counter()
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                response: dict[str, Any]
                try:
                    payload = self._read_json() if method in {"POST", "PUT"} else None
                    status, response = app.route(method, self.path, payload)
                except ContractError as error:
                    status, response = HTTPStatus.BAD_REQUEST, self._error("invalid_request", error)
                except NotFoundError as error:
                    status, response = HTTPStatus.NOT_FOUND, self._error("not_found", error)
                except ConflictError as error:
                    status, response = HTTPStatus.CONFLICT, self._error("conflict", error)
                except UnavailableError as error:
                    status, response = HTTPStatus.SERVICE_UNAVAILABLE, self._error("artifact_unavailable", error)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    status, response = HTTPStatus.BAD_REQUEST, self._error("invalid_json", error)
                except Exception:
                    status, response = HTTPStatus.INTERNAL_SERVER_ERROR, {
                        "error": "internal_error",
                        "message": "local serving request failed",
                        "dataBoundary": DATA_BOUNDARY,
                    }
                body = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                elapsed = (time.perf_counter() - started) * 1000.0
                app.observe_result(
                    f"{method} {urlparse(self.path).path}",
                    status=int(status),
                    latency_ms=elapsed,
                )

            def _read_json(self) -> object:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise ContractError("Content-Length is required")
                try:
                    length = int(raw_length)
                except ValueError as error:
                    raise ContractError("Content-Length must be an integer") from error
                if not 0 < length <= 1024 * 1024:
                    raise ContractError("request body must be between 1 byte and 1 MiB")
                return json.loads(self.rfile.read(length).decode("utf-8"))

            @staticmethod
            def _error(code: str, error: Exception) -> dict[str, Any]:
                return {"error": code, "message": str(error), "dataBoundary": DATA_BOUNDARY}

            def log_message(self, format: str, *args: object) -> None:
                # Keep local demo output concise and avoid logging request bodies.
                return

        return ThreadingHTTPServer((host or self.config.host, self.config.port if port is None else port), Handler)


def serve(config: ServingConfig) -> None:
    app = LocalServingApp(config)
    server = app.make_server()
    host, port = server.server_address[:2]
    print(f"local recommendation service: http://{host}:{port}", flush=True)
    print("boundary: PUBLIC OFFLINE DATA · LOCAL DEMO · NO PRODUCTION CLAIM", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

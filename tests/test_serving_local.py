from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serving.app import LocalServingApp
from serving.config import load_config
from serving.contracts import ConflictError, ContractError, UnavailableError


def _write_test_project(root: Path, *, invalid_digest: bool = False) -> Path:
    fixture = {
        "schemaVersion": 1,
        "dataBoundary": "public-offline-demo-fixture",
        "productionClaim": False,
        "amazon": {
            "histories": [
                {
                    "id": "public-history-1",
                    "userAlias": "PUBLIC-USER-1",
                    "recommendations": {
                        "popularity": ["POPULAR-1", "POPULAR-2", "POPULAR-3"],
                        "itemKnn": ["KNN-1", "KNN-2", "KNN-3"],
                    },
                }
            ]
        },
    }
    artifact = root / "fixture.json"
    artifact.write_text(json.dumps(fixture), encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if invalid_digest:
        digest = "0" * 64
    config_dir = root / "configs"
    config_dir.mkdir()
    config = {
        "schemaVersion": 1,
        "dataBoundary": "public-offline-local-demo",
        "productionClaim": False,
        "productionFlywheel": False,
        "experimentRoot": "..",
        "stateDir": "runtime",
        "assignmentSalt": "unit-test-stable-salt",
        "activeModel": "itemknn-v1",
        "server": {"host": "127.0.0.1", "port": 0},
        "models": {
            "itemknn-v1": {
                "productionClaim": False,
                "retriever": "playground_replay",
                "algorithm": "itemKnn",
                "ranking": "preserve_retrieval_order",
                "ctrArtifactAvailable": False,
                "artifact": {"path": "fixture.json", "sha256": digest},
            },
            "popularity-v1": {
                "productionClaim": False,
                "retriever": "playground_replay",
                "algorithm": "popularity",
                "ranking": "preserve_retrieval_order",
                "ctrArtifactAvailable": False,
                "artifact": {"path": "fixture.json", "sha256": digest},
            },
        },
        "experiments": {
            "retrieval-ab": {
                "variants": [
                    {"name": "knn", "modelVersion": "itemknn-v1", "weight": 1},
                    {"name": "popular", "modelVersion": "popularity-v1", "weight": 1},
                ]
            }
        },
        "monitoring": {
            "driftFeatureKeys": ["region", "device"],
            "missingRateThreshold": 0.5,
            "autoRollback": {
                "enabled": True,
                "minRequests": 2,
                "errorRateThreshold": 0.5,
                "p95LatencyMsThreshold": 500.0,
            },
        },
    }
    path = config_dir / "serving.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class LocalServingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.app = LocalServingApp(load_config(_write_test_project(self.root)))
        self.app.route(
            "PUT",
            "/v1/profiles/local-user",
            {
                "sourceProfileId": "public-history-1",
                "history": ["HISTORY-1"],
                "features": {"region": "public-demo", "device": "test"},
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _recommend(self, request_id: str = "request-1") -> dict[str, object]:
        _, response = self.app.route(
            "POST",
            "/v1/recommendations",
            {"requestId": request_id, "userId": "local-user", "limit": 3},
        )
        return response

    def test_recommendation_preserves_truthful_scoring_boundary(self) -> None:
        response = self._recommend()
        self.assertEqual([item["itemId"] for item in response["items"]], ["KNN-1", "KNN-2", "KNN-3"])
        self.assertTrue(all(item["ctrScore"] is None for item in response["items"]))
        self.assertFalse(response["stages"]["biasCorrection"]["applied"])
        self.assertFalse(response["productionClaim"])

    def test_recommendation_request_is_idempotent_and_rejects_conflict(self) -> None:
        first = self._recommend()
        second = self._recommend()
        self.assertFalse(first["idempotentReplay"])
        self.assertTrue(second["idempotentReplay"])
        with self.assertRaises(ConflictError):
            self.app.route(
                "POST",
                "/v1/recommendations",
                {"requestId": "request-1", "userId": "local-user", "limit": 2},
            )

    def test_profile_updates_are_versioned_and_feed_drift_signal(self) -> None:
        _, response = self.app.route(
            "PUT",
            "/v1/profiles/local-user",
            {"sourceProfileId": "public-history-1", "history": [], "features": {"region": "x"}},
        )
        self.assertEqual(response["profile"]["version"], 2)
        _, monitoring = self.app.route("GET", "/v1/monitoring", None)
        self.assertEqual(monitoring["featureDrift"]["device"]["missing"], 1)
        self.assertFalse(monitoring["productionMonitoring"])

    def test_ab_assignment_is_stable(self) -> None:
        assignments = {self.app.registry.assign("retrieval-ab", "local-user") for _ in range(20)}
        self.assertEqual(len(assignments), 1)
        _, response = self.app.route(
            "POST",
            "/v1/recommendations",
            {
                "requestId": "ab-request",
                "userId": "local-user",
                "limit": 2,
                "experimentId": "retrieval-ab",
            },
        )
        self.assertIn(response["assignment"]["variant"], {"knn", "popular"})

    def test_event_log_is_linked_and_idempotent(self) -> None:
        recommendation = self._recommend()
        payload = {
            "eventId": "event-1",
            "eventType": "impression",
            "userId": "local-user",
            "requestId": "request-1",
            "itemId": recommendation["items"][0]["itemId"],
            "occurredAt": "2026-08-27T10:00:00+08:00",
        }
        status, first = self.app.route("POST", "/v1/events", payload)
        repeated_status, repeated = self.app.route("POST", "/v1/events", payload)
        self.assertEqual(status, 201)
        self.assertEqual(repeated_status, 200)
        self.assertTrue(first["created"])
        self.assertFalse(repeated["created"])
        invalid = dict(payload, eventId="event-2", itemId="NOT-RECOMMENDED")
        with self.assertRaises(ContractError):
            self.app.route("POST", "/v1/events", invalid)

    def test_activation_and_rollback(self) -> None:
        _, activated = self.app.route(
            "POST", "/v1/models/activate", {"modelVersion": "popularity-v1"}
        )
        self.assertEqual(activated["activeModel"], "popularity-v1")
        _, rolled_back = self.app.route("POST", "/v1/models/rollback", {})
        self.assertEqual(rolled_back["activeModel"], "itemknn-v1")

    def test_server_failures_trigger_validated_automatic_rollback(self) -> None:
        self.app.route("POST", "/v1/models/activate", {"modelVersion": "popularity-v1"})
        self.app.observe_result("POST /v1/recommendations", status=503, latency_ms=10.0)
        self.assertEqual(self.app.registry.active, "popularity-v1")
        self.app.observe_result("POST /v1/recommendations", status=503, latency_ms=12.0)
        self.assertEqual(self.app.registry.active, "itemknn-v1")
        _, monitoring = self.app.route("GET", "/v1/monitoring", None)
        self.assertEqual(monitoring["autoRollback"]["events"][0]["reason"], "server_error_rate")

    def test_client_errors_do_not_trigger_automatic_rollback(self) -> None:
        self.app.route("POST", "/v1/models/activate", {"modelVersion": "popularity-v1"})
        for _ in range(3):
            self.app.observe_result("POST /v1/recommendations", status=400, latency_ms=10.0)
        self.assertEqual(self.app.registry.active, "popularity-v1")


class LocalServingFailClosedTest(unittest.TestCase):
    def test_digest_mismatch_prevents_boot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_test_project(root, invalid_digest=True))
            with self.assertRaises(UnavailableError):
                LocalServingApp(config)

    def test_http_server_runs_real_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = LocalServingApp(load_config(_write_test_project(Path(directory))))
            server = app.make_server(port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base}/healthz", timeout=5) as response:
                    body = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertFalse(body["productionClaim"])
                request = urllib.request.Request(
                    f"{base}/v1/recommendations",
                    data=json.dumps(
                        {"requestId": "missing-profile", "userId": "unknown", "limit": 2}
                    ).encode(),
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

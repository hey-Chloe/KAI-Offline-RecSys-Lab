from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def request(url: str, method: str = "GET", payload: object | None = None) -> tuple[int, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    call = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        call.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(call, timeout=20) as response:
        return response.status, json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a running local recommendation service")
    parser.add_argument("--url", default="http://127.0.0.1:4280")
    arguments = parser.parse_args()
    base = arguments.url.rstrip("/")
    status, health = request(f"{base}/healthz")
    if status != 200 or health.get("productionClaim") is not False:
        raise SystemExit("health boundary failed")
    request(
        f"{base}/v1/profiles/smoke-user",
        "PUT",
        {
            "sourceProfileId": "public-history-1",
            "history": ["ITEM-A1EB9DF5"],
            "features": {"region": "public-demo", "device": "local"},
        },
    )
    status, recommendation = request(
        f"{base}/v1/recommendations",
        "POST",
        {"requestId": "smoke-recommendation-1", "userId": "smoke-user", "limit": 5},
    )
    if status != 200 or len(recommendation.get("items", [])) != 5:
        raise SystemExit("recommendation smoke failed")
    if any(item.get("ctrScore") is not None for item in recommendation["items"]):
        raise SystemExit("service fabricated a CTR score")
    print("local serving smoke passed: profile -> retrieval replay -> explicit no-CTR boundary")


if __name__ == "__main__":
    main()

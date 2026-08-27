#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kai_recsys_lab.public_report import (  # noqa: E402
    canonical_sha256,
    load_and_validate_public_report,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the frozen CVR/ESMM V1 evidence")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "criteo-attribution-cvr-esmm-v1-results.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "criteo-attribution-cvr-esmm-v1.json",
    )
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()

    report = load_and_validate_public_report(args.report)
    recorded_result_hash = report.pop("resultSha256")
    if canonical_sha256(report) != recorded_result_hash:
        raise SystemExit("resultSha256 mismatch")
    if sha256_file(args.config) != report["evidence"]["configSha256"]:
        raise SystemExit("configSha256 mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_path = PROJECT_ROOT / config["dataset"]["rawFile"]
    expected = report["evidence"]["datasetFiles"][0]
    if raw_path.is_file():
        if raw_path.stat().st_size != expected["bytes"]:
            raise SystemExit("raw dataset size mismatch")
        if sha256_file(raw_path) != expected["sha256"]:
            raise SystemExit("raw dataset SHA-256 mismatch")
        data_status = "DATA_VERIFIED"
    elif args.require_data:
        raise SystemExit(f"required raw dataset is missing: {raw_path}")
    else:
        data_status = "DATA_NOT_PRESENT_REPORT_ONLY"
    print(f"valid CVR/ESMM V1: {data_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

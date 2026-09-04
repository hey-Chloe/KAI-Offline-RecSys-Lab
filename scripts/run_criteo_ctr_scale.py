#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kai_recsys_lab.pipelines.criteo_ctr_scale import render_markdown, run_criteo_ctr_scale, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete pinned-shard Criteo CTR scale protocol")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "criteo-ctr-scale-v2.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "criteo-ctr-scale-v2-results.json")
    parser.add_argument("--markdown-output", type=Path, default=PROJECT_ROOT / "reports" / "criteo-ctr-scale-v2.md")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_criteo_ctr_scale(config, config_path=args.config)
    write_report(report, args.output)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

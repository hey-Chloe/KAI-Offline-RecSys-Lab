#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kai_recsys_lab.pipelines.amazon_sequence_v2 import render_markdown, run_dev_selection, run_final_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gated retriever-aligned Amazon DIN V2 experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/amazon-sequence-v2.json"))
    parser.add_argument("--phase", choices=("dev-select", "test-final"), required=True)
    parser.add_argument("--markdown", type=Path, default=Path("reports/amazon-sequence-v2.md"))
    args = parser.parse_args()
    if args.phase == "dev-select":
        result = run_dev_selection(args.config)
        print(f"selected={result['selectedCandidateId']} testMetrics={result['testMetrics']}")
        return
    report = run_final_test(args.config)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"status={report['status']} outcome={report['outcome']}")


if __name__ == "__main__":
    main()

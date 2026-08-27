from __future__ import annotations

import argparse
from pathlib import Path

from kai_recsys_lab.pipelines.amazon_million_scale import render_markdown, run_million_scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent public million-item embedding/HNSW scalability benchmark"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/amazon-million-scale-v1.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("reports/amazon-million-scale-v1.md"),
    )
    args = parser.parse_args()
    report = run_million_scale(args.config)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    counts = report["protocol"]["counts"]
    print(
        f"status={report['status']} uniqueItems={counts['uniqueCatalogItems']} "
        f"queries={counts['queryUsers']} claim=retrieval_scalability_only"
    )


if __name__ == "__main__":
    main()

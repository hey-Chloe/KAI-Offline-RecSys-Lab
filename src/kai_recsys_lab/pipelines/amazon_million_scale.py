from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kai_recsys_lab.pipelines.amazon_retrieval import prepare_amazon_data
from kai_recsys_lab.pipelines.amazon_two_tower_v2 import (
    build_metadata_catalog,
    load_checkpoint,
    load_v2_config,
    metadata_two_tower_vectors,
)
from kai_recsys_lab.retrieval.scale_million import (
    benchmark_hnsw,
    build_hnsw,
    encode_unique_catalog_to_memmap,
    exact_chunked_topk,
    runtime_evidence,
    scan_public_catalog,
    sha256_file,
)


def load_million_scale_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("dataOrigin") != "public" or config.get("claimableOnlinePerformance") is not False:
        raise ValueError("million-scale evaluation must remain public and offline")
    if config.get("purpose") != "retrieval_scalability_only":
        raise ValueError("million-scale work cannot claim recommendation quality")
    sources = config.get("catalogSources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("at least two audited real public source files are required")
    for source in sources:
        if source.get("origin") != "public" or source.get("synthetic") is not False:
            raise ValueError("catalog sources must be real public records, never synthetic expansion")
        expected = source.get("expectedFile")
        if not isinstance(expected, Mapping) or int(expected.get("bytes", 0)) < 1:
            raise ValueError("each source requires an expected byte boundary")
        digest = str(expected.get("sha256") or "")
        if len(digest) != 64:
            raise ValueError("each source requires an expected SHA-256 boundary")
    protocol = config.get("protocol", {})
    if protocol.get("ks") != [20, 50, 100]:
        raise ValueError("K protocol must remain [20, 50, 100]")
    if int(protocol.get("minimumUniqueItems", 0)) < 1_000_000:
        raise ValueError("scalability gate must require at least one million unique real items")
    if int(protocol.get("queryCount", 0)) < 100:
        raise ValueError("at least 100 deterministic real user queries are required")
    return config


def _source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for source in config["catalogSources"]:
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(f"audited public metadata is missing: {path}")
        expected = source["expectedFile"]
        actual_size = path.stat().st_size
        if actual_size != int(expected["bytes"]):
            raise ValueError(f"source byte boundary mismatch for {source['id']}: {actual_size}")
        actual_digest = sha256_file(path)
        if actual_digest != expected["sha256"]:
            raise ValueError(f"source SHA-256 boundary mismatch for {source['id']}")
        output.append((str(source["id"]), path))
    return output


def _config_evidence(config_path: Path, config: Mapping[str, Any]) -> Mapping[str, str]:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "fileSha256": sha256_file(config_path),
        "canonicalSha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def run_million_scale(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_million_scale_config(config_path)
    sources = _source_paths(config)
    scan = scan_public_catalog(sources)
    minimum = int(config["protocol"]["minimumUniqueItems"])
    if scan.unique_items < minimum:
        raise RuntimeError(
            f"real-item gate failed: found {scan.unique_items:,}, require at least {minimum:,}; "
            "synthetic expansion and duplicate vectors are forbidden"
        )

    v2_config_path = Path(config["frozenModel"]["config"])
    v2_config = load_v2_config(v2_config_path)
    raw_root = Path(v2_config["dataset"]["rawDir"])
    frozen_data = prepare_amazon_data(raw_root, v2_config)
    metadata_path = raw_root / v2_config["metadata"]["file"]
    frozen_metadata = build_metadata_catalog(
        metadata_path,
        frozen_data.catalog,
        max_vocabulary=int(v2_config["metadata"]["maxVocabulary"]),
        max_title_tokens=int(v2_config["metadata"]["maxTitleTokens"]),
        max_category_tokens=int(v2_config["metadata"]["maxCategoryTokens"]),
    )
    checkpoint_path = Path(config["frozenModel"]["checkpoint"])
    if sha256_file(checkpoint_path) != config["frozenModel"]["expectedSha256"]:
        raise ValueError("frozen Two-Tower checkpoint SHA-256 boundary mismatch")
    model, checkpoint = load_checkpoint(checkpoint_path, frozen_data, frozen_metadata)

    artifact_root = Path(config["artifactsDir"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    vectors_path = artifact_root / "amazon-public-million-item-vectors.npy"
    vector_started = time.perf_counter()
    vector_evidence = encode_unique_catalog_to_memmap(
        sources,
        model=model,
        vocabulary=frozen_metadata.vocabulary,
        scan=scan,
        destination=vectors_path,
        max_title_tokens=int(v2_config["metadata"]["maxTitleTokens"]),
        max_category_tokens=int(v2_config["metadata"]["maxCategoryTokens"]),
        batch_size=int(config["protocol"]["embeddingBatchSize"]),
    )
    vector_evidence = {**vector_evidence, "encodingSeconds": time.perf_counter() - vector_started}
    item_vectors = np.load(vectors_path, mmap_mode="r")

    all_user_vectors, _, _ = metadata_two_tower_vectors(
        model,
        frozen_data,
        checkpoint["candidateConfig"],
        split="test",
    )
    query_count = int(config["protocol"]["queryCount"])
    seed = int(config["protocol"]["seed"])
    rng = np.random.default_rng(seed)
    query_rows = np.sort(rng.choice(len(all_user_vectors), size=query_count, replace=False))
    queries = all_user_vectors[query_rows].astype(np.float32, copy=False)
    query_digest = hashlib.sha256(query_rows.astype("<i8").tobytes()).hexdigest()

    exact_started = time.perf_counter()
    exact_top100 = exact_chunked_topk(
        queries,
        item_vectors,
        k=100,
        item_chunk_size=int(config["protocol"]["exactItemChunkSize"]),
    )
    exact_seconds = time.perf_counter() - exact_started

    index_path = artifact_root / "amazon-public-million-hnsw.bin"
    index, index_evidence = build_hnsw(
        item_vectors,
        index_path,
        m=int(config["hnsw"]["m"]),
        ef_construction=int(config["hnsw"]["efConstruction"]),
        ef_search=int(config["hnsw"]["efSearch"]),
        threads=int(config["hnsw"]["buildThreads"]),
        seed=seed,
    )
    ann_results = benchmark_hnsw(
        index,
        queries,
        exact_top100,
        ks=config["protocol"]["ks"],
        warmup_queries=int(config["protocol"]["warmupQueries"]),
    )

    dataset_files = []
    for source in config["catalogSources"]:
        expected = source["expectedFile"]
        dataset_files.append(
            {
                "name": Path(source["path"]).name,
                "bytes": int(expected["bytes"]),
                "sha256": expected["sha256"],
                "sourceId": source["id"],
            }
        )
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": config["experimentId"],
        "status": "COMPLETE",
        "dataOrigin": "public",
        "claimableOnlinePerformance": False,
        "purpose": "retrieval_scalability_only",
        "source": {
            "id": "amazon-reviews-2023-multi-category-metadata",
            "officialUrl": "https://amazon-reviews-2023.github.io/main.html",
            "terms": "Research-only local use; provider does not assign a dataset license; raw rows, embeddings, and index are not redistributed.",
        },
        "evidence": {
            "datasetFiles": dataset_files,
            "configSha256": _config_evidence(config_path, config)["fileSha256"],
            "configCanonicalSha256": _config_evidence(config_path, config)["canonicalSha256"],
            "splitSha256": query_digest,
            "catalogOrderedSha256": scan.ordered_catalog_sha256,
            "frozenCheckpoint": {
                "name": checkpoint_path.name,
                "sha256": config["frozenModel"]["expectedSha256"],
                "seed": checkpoint["seed"],
                "candidateId": checkpoint["candidateConfig"]["id"],
            },
        },
        "protocol": {
            "seeds": [seed],
            "ks": config["protocol"]["ks"],
            "counts": {
                "sourceRows": scan.rows,
                "uniqueCatalogItems": scan.unique_items,
                "duplicateRowsExcluded": scan.duplicate_items,
                "malformedRows": scan.malformed_rows,
                "missingIdentifierRows": scan.missing_identifiers,
                "queryUsers": query_count,
            },
            "sameRealUserQueriesForExactAndAnn": True,
            "sameFullCatalogForExactAndAnn": True,
            "syntheticExpansion": False,
            "vectorDuplicationForScale": False,
            "querySelection": "deterministic sample from frozen Industrial_and_Scientific test users",
            "coldStartItemIdentity": "shared UNK item ID; only title/category metadata varies",
        },
        "catalogAudit": {
            "rows": scan.rows,
            "uniqueItems": scan.unique_items,
            "duplicateItemsExcluded": scan.duplicate_items,
            "malformedRows": scan.malformed_rows,
            "missingIdentifiers": scan.missing_identifiers,
            "titlePresent": scan.title_present,
            "categoryPresent": scan.category_present,
            "perSource": scan.per_source,
        },
        "results": {
            "exact": {
                "topK": 100,
                "queryCount": query_count,
                "totalSeconds": exact_seconds,
                "meanAmortizedLatencyMs": exact_seconds * 1000.0 / query_count,
            },
            "ann": ann_results,
            "itemEmbeddings": vector_evidence,
            "hnswIndex": index_evidence,
        },
        "runtime": runtime_evidence(),
        "limitations": [
            "This experiment proves local CPU embedding-index scalability only; it does not measure recommendation relevance or online impact.",
            "Amazon Reviews'23 metadata has no provider-assigned dataset license; use remains local, research-only, and non-redistributive.",
            "The frozen model was trained on Industrial_and_Scientific interactions; cross-category products use a shared UNK identity and metadata features only.",
            "Queries are frozen public review/rating-proxy users, not production impressions, clicks, purchases, or KAI business data.",
            "Sequential single-query latency on this host is not a production service SLO and excludes network/service overhead.",
        ],
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["protocol"]["counts"]
    ann = report["results"]["ann"]
    index = report["results"]["hnswIndex"]
    lines = [
        "# Amazon Reviews'23 Million-Item HNSW Scalability V1",
        "",
        "## Boundary",
        "",
        "This is a **public-data, local, offline scalability benchmark**. It does not prove recommendation quality, online CTR/CVR, revenue impact, or production readiness.",
        "",
        "## Catalog",
        "",
        f"- Real source rows: {counts['sourceRows']:,}",
        f"- Unique real `parent_asin`: {counts['uniqueCatalogItems']:,}",
        f"- Cross-source duplicates excluded: {counts['duplicateRowsExcluded']:,}",
        "- Synthetic expansion / copied vectors: none",
        "- Raw metadata, embeddings, and HNSW index: local ignored artifacts",
        "",
        "## Exact vs HNSW",
        "",
        "| K | ANN Recall@K | p50 (ms) | p95 (ms) | Sequential QPS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (20, 50, 100):
        row = ann[str(k)]
        lines.append(
            f"| {k} | {row['annRecallAtK']:.6f} | {row['p50LatencyMs']:.4f} | "
            f"{row['p95LatencyMs']:.4f} | {row['qpsSequential']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Index and runtime",
            "",
            f"- HNSW index size: {index['bytes'] / (1024 ** 2):.2f} MiB",
            f"- HNSW build time: {index['buildSeconds']:.2f} s",
            f"- Exact Top-100 time: {report['results']['exact']['totalSeconds']:.2f} s for {counts['queryUsers']} queries",
            f"- Host: {report['runtime']['platform']}; {report['runtime']['logicalCpuCount']} logical CPUs",
            f"- Runtime: Python {report['runtime']['python']}, NumPy {report['runtime']['numpy']}, PyTorch {report['runtime']['torch']}, hnswlib {report['runtime']['hnswlib']}",
            "",
            "## Representation boundary",
            "",
            "The frozen V2 checkpoint has trained item IDs only for the original 25,754-item catalog. Every scale-corpus product therefore uses the same UNK item ID; only its real title/category metadata changes the embedding. This avoids fabricating identity features and keeps the claim limited to systems scalability.",
            "",
            "## License / use boundary",
            "",
            "McAuley Lab publishes Amazon Reviews'23 primarily for research and states that it is not in a position to assign a dataset license. This experiment is local research-only. Raw rows, generated embeddings, checkpoints, and the index are not redistributed.",
            "",
        ]
    )
    return "\n".join(lines)

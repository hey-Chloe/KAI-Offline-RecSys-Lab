from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from kai_recsys_lab.pipelines.amazon_million_scale import load_million_scale_config
from kai_recsys_lab.retrieval.metadata_two_tower import MetadataTwoTower, MetadataTwoTowerConfig
from kai_recsys_lab.retrieval.scale_million import (
    benchmark_hnsw,
    build_hnsw,
    encode_cold_start_metadata_batch,
    encode_unique_catalog_to_memmap,
    exact_chunked_topk,
    scan_public_catalog,
)


def _metadata(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _model() -> MetadataTwoTower:
    config = MetadataTwoTowerConfig(
        num_users=4,
        num_items=5,
        metadata_vocabulary_size=16,
        embedding_dim=4,
        hidden_dim=8,
        output_dim=4,
    )
    return MetadataTwoTower(
        config,
        title_token_ids=torch.zeros((5, 3), dtype=torch.long),
        category_token_ids=torch.zeros((5, 2), dtype=torch.long),
    ).eval()


def test_catalog_scan_counts_real_unique_ids_without_expansion(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl.gz"
    right = tmp_path / "right.jsonl.gz"
    _metadata(
        left,
        [
            {"parent_asin": "a", "title": "red tool", "main_category": "Tools"},
            {"parent_asin": "b", "title": "blue tool", "main_category": "Tools"},
        ],
    )
    _metadata(
        right,
        [
            {"parent_asin": "b", "title": "duplicate", "main_category": "Fashion"},
            {"parent_asin": "c", "title": "cotton shirt", "main_category": "Fashion"},
        ],
    )
    scan = scan_public_catalog((("left", left), ("right", right)))

    assert scan.rows == 4
    assert scan.unique_items == 3
    assert scan.duplicate_items == 1
    assert scan.per_source["right"]["uniqueItemsAdded"] == 1
    assert len(scan.ordered_catalog_sha256) == 64


def test_unseen_products_share_identity_but_metadata_changes_vectors() -> None:
    model = _model()
    vectors = encode_cold_start_metadata_batch(
        model,
        title_token_ids=np.asarray([[2, 3, 0], [4, 5, 0]], dtype=np.int64),
        category_token_ids=np.asarray([[6, 0], [7, 0]], dtype=np.int64),
    )

    assert vectors.shape == (2, 4)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert not np.allclose(vectors[0], vectors[1])


def test_streaming_embedding_memmap_preserves_unique_count(tmp_path: Path) -> None:
    source = tmp_path / "metadata.jsonl.gz"
    _metadata(
        source,
        [
            {"parent_asin": "a", "title": "red tool", "main_category": "Tools"},
            {"parent_asin": "b", "title": "blue shirt", "main_category": "Fashion"},
            {"parent_asin": "a", "title": "duplicate", "main_category": "Tools"},
        ],
    )
    sources = (("fixture", source),)
    scan = scan_public_catalog(sources)
    destination = tmp_path / "vectors.npy"
    evidence = encode_unique_catalog_to_memmap(
        sources,
        model=_model(),
        vocabulary={"red": 2, "blue": 3, "tool": 4, "shirt": 5, "tools": 6, "fashion": 7},
        scan=scan,
        destination=destination,
        max_title_tokens=3,
        max_category_tokens=2,
        batch_size=1,
    )

    vectors = np.load(destination, mmap_mode="r")
    assert evidence["items"] == 2
    assert vectors.shape == (2, 4)
    assert evidence["allFinite"] is True


def test_chunked_exact_and_hnsw_use_same_full_catalog(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260827)
    items = rng.normal(size=(256, 8)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    queries = items[[3, 17, 41, 99]]
    exact = exact_chunked_topk(queries, items, k=20, item_chunk_size=64)
    index, evidence = build_hnsw(
        items,
        tmp_path / "fixture.bin",
        m=16,
        ef_construction=100,
        ef_search=200,
        threads=2,
        seed=20260827,
    )
    result = benchmark_hnsw(index, queries, exact, ks=(5, 10, 20), warmup_queries=1)

    assert evidence["bytes"] > 0
    assert result["20"]["annRecallAtK"] >= 0.95
    assert result["5"]["queryCount"] == 4
    assert result["10"]["p95LatencyMs"] >= 0


def test_config_rejects_sub_million_gate(tmp_path: Path) -> None:
    config = {
        "dataOrigin": "public",
        "claimableOnlinePerformance": False,
        "purpose": "retrieval_scalability_only",
        "catalogSources": [
            {
                "id": "a",
                "origin": "public",
                "synthetic": False,
                "expectedFile": {"bytes": 1, "sha256": "a" * 64},
            },
            {
                "id": "b",
                "origin": "public",
                "synthetic": False,
                "expectedFile": {"bytes": 1, "sha256": "b" * 64},
            },
        ],
        "protocol": {"ks": [20, 50, 100], "minimumUniqueItems": 999999, "queryCount": 100},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    try:
        load_million_scale_config(path)
    except ValueError as error:
        assert "one million" in str(error)
    else:
        raise AssertionError("sub-million gate must fail")

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import hnswlib
import numpy as np
import torch
from torch.nn import functional as F

from kai_recsys_lab.pipelines.amazon_two_tower_v2 import _tokens
from kai_recsys_lab.retrieval.metadata_two_tower import MetadataTwoTower


@dataclass(frozen=True, slots=True)
class PublicProduct:
    source_id: str
    parent_asin: str
    title_tokens: tuple[str, ...]
    category_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogScan:
    rows: int
    unique_items: int
    duplicate_items: int
    malformed_rows: int
    missing_identifiers: int
    title_present: int
    category_present: int
    ordered_catalog_sha256: str
    per_source: Mapping[str, Mapping[str, int]]


def iter_public_products(source_files: Sequence[tuple[str, Path]]) -> Iterator[PublicProduct]:
    """Stream real public metadata rows without retaining or rewriting raw text."""

    for source_id, path in source_files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                parent_asin = str(payload.get("parent_asin") or "").strip()
                categories = payload.get("categories")
                category_values = [str(payload.get("main_category") or "")]
                if isinstance(categories, list):
                    category_values.extend(str(value) for value in categories)
                yield PublicProduct(
                    source_id=source_id,
                    parent_asin=parent_asin,
                    title_tokens=tuple(_tokens(str(payload.get("title") or ""))),
                    category_tokens=tuple(_tokens(" ".join(category_values))),
                )


def scan_public_catalog(source_files: Sequence[tuple[str, Path]]) -> CatalogScan:
    """Count and fingerprint the unique real-item catalog in source order."""

    seen: set[str] = set()
    ordered_digest = hashlib.sha256()
    rows = duplicates = malformed = missing = title_present = category_present = 0
    per_source: dict[str, dict[str, int]] = {}
    for source_id, path in source_files:
        stats = {
            "rows": 0,
            "uniqueItemsAdded": 0,
            "duplicateItems": 0,
            "malformedRows": 0,
            "missingIdentifiers": 0,
        }
        per_source[source_id] = stats
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                rows += 1
                stats["rows"] += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    stats["malformedRows"] += 1
                    continue
                item_id = str(payload.get("parent_asin") or "").strip()
                if not item_id:
                    missing += 1
                    stats["missingIdentifiers"] += 1
                    continue
                if item_id in seen:
                    duplicates += 1
                    stats["duplicateItems"] += 1
                    continue
                seen.add(item_id)
                stats["uniqueItemsAdded"] += 1
                ordered_digest.update(source_id.encode("utf-8"))
                ordered_digest.update(b"\0")
                ordered_digest.update(item_id.encode("utf-8"))
                ordered_digest.update(b"\n")
                title_present += int(bool(str(payload.get("title") or "").strip()))
                category_present += int(
                    bool(str(payload.get("main_category") or "").strip() or payload.get("categories"))
                )
    return CatalogScan(
        rows=rows,
        unique_items=len(seen),
        duplicate_items=duplicates,
        malformed_rows=malformed,
        missing_identifiers=missing,
        title_present=title_present,
        category_present=category_present,
        ordered_catalog_sha256=ordered_digest.hexdigest(),
        per_source=per_source,
    )


def encode_cold_start_metadata_batch(
    model: MetadataTwoTower,
    *,
    title_token_ids: np.ndarray,
    category_token_ids: np.ndarray,
) -> np.ndarray:
    """Encode unseen products with the frozen item tower and shared UNK item ID.

    The checkpoint has learned ID embeddings only for its frozen 25,754-item
    training catalog. Using one shared UNK ID for every scale-corpus product
    prevents fabricated identity features; title/category metadata is the only
    item-varying input. This is a systems scalability experiment, not a quality
    evaluation for those cold-start items.
    """

    title = torch.as_tensor(title_token_ids, dtype=torch.long)
    category = torch.as_tensor(category_token_ids, dtype=torch.long)
    if title.ndim != 2 or category.ndim != 2 or title.shape[0] != category.shape[0]:
        raise ValueError("title and category batches must be aligned rank-two matrices")
    with torch.no_grad():
        unknown_ids = torch.ones(title.shape[0], dtype=torch.long)
        id_features = model.item_id_embedding(unknown_ids)
        title_features = model._mean_metadata(title)
        category_features = model._mean_metadata(category)
        vectors = F.normalize(
            model.item_mlp(torch.cat((id_features, title_features, category_features), dim=-1)),
            dim=-1,
        )
    return vectors.numpy().astype(np.float32, copy=False)


def encode_unique_catalog_to_memmap(
    source_files: Sequence[tuple[str, Path]],
    *,
    model: MetadataTwoTower,
    vocabulary: Mapping[str, int],
    scan: CatalogScan,
    destination: Path,
    max_title_tokens: int,
    max_category_tokens: int,
    batch_size: int,
) -> Mapping[str, Any]:
    if scan.unique_items < 1:
        raise ValueError("catalog must contain at least one unique item")
    if min(max_title_tokens, max_category_tokens, batch_size) < 1:
        raise ValueError("token widths and batch size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(scan.unique_items, model.config.output_dim),
    )
    seen: set[str] = set()
    title_rows: list[tuple[str, ...]] = []
    category_rows: list[tuple[str, ...]] = []
    written = 0

    def encoded_matrix(rows: Sequence[Sequence[str]], width: int) -> np.ndarray:
        output = np.zeros((len(rows), width), dtype=np.int64)
        for row_index, tokens in enumerate(rows):
            encoded = [vocabulary.get(token, 1) for token in tokens[:width]]
            output[row_index, : len(encoded)] = encoded
        return output

    def flush() -> None:
        nonlocal written
        if not title_rows:
            return
        batch = encode_cold_start_metadata_batch(
            model,
            title_token_ids=encoded_matrix(title_rows, max_title_tokens),
            category_token_ids=encoded_matrix(category_rows, max_category_tokens),
        )
        vectors[written : written + len(batch)] = batch
        written += len(batch)
        title_rows.clear()
        category_rows.clear()

    for product in iter_public_products(source_files):
        if not product.parent_asin or product.parent_asin in seen:
            continue
        seen.add(product.parent_asin)
        title_rows.append(product.title_tokens)
        category_rows.append(product.category_tokens)
        if len(title_rows) >= batch_size:
            flush()
    flush()
    if written != scan.unique_items:
        raise RuntimeError(f"encoded {written} items but scan found {scan.unique_items}")
    vectors.flush()
    return {
        "items": written,
        "dimension": model.config.output_dim,
        "dtype": "float32",
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "allFinite": bool(np.isfinite(vectors).all()),
        "unitNormMean": float(np.linalg.norm(vectors, axis=1).mean()),
    }


def exact_chunked_topk(
    query_vectors: np.ndarray,
    item_vectors: np.ndarray,
    *,
    k: int,
    item_chunk_size: int,
) -> np.ndarray:
    queries = np.asarray(query_vectors, dtype=np.float32)
    if queries.ndim != 2 or item_vectors.ndim != 2 or queries.shape[1] != item_vectors.shape[1]:
        raise ValueError("query and item vectors must be aligned rank-two matrices")
    if k < 1 or k > item_vectors.shape[0] or item_chunk_size < k:
        raise ValueError("invalid k or item chunk size")
    best_scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
    best_ids = np.full((queries.shape[0], k), -1, dtype=np.int64)
    for start in range(0, item_vectors.shape[0], item_chunk_size):
        block = np.asarray(item_vectors[start : start + item_chunk_size], dtype=np.float32)
        scores = queries @ block.T
        local_k = min(k, block.shape[0])
        local = np.argpartition(scores, -local_k, axis=1)[:, -local_k:]
        local_scores = np.take_along_axis(scores, local, axis=1)
        local_ids = local.astype(np.int64) + start
        merged_scores = np.concatenate((best_scores, local_scores), axis=1)
        merged_ids = np.concatenate((best_ids, local_ids), axis=1)
        keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        best_scores = np.take_along_axis(merged_scores, keep, axis=1)
        best_ids = np.take_along_axis(merged_ids, keep, axis=1)
    order = np.argsort(-best_scores, axis=1, kind="stable")
    return np.take_along_axis(best_ids, order, axis=1)


def build_hnsw(
    vectors: np.ndarray,
    destination: Path,
    *,
    m: int,
    ef_construction: int,
    ef_search: int,
    threads: int,
    seed: int,
) -> tuple[hnswlib.Index, Mapping[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    index = hnswlib.Index(space="ip", dim=vectors.shape[1])
    index.init_index(
        max_elements=vectors.shape[0],
        ef_construction=ef_construction,
        M=m,
        random_seed=seed,
    )
    started = time.perf_counter()
    index.add_items(vectors, np.arange(vectors.shape[0], dtype=np.int64), num_threads=threads)
    build_seconds = time.perf_counter() - started
    index.set_ef(ef_search)
    index.save_index(str(destination))
    return index, {
        "buildSeconds": build_seconds,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "m": m,
        "efConstruction": ef_construction,
        "efSearch": ef_search,
        "buildThreads": threads,
    }


def benchmark_hnsw(
    index: hnswlib.Index,
    query_vectors: np.ndarray,
    exact_top100: np.ndarray,
    *,
    ks: Sequence[int],
    warmup_queries: int,
) -> Mapping[str, Mapping[str, float | int]]:
    queries = np.asarray(query_vectors, dtype=np.float32)
    results: dict[str, Mapping[str, float | int]] = {}
    for k in ks:
        for row in range(min(warmup_queries, len(queries))):
            index.knn_query(queries[row : row + 1], k=k, num_threads=1)
        latencies: list[float] = []
        retrieved: list[np.ndarray] = []
        for query in queries:
            started = time.perf_counter_ns()
            labels, _ = index.knn_query(query.reshape(1, -1), k=k, num_threads=1)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            retrieved.append(labels[0])
        overlaps = [
            len(set(actual.tolist()) & set(expected[:k].tolist())) / k
            for actual, expected in zip(retrieved, exact_top100, strict=True)
        ]
        values = np.asarray(latencies, dtype=np.float64)
        results[str(k)] = {
            "annRecallAtK": float(np.mean(overlaps)),
            "p50LatencyMs": float(np.percentile(values, 50)),
            "p95LatencyMs": float(np.percentile(values, 95)),
            "meanLatencyMs": float(np.mean(values)),
            "qpsSequential": float(1000.0 / np.mean(values)),
            "queryCount": len(queries),
        }
    return results


def runtime_evidence() -> Mapping[str, Any]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        memory_bytes: int | None = page_size * physical_pages
    except (ValueError, OSError):
        memory_bytes = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Apple M-series reported by host",
        "logicalCpuCount": os.cpu_count(),
        "memoryBytes": memory_bytes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "hnswlib": importlib_metadata.version("hnswlib"),
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

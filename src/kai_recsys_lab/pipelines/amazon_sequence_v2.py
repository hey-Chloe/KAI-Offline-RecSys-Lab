from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..public_report import canonical_sha256, validate_public_report
from ..sequence.models import DinSequenceScorer, MeanPoolingSequenceScorer
from .amazon_retrieval import (
    PreparedAmazonData,
    _batched_dot_topk,
    _encode_history,
    _file_evidence,
    _history_matrix,
    _metric_summary,
    _paired_metric_delta,
    _sample_unseen_negatives,
    _seed_everything,
    prepare_amazon_data,
    ranking_metrics,
)
from .amazon_two_tower_v2 import (
    build_metadata_catalog,
    load_checkpoint,
    load_v2_config,
    metadata_two_tower_vectors,
)


SequenceModel = MeanPoolingSequenceScorer | DinSequenceScorer


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("dataOrigin") != "public" or config.get("claimableOnlinePerformance") is not False:
        raise ValueError("sequence V2 must remain public offline research")
    protocol = config["protocol"]
    if protocol.get("selectionSplit") != "dev" or protocol.get("selectionSeed") != 3407:
        raise ValueError("sequence V2 selection is frozen to dev seed 3407")
    if protocol.get("testSeeds") != [3407, 6502, 9109]:
        raise ValueError("sequence V2 final seeds must remain fixed")
    if protocol.get("ks") != [20, 50, 100] or int(protocol.get("candidateCount", 0)) < 100:
        raise ValueError("sequence V2 needs K 20/50/100 and at least 100 candidates")
    candidates = config.get("candidateConfigs", [])
    if len(candidates) != 3 or len({row.get("id") for row in candidates}) != 3:
        raise ValueError("sequence V2 must preregister exactly three dev candidates")
    return config


def _prepare(config: Mapping[str, Any]) -> tuple[PreparedAmazonData, Any, Any, Mapping[str, Any]]:
    retrieval_config_path = Path(config["retriever"]["config"])
    retrieval_config = load_v2_config(retrieval_config_path)
    data = prepare_amazon_data(retrieval_config["dataset"]["rawDir"], retrieval_config)
    metadata_path = Path(retrieval_config["dataset"]["rawDir"]) / retrieval_config["metadata"]["file"]
    metadata = build_metadata_catalog(
        metadata_path,
        data.catalog,
        max_vocabulary=int(retrieval_config["metadata"]["maxVocabulary"]),
        max_title_tokens=int(retrieval_config["metadata"]["maxTitleTokens"]),
        max_category_tokens=int(retrieval_config["metadata"]["maxCategoryTokens"]),
    )
    checkpoint_path = Path(config["retriever"]["checkpoint"])
    checkpoint_evidence = _file_evidence(checkpoint_path)
    if checkpoint_evidence["sha256"] != config["retriever"]["checkpointSha256"]:
        raise ValueError("frozen retriever checkpoint digest drifted")
    retriever, payload = load_checkpoint(checkpoint_path, data, metadata)
    if int(payload["seed"]) != int(config["retriever"]["seed"]):
        raise ValueError("frozen retriever seed drifted")
    return data, metadata, retriever, payload


def pretrained_item_embeddings(retriever: Any, item_count: int) -> np.ndarray:
    with torch.no_grad():
        vectors = retriever.encode_items(torch.arange(2, item_count + 2)).cpu().numpy()
    result = np.zeros((item_count + 2, vectors.shape[1]), dtype=np.float32)
    result[2:] = vectors
    return result


def initialize_item_embeddings(model: SequenceModel, values: np.ndarray, *, trainable: bool) -> None:
    if tuple(model.item_embedding.weight.shape) != tuple(values.shape):
        raise ValueError("pretrained item vectors do not match sequence embedding shape")
    with torch.no_grad():
        model.item_embedding.weight.copy_(torch.from_numpy(values))
        model.item_embedding.weight[0].zero_()
    model.item_embedding.weight.requires_grad_(trainable)


def _candidates(
    retriever: Any,
    data: PreparedAmazonData,
    payload: Mapping[str, Any],
    *,
    split: str,
    k: int,
) -> np.ndarray:
    users, items, exclusions = metadata_two_tower_vectors(
        retriever,
        data,
        payload["candidateConfig"],
        split=split,
    )
    return _batched_dot_topk(users, items, exclusions, k=k)


def _training_arrays(
    data: PreparedAmazonData,
    *,
    max_history: int,
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if negatives_per_positive < 1:
        raise ValueError("negativesPerPositive must be positive")
    users: list[int] = []
    positives: list[int] = []
    histories: list[np.ndarray] = []
    for row in data.train.itertuples(index=False):
        history = _encode_history(str(row.history), data.item_lookup)
        if history.size == 0:
            continue
        users.append(data.user_lookup[str(row.user_id)])
        positives.append(data.item_lookup[str(row.parent_asin)])
        histories.append(history)
    user_array = np.asarray(users, dtype=np.int64)
    rng = np.random.default_rng(seed)
    negatives = np.stack(
        [_sample_unseen_negatives(data, user_array, rng) for _ in range(negatives_per_positive)],
        axis=1,
    )
    return (
        np.asarray(positives, dtype=np.int64),
        negatives,
        _history_matrix(histories, max_history=max_history),
    )


def _train_model(
    model: SequenceModel,
    *,
    histories: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    candidate: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[SequenceModel, list[float]]:
    _seed_everything(seed)
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(candidate["learningRate"]))
    rng = np.random.default_rng(seed)
    batch_size = int(candidate["batchSize"])
    losses: list[float] = []
    for _ in range(int(candidate["epochs"])):
        total_loss = 0.0
        total_rows = 0
        for row_ids in np.array_split(rng.permutation(len(positives)), math.ceil(len(positives) / batch_size)):
            positive = positives[row_ids, None]
            items = np.concatenate((positive, negatives[row_ids]), axis=1)
            labels = np.zeros(items.shape, dtype=np.float32)
            labels[:, 0] = 1.0
            repeated_history = np.repeat(histories[row_ids], items.shape[1], axis=0)
            item_tensor = torch.from_numpy(items.reshape(-1) + 2).to(device)
            history_tensor = torch.from_numpy(repeated_history).to(device)
            label_tensor = torch.from_numpy(labels.reshape(-1)).to(device)
            logits = model(history_tensor, item_tensor).logits
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(label_tensor.numel())
            total_loss += float(loss.detach().cpu()) * count
            total_rows += count
        losses.append(total_loss / total_rows)
    return model.eval(), losses


def _rerank(
    model: SequenceModel,
    histories: Sequence[np.ndarray],
    candidates: np.ndarray,
    *,
    max_history: int,
    query_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    history_matrix = _history_matrix(histories, max_history=max_history)
    output = np.empty_like(candidates)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(histories), query_batch_size):
            end = min(offset + query_batch_size, len(histories))
            candidate_batch = candidates[offset:end]
            count = candidate_batch.shape[1]
            expanded_history = torch.from_numpy(history_matrix[offset:end]).repeat_interleave(count, dim=0).to(device)
            candidate_tensor = torch.from_numpy(candidate_batch.astype(np.int64).reshape(-1) + 2).to(device)
            scores = model(expanded_history, candidate_tensor).logits.reshape(end - offset, count).cpu().numpy()
            for row in range(end - offset):
                items = candidate_batch[row]
                output[offset + row] = items[np.lexsort((items, -scores[row]))]
    return output


def _make_model(
    model_name: str,
    *,
    item_count: int,
    candidate: Mapping[str, Any],
    pretrained: np.ndarray,
) -> SequenceModel:
    if model_name == "meanPooling":
        model: SequenceModel = MeanPoolingSequenceScorer(
            item_count + 2,
            int(candidate["embeddingDim"]),
            int(candidate["hiddenDim"]),
        )
    elif model_name == "din":
        model = DinSequenceScorer(
            item_count + 2,
            int(candidate["embeddingDim"]),
            int(candidate["hiddenDim"]),
            int(candidate["attentionDim"]),
        )
    else:
        raise ValueError("unknown sequence model")
    initialize_item_embeddings(model, pretrained, trainable=bool(candidate["trainItemEmbeddings"]))
    return model


def _run_one(
    model_name: str,
    *,
    data: PreparedAmazonData,
    pretrained: np.ndarray,
    candidate: Mapping[str, Any],
    candidates: np.ndarray,
    histories: Sequence[np.ndarray],
    targets: np.ndarray,
    seed: int,
    ks: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    positives, negatives, training_histories = _training_arrays(
        data,
        max_history=int(candidate["maxHistory"]),
        negatives_per_positive=int(candidate["negativesPerPositive"]),
        seed=seed,
    )
    _seed_everything(seed)
    model = _make_model(
        model_name,
        item_count=len(data.catalog),
        candidate=candidate,
        pretrained=pretrained,
    )
    model, losses = _train_model(
        model,
        histories=training_histories,
        positives=positives,
        negatives=negatives,
        candidate=candidate,
        seed=seed,
        device=device,
    )
    rows = _rerank(
        model,
        histories,
        candidates,
        max_history=int(candidate["maxHistory"]),
        query_batch_size=int(candidate["evaluationQueryBatchSize"]),
        device=device,
    )
    metrics = ranking_metrics(rows, targets, ks)
    model.to("cpu")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"metrics": metrics, "epochLosses": losses, "elapsedSeconds": time.perf_counter() - started}


def _paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(config["artifactsDir"])
    return root, root / "dev-selection.json", root / "test-final-receipt.json"


def run_dev_selection(config_path: str | Path) -> Mapping[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    data, metadata, retriever, payload = _prepare(config)
    root, selection_path, receipt_path = _paths(config)
    if selection_path.exists() or receipt_path.exists() or Path(config["output"]).exists():
        raise FileExistsError("sequence V2 frozen phases cannot be overwritten")
    root.mkdir(parents=True, exist_ok=True)
    pretrained = pretrained_item_embeddings(retriever, len(data.catalog))
    dev_candidates = _candidates(
        retriever,
        data,
        payload,
        split="dev",
        k=int(config["protocol"]["candidateCount"]),
    )
    rows: list[dict[str, Any]] = []
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for candidate in config["candidateConfigs"]:
        result = _run_one(
            "din",
            data=data,
            pretrained=pretrained,
            candidate=candidate,
            candidates=dev_candidates,
            histories=data.dev_histories,
            targets=data.dev_targets,
            seed=int(config["protocol"]["selectionSeed"]),
            ks=config["protocol"]["ks"],
            device=device,
        )
        rows.append({
            "id": candidate["id"],
            "config": dict(candidate),
            "dinDevMetrics": result["metrics"],
            "selectionValue": result["metrics"]["100"]["ndcg"],
            "epochLosses": result["epochLosses"],
            "elapsedSeconds": result["elapsedSeconds"],
        })
    winner = sorted(rows, key=lambda row: (-float(row["selectionValue"]), str(row["id"])))[0]
    manifest = {
        "schemaVersion": 1,
        "status": "DEV_SELECTION_COMPLETE_TEST_UNSEEN",
        "experimentId": config["experimentId"],
        "configCanonicalSha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "splitSha256": data.split_hash,
        "retrieverCheckpointSha256": config["retriever"]["checkpointSha256"],
        "metadataFeatureSha256": metadata.evidence["featureSha256"],
        "selectionSeed": config["protocol"]["selectionSeed"],
        "selectionMetric": config["protocol"]["selectionMetric"],
        "candidateGeneratorDevMetrics": ranking_metrics(dev_candidates, data.dev_targets, config["protocol"]["ks"]),
        "candidates": rows,
        "selectedCandidateId": winner["id"],
        "selectedConfig": winner["config"],
        "testMetrics": None,
    }
    selection_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_final_test(config_path: str | Path) -> Mapping[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    data, metadata, retriever, payload = _prepare(config)
    _, selection_path, receipt_path = _paths(config)
    output_path = Path(config["output"])
    if not selection_path.is_file():
        raise FileNotFoundError("dev selection must complete before test")
    if receipt_path.exists() or output_path.exists():
        raise FileExistsError("sequence V2 test can only run once")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "DEV_SELECTION_COMPLETE_TEST_UNSEEN":
        raise ValueError("invalid selection gate")
    if selection.get("configCanonicalSha256") != hashlib.sha256(_canonical_json(config)).hexdigest():
        raise ValueError("config changed after dev selection")
    if selection.get("splitSha256") != data.split_hash:
        raise ValueError("split changed after dev selection")
    candidate = selection["selectedConfig"]
    pretrained = pretrained_item_embeddings(retriever, len(data.catalog))
    test_candidates = _candidates(
        retriever,
        data,
        payload,
        split="test",
        k=int(config["protocol"]["candidateCount"]),
    )
    ks = config["protocol"]["ks"]
    candidate_metrics = ranking_metrics(test_candidates, data.test_targets, ks)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    per_model_metrics: dict[str, list[Mapping[str, Mapping[str, float | int]]]] = {"meanPooling": [], "din": []}
    per_seed: dict[str, dict[str, Any]] = {"meanPooling": {}, "din": {}}
    for seed in config["protocol"]["testSeeds"]:
        for model_name in ("meanPooling", "din"):
            result = _run_one(
                model_name,
                data=data,
                pretrained=pretrained,
                candidate=candidate,
                candidates=test_candidates,
                histories=data.test_histories,
                targets=data.test_targets,
                seed=int(seed),
                ks=ks,
                device=device,
            )
            per_model_metrics[model_name].append(result["metrics"])
            per_seed[model_name][str(seed)] = result
    delta = _paired_metric_delta(per_model_metrics["din"], per_model_metrics["meanPooling"])
    improved = float(delta["100"]["ndcg"]["mean"]) > 0
    outcome = "POSITIVE_DIN_TEST_IMPROVEMENT" if improved else "NEGATIVE_NO_STABLE_DIN_TEST_IMPROVEMENT"
    retrieval_config = load_v2_config(config["retriever"]["config"])
    raw_root = Path(retrieval_config["dataset"]["rawDir"])
    dataset_files = []
    for split in ("train", "dev", "test"):
        source_path = raw_root / retrieval_config["dataset"]["files"][split]["name"]
        evidence = _file_evidence(source_path)
        dataset_files.append({"name": source_path.name, "bytes": evidence["bytes"], "sha256": evidence["sha256"], "role": split})
    metadata_path = raw_root / retrieval_config["metadata"]["file"]
    metadata_evidence = _file_evidence(metadata_path)
    dataset_files.append({"name": metadata_path.name, "bytes": metadata_evidence["bytes"], "sha256": metadata_evidence["sha256"], "role": "item_metadata"})
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": config["experimentId"],
        "status": "COMPLETE",
        "outcome": outcome,
        "dataOrigin": "public",
        "claimableOnlinePerformance": False,
        "source": config["source"],
        "evidence": {
            "configSha256": _file_evidence(path)["sha256"],
            "splitSha256": data.split_hash,
            "datasetFiles": dataset_files,
            "retrieverCheckpoint": _file_evidence(Path(config["retriever"]["checkpoint"])),
            "metadataFeatureSha256": metadata.evidence["featureSha256"],
            "selectionManifestSha256": _file_evidence(selection_path)["sha256"],
        },
        "protocol": {
            "selectionSplit": "dev",
            "selectionSeed": config["protocol"]["selectionSeed"],
            "selectionMetric": config["protocol"]["selectionMetric"],
            "testOpenedAfterSelectionFrozen": True,
            "testExecutionCount": 1,
            "testSeeds": config["protocol"]["testSeeds"],
            "seeds": config["protocol"]["testSeeds"],
            "candidateCount": config["protocol"]["candidateCount"],
            "sameCandidatesNegativesAndInitializationAcrossModels": True,
            "testDataUsedForSelection": False,
            "catalogItems": len(data.catalog),
            "testEvaluationUsers": len(data.test_query_users),
            "counts": {
                "trainRows": len(data.train),
                "devRows": len(data.dev),
                "testRows": len(data.test),
                "trainCatalogItems": len(data.catalog),
                "devEvaluationUsers": len(data.dev_query_users),
                "testEvaluationUsers": len(data.test_query_users)
            },
        },
        "devSelection": selection,
        "selectedCandidate": candidate,
        "results": {
            "candidateGeneration": {
                "name": "frozen_metadata_two_tower_v2",
                "targetInjected": False,
                "testMetrics": candidate_metrics,
            },
            "models": {
                model_name: {
                    "perSeed": per_seed[model_name],
                    "testSummary": _metric_summary(per_model_metrics[model_name]),
                }
                for model_name in ("meanPooling", "din")
            },
            "comparison": {
                "contrast": "din_minus_mean_pooling_paired_by_seed",
                "delta": delta,
                "outcome": outcome,
            },
        },
        "limitations": config["limitations"],
    }
    validate_public_report(report)
    report["resultSha256"] = canonical_sha256(report)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps({
        "schemaVersion": 1,
        "status": "TEST_FINAL_EXECUTED_ONCE",
        "resultSha256": _file_evidence(output_path)["sha256"],
        "outcome": outcome,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Amazon Sequence V2 — Retriever-aligned pretrained DIN",
        "",
        f"Status: `{report['status']}` / `{report['outcome']}`; public offline only.",
        "",
        "## Frozen protocol",
        "",
        f"- Candidate generator: metadata Two-Tower V2 Top-{report['protocol']['candidateCount']}; target never injected.",
        "- Mean Pooling and DIN share candidates, negatives, pretrained item vectors, optimizer settings, and seeds.",
        "- Three configurations selected using dev DIN NDCG@100; test opened once afterward.",
        "",
        "## Test metrics — mean ± population std",
        "",
        "| Model | NDCG@20 | NDCG@50 | NDCG@100 | MRR@100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (("meanPooling", "Mean Pooling"), ("din", "DIN")):
        summary = report["results"]["models"][name]["testSummary"]
        values = [summary[k][metric] for k, metric in (("20", "ndcg"), ("50", "ndcg"), ("100", "ndcg"), ("100", "mrr"))]
        lines.append("| " + label + " | " + " | ".join(f"{value['mean']:.6f} ± {value['std']:.6f}" for value in values) + " |")
    delta = report["results"]["comparison"]["delta"]
    lines.extend([
        "",
        f"DIN minus Mean Pooling NDCG@100: {delta['100']['ndcg']['mean']:+.6f} ± {delta['100']['ndcg']['std']:.6f}.",
        "",
        "## Limitations",
        "",
        *(f"- {item}" for item in report["limitations"]),
        "",
    ])
    return "\n".join(lines)

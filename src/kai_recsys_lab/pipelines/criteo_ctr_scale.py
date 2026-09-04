from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..ctr.encoding import TabularBatch, TabularSchema
from ..ctr.metrics import BinaryPredictionMetrics, evaluate_binary_predictions
from ..ctr.models import DCNv2, DeepFM, binary_logit_loss
from ..public_report import canonical_sha256, validate_public_report
from .criteo_ctr import _device, sha256_file


RAW_NUMERIC_COLUMNS = tuple(f"integer_feature_{index}" for index in range(1, 14))
RAW_CATEGORICAL_COLUMNS = tuple(f"categorical_feature_{index}" for index in range(1, 27))
NUMERIC_NAMES = tuple(f"int_{index}" for index in range(1, 14))
CATEGORICAL_NAMES = tuple(f"cat_{index}" for index in range(1, 27))


class ScalableCriteoPreprocessor:
    """Array-native, train-fitted Criteo preprocessing for a complete parquet shard."""

    def __init__(self, *, min_category_count: int, max_categories_per_feature: int) -> None:
        if min_category_count < 1 or max_categories_per_feature < 1:
            raise ValueError("categorical thresholds must be positive")
        self.min_category_count = min_category_count
        self.max_categories_per_feature = max_categories_per_feature
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.vocabularies: list[dict[str, int]] = []

    @staticmethod
    def _numeric_frame(frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, RAW_NUMERIC_COLUMNS].fillna(0).to_numpy(dtype=np.float64)
        values = np.log1p(np.maximum(values, 0.0))
        if not np.isfinite(values).all():
            raise ValueError("numeric Criteo values must be finite")
        return values

    @staticmethod
    def _categorical_series(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame[name].fillna("__MISSING__").astype(str)

    def fit(self, train: pd.DataFrame) -> "ScalableCriteoPreprocessor":
        if train.empty:
            raise ValueError("preprocessor requires non-empty train data")
        numeric = self._numeric_frame(train)
        self.numeric_mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        self.numeric_std = np.where(std > 1e-12, std, 1.0)
        self.vocabularies = []
        for name in RAW_CATEGORICAL_COLUMNS:
            counts = self._categorical_series(train, name).value_counts(sort=False)
            retained = [
                (str(value), int(count))
                for value, count in counts.items()
                if int(count) >= self.min_category_count
            ]
            retained.sort(key=lambda row: (-row[1], row[0]))
            self.vocabularies.append(
                {value: index + 1 for index, (value, _) in enumerate(retained[: self.max_categories_per_feature])}
            )
        return self

    @property
    def schema(self) -> TabularSchema:
        if len(self.vocabularies) != len(CATEGORICAL_NAMES):
            raise RuntimeError("preprocessor must be fitted")
        return TabularSchema(
            numeric_names=NUMERIC_NAMES,
            categorical_names=CATEGORICAL_NAMES,
            categorical_cardinalities=tuple(len(vocabulary) + 1 for vocabulary in self.vocabularies),
        )

    def transform(self, frame: pd.DataFrame) -> TabularBatch:
        if self.numeric_mean is None or self.numeric_std is None or not self.vocabularies:
            raise RuntimeError("preprocessor must be fitted")
        numeric = (self._numeric_frame(frame) - self.numeric_mean) / self.numeric_std
        categorical = np.zeros((len(frame), len(RAW_CATEGORICAL_COLUMNS)), dtype=np.int64)
        for column, (name, vocabulary) in enumerate(zip(RAW_CATEGORICAL_COLUMNS, self.vocabularies, strict=True)):
            categorical[:, column] = (
                self._categorical_series(frame, name).map(vocabulary).fillna(0).to_numpy(dtype=np.int64)
            )
        batch = TabularBatch(
            numeric=torch.from_numpy(numeric.astype(np.float32, copy=False)),
            categorical=torch.from_numpy(categorical),
        )
        batch.validate(self.schema)
        return batch

    def report(self) -> dict[str, Any]:
        return {
            "numericTransform": "log1p_nonnegative_then_train_zscore",
            "categoricalUnknownIndex": 0,
            "minCategoryCount": self.min_category_count,
            "maxCategoriesPerFeature": self.max_categories_per_feature,
            "keptCategories": {
                name: len(vocabulary)
                for name, vocabulary in zip(CATEGORICAL_NAMES, self.vocabularies, strict=True)
            },
        }


class SparseLinearCTR(nn.Module):
    """First-order logistic model with one scalar parameter per category."""

    def __init__(self, schema: TabularSchema) -> None:
        super().__init__()
        self.schema = schema
        self.numeric = nn.Parameter(torch.zeros(len(schema.numeric_names)))
        self.categorical = nn.ModuleList(
            nn.Embedding(cardinality, 1, padding_idx=0)
            for cardinality in schema.categorical_cardinalities
        )
        for embedding in self.categorical:
            nn.init.zeros_(embedding.weight)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, batch: TabularBatch) -> torch.Tensor:
        batch.validate(self.schema)
        result = self.bias + batch.numeric @ self.numeric
        for index, embedding in enumerate(self.categorical):
            result = result + embedding(batch.categorical[:, index]).squeeze(-1)
        return result


def _labels(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Criteo label must be binary")
    return labels


def _predict(model: nn.Module, batch: TabularBatch, *, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, batch_size):
            stop = min(batch.batch_size, start + batch_size)
            inputs = TabularBatch(
                numeric=batch.numeric[start:stop].to(device),
                categorical=batch.categorical[start:stop].to(device),
            )
            result.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.concatenate(result)


def _train(
    model: nn.Module,
    *,
    train_batch: TabularBatch,
    train_labels: np.ndarray,
    dev_batch: TabularBatch,
    dev_labels: np.ndarray,
    seed: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learningRate"]),
        weight_decay=float(config["weightDecay"]),
    )
    labels = torch.from_numpy(train_labels.astype(np.float32, copy=False))
    batch_size = int(config["batchSize"])
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(train_batch.batch_size, generator=generator)
        train_loss = 0.0
        seen = 0
        for start in range(0, train_batch.batch_size, batch_size):
            indices = permutation[start : start + batch_size]
            inputs = TabularBatch(
                numeric=train_batch.numeric[indices].to(device),
                categorical=train_batch.categorical[indices].to(device),
            )
            target = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = binary_logit_loss(model(inputs), target)
            loss.backward()
            optimizer.step()
            count = int(indices.numel())
            train_loss += float(loss.detach().cpu()) * count
            seen += count
        probability = _predict(model, dev_batch, device=device, batch_size=batch_size)
        dev_metrics = evaluate_binary_predictions(dev_labels, probability)
        history.append({"epoch": epoch, "trainLogLoss": train_loss / seen, "devLogLoss": dev_metrics.log_loss})
        if dev_metrics.log_loss < best_loss:
            best_loss = dev_metrics.log_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "bestEpoch": best_epoch,
        "bestDevLogLoss": best_loss,
        "history": history,
        "trainSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def _metric_payload(metrics: BinaryPredictionMetrics) -> dict[str, Any]:
    value = asdict(metrics)
    return {
        "nExamples": value["n_examples"],
        "positiveRate": value["positive_rate"],
        "rocAuc": value["auc"],
        "prAuc": value["pr_auc"],
        "logLoss": value["log_loss"],
        "brierScore": value["brier_score"],
        "ece": value["expected_calibration_error"],
    }


def _summaries(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in sorted({str(run["model"]) for run in runs}):
        rows = [run for run in runs if run["model"] == model]
        result[model] = {"seeds": [row["seed"] for row in rows]}
        for metric in ("rocAuc", "prAuc", "logLoss", "brierScore", "ece"):
            values = [float(row["testMetrics"][metric]) for row in rows]
            result[model][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
            }
    return result


def run_criteo_ctr_scale(config: Mapping[str, Any], *, config_path: str | Path) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["protocol"]["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("scale protocol needs unique seeds")
    raw_path = Path(config["dataset"]["rawFile"])
    frame = pd.read_parquet(raw_path)
    expected_columns = {"label", *RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS}
    if set(frame.columns) != expected_columns:
        raise ValueError("pinned Criteo parquet schema drifted")
    row_limit = config["dataset"].get("rowLimit")
    if row_limit is not None:
        frame = frame.iloc[: int(row_limit)].copy()
    expected_rows = int(config["dataset"]["expectedRows"])
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(frame)}")
    train_end = int(len(frame) * float(config["split"]["trainFraction"]))
    dev_end = train_end + int(len(frame) * float(config["split"]["devFraction"]))
    train_frame = frame.iloc[:train_end]
    dev_frame = frame.iloc[train_end:dev_end]
    test_frame = frame.iloc[dev_end:]
    if min(len(train_frame), len(dev_frame), len(test_frame)) < 1:
        raise ValueError("all splits must be non-empty")
    split_digest = hashlib.sha256()
    for name, rows in (("train", train_frame), ("dev", dev_frame), ("test", test_frame)):
        for offset, label in enumerate(_labels(rows), start=int(rows.index[0])):
            split_digest.update(f"{name}\t{offset}\t{int(label)}\n".encode())

    preprocessor = ScalableCriteoPreprocessor(
        min_category_count=int(config["preprocessing"]["minCategoryCount"]),
        max_categories_per_feature=int(config["preprocessing"]["maxCategoriesPerFeature"]),
    ).fit(train_frame)
    train_batch = preprocessor.transform(train_frame)
    dev_batch = preprocessor.transform(dev_frame)
    test_batch = preprocessor.transform(test_frame)
    train_labels, dev_labels, test_labels = _labels(train_frame), _labels(dev_frame), _labels(test_frame)
    del frame

    device = _device(str(config["training"]["device"]))
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        factories = {
            "logistic_regression": lambda: SparseLinearCTR(preprocessor.schema),
            "deepfm": lambda: DeepFM(
                preprocessor.schema,
                embedding_dim=int(config["models"]["deepfm"]["embeddingDim"]),
                hidden_dims=tuple(config["models"]["deepfm"]["hiddenDims"]),
                seed=seed,
            ),
            "dcnv2": lambda: DCNv2(
                preprocessor.schema,
                embedding_dim=int(config["models"]["dcnv2"]["embeddingDim"]),
                cross_depth=int(config["models"]["dcnv2"]["crossDepth"]),
                hidden_dims=tuple(config["models"]["dcnv2"]["hiddenDims"]),
                seed=seed,
            ),
        }
        for model_name, factory in factories.items():
            model, training = _train(
                factory(),
                train_batch=train_batch,
                train_labels=train_labels,
                dev_batch=dev_batch,
                dev_labels=dev_labels,
                seed=seed,
                config=config["training"],
                device=device,
            )
            probability = _predict(
                model,
                test_batch,
                device=device,
                batch_size=int(config["training"]["batchSize"]),
            )
            runs.append({
                "model": model_name,
                "seed": seed,
                "selectionMetric": {"name": "devLogLoss", "value": training["bestDevLogLoss"]},
                "testMetrics": _metric_payload(evaluate_binary_predictions(test_labels, probability)),
                "training": training,
            })
            model.to("cpu")
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    counts = {
        "allRows": expected_rows,
        "trainRows": len(train_frame),
        "devRows": len(dev_frame),
        "testRows": len(test_frame),
        "allPositiveRows": int(train_labels.sum() + dev_labels.sum() + test_labels.sum()),
        "trainPositiveRows": int(train_labels.sum()),
        "devPositiveRows": int(dev_labels.sum()),
        "testPositiveRows": int(test_labels.sum()),
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": config["experimentId"],
        "status": "COMPLETE",
        "dataOrigin": "public",
        "claimableOnlinePerformance": False,
        "source": config["source"],
        "evidence": {
            "datasetFiles": [{
                "name": raw_path.name,
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
                "role": "complete_pinned_official_parquet_shard",
            }],
            "configSha256": sha256_file(config_path),
            "splitSha256": split_digest.hexdigest(),
        },
        "dataset": {
            "scope": "complete_pinned_official_parquet_shard_not_full_1tb_dataset",
            "rowsRead": expected_rows,
            "positiveRows": counts["allPositiveRows"],
            "samplingRule": config["dataset"]["samplingRule"],
        },
        "protocol": {
            "counts": counts,
            "splitSemantics": config["split"]["semantics"],
            "seeds": seeds,
            "sameRowsAndFeaturesForAllModels": True,
            "preprocessing": preprocessor.report(),
            "testDataUsedForSelection": False,
            "training": config["training"],
        },
        "results": {"runs": runs, "summary": _summaries(runs)},
        "limitations": config["limitations"],
    }
    validate_public_report(report)
    report["resultSha256"] = canonical_sha256(report)
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["results"]["summary"]
    lines = [
        "# Criteo CTR complete pinned-shard scale V2",
        "",
        f"Status: `{report['status']}`; public offline only; online claim: `false`.",
        "",
        "## Protocol",
        "",
        f"- Complete pinned parquet shard rows: {report['dataset']['rowsRead']:,}.",
        f"- Seeds: {', '.join(str(seed) for seed in report['protocol']['seeds'])}.",
        "- Same rows, 13 numeric + 26 categorical fields, train-fitted preprocessing, and dev checkpoint rule.",
        "- Source-order split; parquet conversion order is not claimed chronological.",
        "",
        "## Test metrics — mean ± population std",
        "",
        "| Model | ROC-AUC | PR-AUC | LogLoss | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("logistic_regression", "LR"), ("deepfm", "DeepFM"), ("dcnv2", "DCNv2")):
        row = summary[name]
        values = [row[key] for key in ("rocAuc", "prAuc", "logLoss", "brierScore", "ece")]
        lines.append("| " + label + " | " + " | ".join(f"{value['mean']:.6f} ± {value['std']:.6f}" for value in values) + " |")
    lines.extend(["", "## Limitations", "", *(f"- {item}" for item in report["limitations"]), ""])
    return "\n".join(lines)

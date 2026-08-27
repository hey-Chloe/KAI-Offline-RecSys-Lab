from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..conversion import (
    ESMM,
    IndependentBinaryTower,
    esmm_loss,
    evaluate_esmm_predictions,
)
from ..ctr.encoding import TabularBatch, TabularSchema
from ..ctr.metrics import BinaryPredictionMetrics, evaluate_binary_predictions
from ..public_report import canonical_sha256, sha256_file, validate_public_report


ATTRIBUTION_COLUMNS = (
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
    "click_pos",
    "click_nb",
    "cost",
    "cpo",
    "time_since_last_click",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
)
NUMERIC_NAMES = ("log_cost", "log_time_since_last_click", "prior_click_observed")
CATEGORICAL_NAMES = ("campaign",) + tuple(f"cat{index}" for index in range(1, 10))
EXCLUDED_OUTCOME_OR_IDENTITY_FEATURES = (
    "uid",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
    "click_pos",
    "click_nb",
    "cpo",
)


@dataclass(frozen=True, slots=True)
class TemporalFrameSplit:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    digest: str


def load_criteo_attribution_impressions(
    path: str | Path,
    *,
    row_limit: int,
) -> pd.DataFrame:
    """Load a deterministic public-data prefix and construct an explicit CTCVR label.

    Criteo's published ``conversion`` label may include view-through outcomes.
    ESMM supervision here is therefore the conservative observable funnel label
    ``ctcvr = click * conversion``.  The raw conversion column is retained only
    for audit counts and is never silently called post-click conversion.
    """

    if row_limit < 100:
        raise ValueError("row_limit must be at least 100")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(
        source,
        sep="\t",
        compression="infer",
        usecols=list(ATTRIBUTION_COLUMNS),
        nrows=row_limit,
    )
    if len(frame) != row_limit:
        raise ValueError(f"expected exactly {row_limit} rows, found {len(frame)}")
    missing = set(ATTRIBUTION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"attribution data is missing columns: {sorted(missing)}")
    for label in ("click", "conversion"):
        values = frame[label].to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{label} must be binary")
        frame[label] = values.astype(np.int8)
    timestamp = frame["timestamp"].to_numpy(dtype=np.int64)
    if np.any(timestamp[1:] < timestamp[:-1]):
        raise ValueError("official attribution rows must remain timestamp-sorted")
    frame["ctcvr"] = (
        frame["click"].to_numpy(dtype=np.int8)
        * frame["conversion"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    return frame


def fixed_temporal_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    dev_fraction: float,
) -> TemporalFrameSplit:
    if not (0.0 < train_fraction < 1.0 and 0.0 < dev_fraction < 1.0):
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("split fractions must leave a test partition")
    train_end = int(len(frame) * train_fraction)
    dev_end = train_end + int(len(frame) * dev_fraction)
    if train_end < 2 or dev_end <= train_end or dev_end >= len(frame):
        raise ValueError("train/dev/test partitions must be non-empty")
    parts = (
        ("train", frame.iloc[:train_end].copy()),
        ("dev", frame.iloc[train_end:dev_end].copy()),
        ("test", frame.iloc[dev_end:].copy()),
    )
    digest = hashlib.sha256()
    for name, partition in parts:
        digest.update(name.encode("ascii"))
        evidence = partition[["timestamp", "click", "conversion", "ctcvr"]].to_numpy(
            dtype="<i8", copy=True
        )
        digest.update(evidence.tobytes(order="C"))
    return TemporalFrameSplit(
        train=parts[0][1],
        dev=parts[1][1],
        test=parts[2][1],
        digest=digest.hexdigest(),
    )


class TrainFittedAttributionPreprocessor:
    """Train-only numeric scaling and categorical vocabulary capping."""

    def __init__(self, *, min_category_count: int, max_categories_per_feature: int) -> None:
        if min_category_count < 1 or max_categories_per_feature < 1:
            raise ValueError("categorical thresholds must be positive")
        self.min_category_count = min_category_count
        self.max_categories_per_feature = max_categories_per_feature
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.fit_rows = 0

    @staticmethod
    def _numeric(frame: pd.DataFrame) -> np.ndarray:
        cost = frame["cost"].to_numpy(dtype=np.float64)
        since = frame["time_since_last_click"].to_numpy(dtype=np.float64)
        if not np.isfinite(cost).all() or not np.isfinite(since).all():
            raise ValueError("numeric features must be finite")
        return np.column_stack(
            (
                np.log1p(np.clip(cost, 0.0, None)),
                np.log1p(np.clip(since, 0.0, None)),
                (since >= 0.0).astype(np.float64),
            )
        )

    def fit(self, frame: pd.DataFrame) -> "TrainFittedAttributionPreprocessor":
        if frame.empty:
            raise ValueError("preprocessor requires train impressions")
        numeric = self._numeric(frame)
        self.numeric_mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        self.numeric_std = np.where(std > 1e-12, std, 1.0)
        for name in CATEGORICAL_NAMES:
            values = frame[name].astype("string")
            counts = values.value_counts(dropna=False)
            eligible = [
                (str(value), int(count))
                for value, count in counts.items()
                if int(count) >= self.min_category_count
            ]
            eligible.sort(key=lambda item: (-item[1], item[0]))
            self.vocabularies[name] = {
                value: index + 1
                for index, (value, _) in enumerate(
                    eligible[: self.max_categories_per_feature]
                )
            }
        self.fit_rows = len(frame)
        return self

    @property
    def schema(self) -> TabularSchema:
        if self.numeric_mean is None or not self.vocabularies:
            raise RuntimeError("preprocessor must be fitted first")
        return TabularSchema(
            numeric_names=NUMERIC_NAMES,
            categorical_names=CATEGORICAL_NAMES,
            categorical_cardinalities=tuple(
                len(self.vocabularies[name]) + 1 for name in CATEGORICAL_NAMES
            ),
        )

    def transform(self, frame: pd.DataFrame) -> TabularBatch:
        if self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("preprocessor must be fitted first")
        numeric = (self._numeric(frame) - self.numeric_mean) / self.numeric_std
        categorical = np.column_stack(
            [
                frame[name]
                .astype("string")
                .map(self.vocabularies[name])
                .fillna(0)
                .to_numpy(dtype=np.int64)
                for name in CATEGORICAL_NAMES
            ]
        )
        batch = TabularBatch(
            numeric=torch.from_numpy(numeric.astype(np.float32, copy=False)),
            categorical=torch.from_numpy(categorical),
        )
        batch.validate(self.schema)
        return batch

    def report(self) -> dict[str, Any]:
        return {
            "fitRows": self.fit_rows,
            "numericTransform": "train_zscore(log1p_nonnegative); explicit prior-click indicator",
            "categoricalUnknownIndex": 0,
            "minCategoryCount": self.min_category_count,
            "maxCategoriesPerFeature": self.max_categories_per_feature,
            "keptCategories": {
                name: len(self.vocabularies[name]) for name in CATEGORICAL_NAMES
            },
        }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def _slice(batch: TabularBatch, indices: torch.Tensor | slice) -> TabularBatch:
    return TabularBatch(
        numeric=batch.numeric[indices],
        categorical=batch.categorical[indices],
    )


def _clicked_batch(batch: TabularBatch, clicked: np.ndarray) -> TabularBatch:
    indices = torch.from_numpy(np.flatnonzero(clicked).astype(np.int64))
    if indices.numel() == 0:
        raise ValueError("post-click training requires clicked impressions")
    return _slice(batch, indices)


def _predict_binary(
    model: nn.Module,
    batch: TabularBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, batch_size):
            inputs = _slice(batch, slice(start, min(start + batch_size, batch.batch_size)))
            logits = model(
                TabularBatch(
                    numeric=inputs.numeric.to(device),
                    categorical=inputs.categorical.to(device),
                )
            )
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions)


def _predict_esmm(
    model: ESMM,
    batch: TabularBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ctr: list[np.ndarray] = []
    ctcvr: list[np.ndarray] = []
    cvr: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, batch_size):
            inputs = _slice(batch, slice(start, min(start + batch_size, batch.batch_size)))
            output = model(
                TabularBatch(
                    numeric=inputs.numeric.to(device),
                    categorical=inputs.categorical.to(device),
                )
            )
            ctr.append(output.ctr_probability.cpu().numpy())
            ctcvr.append(output.ctcvr_probability.cpu().numpy())
            cvr.append(output.inferred_cvr().cpu().numpy())
    return np.concatenate(ctr), np.concatenate(ctcvr), np.concatenate(cvr)


def _train_binary(
    model: IndependentBinaryTower,
    *,
    train_batch: TabularBatch,
    train_labels: np.ndarray,
    dev_batch: TabularBatch,
    dev_labels: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[IndependentBinaryTower, dict[str, Any]]:
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    labels = torch.from_numpy(train_labels.astype(np.float32, copy=False))
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_batch.batch_size, generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, train_batch.batch_size, batch_size):
            indices = permutation[start : start + batch_size]
            inputs = _slice(train_batch, indices)
            target = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                TabularBatch(
                    numeric=inputs.numeric.to(device),
                    categorical=inputs.categorical.to(device),
                )
            )
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
            count = int(indices.numel())
            seen += count
            total_loss += float(loss.detach().cpu()) * count
        dev_probability = _predict_binary(
            model, dev_batch, device=device, batch_size=batch_size
        )
        dev_metrics = evaluate_binary_predictions(dev_labels, dev_probability)
        history.append(
            {
                "epoch": epoch,
                "trainLogLoss": total_loss / seen,
                "devLogLoss": dev_metrics.log_loss,
            }
        )
        if dev_metrics.log_loss < best_loss:
            best_loss = dev_metrics.log_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("binary training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "bestEpoch": best_epoch,
        "bestDevLogLoss": best_loss,
        "history": history,
        "trainSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def _train_esmm(
    model: ESMM,
    *,
    train_batch: TabularBatch,
    train_click: np.ndarray,
    train_ctcvr: np.ndarray,
    dev_batch: TabularBatch,
    dev_click: np.ndarray,
    dev_ctcvr: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[ESMM, dict[str, Any]]:
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    click = torch.from_numpy(train_click.astype(np.float32, copy=False))
    ctcvr = torch.from_numpy(train_ctcvr.astype(np.float32, copy=False))
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_batch.batch_size, generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, train_batch.batch_size, batch_size):
            indices = permutation[start : start + batch_size]
            inputs = _slice(train_batch, indices)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                TabularBatch(
                    numeric=inputs.numeric.to(device),
                    categorical=inputs.categorical.to(device),
                )
            )
            loss = esmm_loss(
                output,
                click[indices].to(device),
                ctcvr[indices].to(device),
            )
            loss.backward()
            optimizer.step()
            count = int(indices.numel())
            seen += count
            total_loss += float(loss.detach().cpu()) * count
        dev_ctr, dev_ctcvr_probability, _ = _predict_esmm(
            model, dev_batch, device=device, batch_size=batch_size
        )
        ctr_loss = evaluate_binary_predictions(dev_click, dev_ctr).log_loss
        ctcvr_loss = evaluate_binary_predictions(dev_ctcvr, dev_ctcvr_probability).log_loss
        selection_loss = ctr_loss + ctcvr_loss
        history.append(
            {
                "epoch": epoch,
                "trainJointLogLoss": total_loss / seen,
                "devCtrLogLoss": ctr_loss,
                "devCtcvrLogLoss": ctcvr_loss,
                "devJointLogLoss": selection_loss,
            }
        )
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("ESMM training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "bestEpoch": best_epoch,
        "bestDevJointLogLoss": best_loss,
        "history": history,
        "trainSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def _metric_payload(metrics: BinaryPredictionMetrics) -> dict[str, Any]:
    value = asdict(metrics)
    return {
        "nExamples": value["n_examples"],
        "positiveRows": round(value["positive_rate"] * value["n_examples"]),
        "positiveRate": value["positive_rate"],
        "rocAuc": value["auc"],
        "prAuc": value["pr_auc"],
        "averagePrecision": value["average_precision"],
        "logLoss": value["log_loss"],
        "brierScore": value["brier_score"],
        "ece": value["expected_calibration_error"],
        "calibration": [
            {
                "lower": item["lower"],
                "upper": item["upper"],
                "count": item["count"],
                "meanPrediction": item["mean_prediction"],
                "positiveRate": item["positive_rate"],
            }
            for item in value["calibration"]
        ],
    }


def _funnel_metrics(
    click: np.ndarray,
    ctcvr: np.ndarray,
    ctr_probability: np.ndarray,
    ctcvr_probability: np.ndarray,
    cvr_probability: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_esmm_predictions(
        click,
        ctcvr,
        ctr_probability,
        ctcvr_probability,
        cvr_probability,
    )
    if metrics.post_click_cvr is None:
        raise ValueError("test partition has no clicked impressions")
    return {
        "ctr": _metric_payload(metrics.ctr),
        "ctcvr": _metric_payload(metrics.ctcvr),
        "postClickCvr": _metric_payload(metrics.post_click_cvr),
    }


def _summaries(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("rocAuc", "prAuc", "logLoss", "brierScore", "ece")
    result: dict[str, Any] = {}
    for model_name in sorted({str(run["model"]) for run in runs}):
        model_runs = [run for run in runs if run["model"] == model_name]
        tasks: dict[str, Any] = {}
        for task in ("ctr", "ctcvr", "postClickCvr"):
            task_summary: dict[str, Any] = {}
            for metric in metric_names:
                values = [run["testMetrics"][task][metric] for run in model_runs]
                if any(value is None for value in values):
                    task_summary[metric] = {"mean": None, "std": None}
                else:
                    numbers = [float(value) for value in values]
                    task_summary[metric] = {
                        "mean": statistics.fmean(numbers),
                        "std": statistics.pstdev(numbers),
                    }
            tasks[task] = task_summary
        result[model_name] = {
            "seeds": [int(run["seed"]) for run in model_runs],
            "tasks": tasks,
        }
    return result


def _labels(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy(dtype=np.int64)


def _count_payload(frame: pd.DataFrame) -> dict[str, int]:
    click = _labels(frame, "click")
    conversion = _labels(frame, "conversion")
    ctcvr = _labels(frame, "ctcvr")
    return {
        "rows": len(frame),
        "clickRows": int(click.sum()),
        "rawConversionRows": int(conversion.sum()),
        "ctcvrRows": int(ctcvr.sum()),
        "viewThroughConversionRowsExcluded": int(((conversion == 1) & (click == 0)).sum()),
    }


def run_cvr_esmm_experiment(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["seeds"]]
    if seeds != [3407, 6502, 9109]:
        raise ValueError("CVR/ESMM v1 seed protocol is frozen to [3407, 6502, 9109]")
    raw_path = Path(config["dataset"]["rawFile"])
    frame = load_criteo_attribution_impressions(
        raw_path, row_limit=int(config["dataset"]["rowLimit"])
    )
    split = fixed_temporal_split(
        frame,
        train_fraction=float(config["split"]["trainFraction"]),
        dev_fraction=float(config["split"]["devFraction"]),
    )
    counts_by_split = {
        name: _count_payload(partition)
        for name, partition in (
            ("all", frame),
            ("train", split.train),
            ("dev", split.dev),
            ("test", split.test),
        )
    }
    for name in ("train", "dev", "test"):
        if counts_by_split[name]["clickRows"] == 0 or counts_by_split[name]["ctcvrRows"] == 0:
            raise ValueError(f"{name} split must contain clicks and CTCVR positives")

    preprocessor = TrainFittedAttributionPreprocessor(
        min_category_count=int(config["preprocessing"]["minCategoryCount"]),
        max_categories_per_feature=int(
            config["preprocessing"]["maxCategoriesPerFeature"]
        ),
    ).fit(split.train)
    train_batch = preprocessor.transform(split.train)
    dev_batch = preprocessor.transform(split.dev)
    test_batch = preprocessor.transform(split.test)
    labels = {
        partition_name: {
            label: _labels(partition, label)
            for label in ("click", "ctcvr")
        }
        for partition_name, partition in (
            ("train", split.train),
            ("dev", split.dev),
            ("test", split.test),
        )
    }
    train_clicked_batch = _clicked_batch(train_batch, labels["train"]["click"])
    dev_clicked_batch = _clicked_batch(dev_batch, labels["dev"]["click"])
    train_clicked_cvr = labels["train"]["ctcvr"][labels["train"]["click"] == 1]
    dev_clicked_cvr = labels["dev"]["ctcvr"][labels["dev"]["click"] == 1]
    device = _device(str(config["training"]["device"]))
    training = config["training"]
    model_config = config["models"]
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        naive_started = time.perf_counter()
        ctr_model, ctr_training = _train_binary(
            IndependentBinaryTower(
                preprocessor.schema,
                embedding_dim=int(model_config["independent"]["embeddingDim"]),
                hidden_dims=tuple(model_config["independent"]["hiddenDims"]),
                seed=seed,
            ),
            train_batch=train_batch,
            train_labels=labels["train"]["click"],
            dev_batch=dev_batch,
            dev_labels=labels["dev"]["click"],
            seed=seed,
            epochs=int(training["epochs"]),
            batch_size=int(training["batchSize"]),
            learning_rate=float(training["learningRate"]),
            weight_decay=float(training["weightDecay"]),
            device=device,
        )
        cvr_model, cvr_training = _train_binary(
            IndependentBinaryTower(
                preprocessor.schema,
                embedding_dim=int(model_config["independent"]["embeddingDim"]),
                hidden_dims=tuple(model_config["independent"]["hiddenDims"]),
                seed=seed + 1,
            ),
            train_batch=train_clicked_batch,
            train_labels=train_clicked_cvr,
            dev_batch=dev_clicked_batch,
            dev_labels=dev_clicked_cvr,
            seed=seed + 1,
            epochs=int(training["epochs"]),
            batch_size=int(training["batchSize"]),
            learning_rate=float(training["learningRate"]),
            weight_decay=float(training["weightDecay"]),
            device=device,
        )
        naive_ctr = _predict_binary(
            ctr_model, test_batch, device=device, batch_size=int(training["batchSize"])
        )
        naive_cvr = _predict_binary(
            cvr_model, test_batch, device=device, batch_size=int(training["batchSize"])
        )
        naive_ctcvr = np.clip(naive_ctr * naive_cvr, 0.0, 1.0)
        runs.append(
            {
                "model": "naive_independent",
                "seed": seed,
                "supervision": {
                    "ctr": "all_train_impressions",
                    "cvr": "clicked_train_impressions_only",
                    "ctcvr": "product_of_independent_ctr_and_cvr",
                },
                "testMetrics": _funnel_metrics(
                    labels["test"]["click"],
                    labels["test"]["ctcvr"],
                    naive_ctr,
                    naive_ctcvr,
                    naive_cvr,
                ),
                "training": {
                    "ctr": ctr_training,
                    "cvr": cvr_training,
                    "totalSeconds": time.perf_counter() - naive_started,
                },
            }
        )

        esmm_model, esmm_training = _train_esmm(
            ESMM(
                preprocessor.schema,
                embedding_dim=int(model_config["esmm"]["embeddingDim"]),
                hidden_dims=tuple(model_config["esmm"]["hiddenDims"]),
                seed=seed,
            ),
            train_batch=train_batch,
            train_click=labels["train"]["click"],
            train_ctcvr=labels["train"]["ctcvr"],
            dev_batch=dev_batch,
            dev_click=labels["dev"]["click"],
            dev_ctcvr=labels["dev"]["ctcvr"],
            seed=seed,
            epochs=int(training["epochs"]),
            batch_size=int(training["batchSize"]),
            learning_rate=float(training["learningRate"]),
            weight_decay=float(training["weightDecay"]),
            device=device,
        )
        esmm_ctr, esmm_ctcvr, esmm_cvr = _predict_esmm(
            esmm_model,
            test_batch,
            device=device,
            batch_size=int(training["batchSize"]),
        )
        runs.append(
            {
                "model": "esmm",
                "seed": seed,
                "supervision": {
                    "ctr": "all_train_impressions",
                    "ctcvr": "all_train_impressions_with_click_times_conversion_label",
                    "cvr": "inferred_as_bounded_ctcvr_over_ctr",
                },
                "testMetrics": _funnel_metrics(
                    labels["test"]["click"],
                    labels["test"]["ctcvr"],
                    esmm_ctr,
                    esmm_ctcvr,
                    esmm_cvr,
                ),
                "training": esmm_training,
            }
        )

    flattened_counts: dict[str, int] = {}
    for split_name, split_counts in counts_by_split.items():
        for key, value in split_counts.items():
            flattened_counts[f"{split_name}{key[0].upper()}{key[1:]}"] = value
    config_hash = (
        sha256_file(config_path)
        if config_path is not None
        else hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": config["experimentId"],
        "status": "COMPLETE",
        "dataOrigin": "public",
        "claimableOnlinePerformance": False,
        "source": config["source"],
        "evidence": {
            "datasetFiles": [
                {
                    "name": raw_path.name,
                    "sha256": sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "role": "official_criteo_attribution_impression_log",
                }
            ],
            "configSha256": config_hash,
            "splitSha256": split.digest,
        },
        "dataset": {
            "scope": "fixed_timestamp_sorted_prefix_of_official_impression_log",
            "samplingRule": config["dataset"]["samplingRule"],
            "labelDefinition": {
                "ctr": "click",
                "ctcvr": "click * raw conversion",
                "postClickCvr": "raw conversion among clicked impressions",
                "rawConversionCaveat": "Criteo raw conversion includes outcomes within 30 days independently of last-click attribution; non-click conversion rows are excluded from CTCVR.",
            },
        },
        "protocol": {
            "counts": flattened_counts,
            "countsBySplit": counts_by_split,
            "splitSemantics": "fixed_contiguous_timestamp_order_70_15_15",
            "seeds": seeds,
            "features": {
                "numeric": list(NUMERIC_NAMES),
                "categorical": list(CATEGORICAL_NAMES),
                "excludedOutcomeOrIdentityFeatures": list(
                    EXCLUDED_OUTCOME_OR_IDENTITY_FEATURES
                ),
                "sameFeatureSetForBothBaselines": True,
            },
            "preprocessing": preprocessor.report(),
            "testDataUsedForSelection": False,
            "selection": "minimum dev LogLoss for each independent tower; minimum dev CTR+CTCVR LogLoss for ESMM",
        },
        "results": {"runs": runs, "summary": _summaries(runs)},
        "limitations": config["limitations"],
    }
    validate_public_report(report)
    report["resultSha256"] = canonical_sha256(report)
    return report


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["results"]["summary"]
    rows = [
        "| Model | Task | ROC-AUC | PR-AUC | LogLoss | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_key, model_label in (
        ("naive_independent", "Naive independent"),
        ("esmm", "ESMM"),
    ):
        for task_key, task_label in (
            ("ctr", "CTR"),
            ("ctcvr", "CTCVR"),
            ("postClickCvr", "Post-click CVR"),
        ):
            task = summary[model_key]["tasks"][task_key]
            formatted = []
            for metric in ("rocAuc", "prAuc", "logLoss", "brierScore", "ece"):
                value = task[metric]
                formatted.append(
                    "N/A"
                    if value["mean"] is None
                    else f"{value['mean']:.6f} ± {value['std']:.6f}"
                )
            rows.append(f"| {model_label} | {task_label} | " + " | ".join(formatted) + " |")
    counts = report["protocol"]["countsBySplit"]
    naive = summary["naive_independent"]["tasks"]
    esmm = summary["esmm"]["tasks"]

    def _delta(task: str, metric: str) -> float:
        return float(esmm[task][metric]["mean"]) - float(naive[task][metric]["mean"])

    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return "\n".join(
        [
            "# Criteo Attribution CVR / ESMM public benchmark V1",
            "",
            f"Status: `{report['status']}`; data origin: `public`; online claim: `false`.",
            "",
            "## Frozen protocol",
            "",
            f"- Rows: {counts['train']['rows']:,} train / {counts['dev']['rows']:,} dev / {counts['test']['rows']:,} test, in timestamp order.",
            f"- Test labels: {counts['test']['clickRows']:,} clicks; {counts['test']['ctcvrRows']:,} click-and-conversion rows; {counts['test']['viewThroughConversionRowsExcluded']:,} non-click conversion rows excluded from CTCVR.",
            "- Naive baseline: independent CTR on all impressions plus CVR on clicked impressions only.",
            "- ESMM: joint entire-space CTR and CTCVR supervision; inferred CVR is evaluated only on clicked test rows.",
            "- Preprocessing is fitted on train only; model/epoch selection uses dev only; test is scored once by the frozen protocol.",
            f"- Seeds: {', '.join(str(seed) for seed in report['protocol']['seeds'])}.",
            f"- Config SHA-256: `{report['evidence']['configSha256']}`.",
            f"- Split SHA-256: `{report['evidence']['splitSha256']}`.",
            f"- Result SHA-256: `{report['resultSha256']}`.",
            "",
            "## Test metrics — mean ± population std",
            "",
            *rows,
            "",
            "## Observed comparison",
            "",
            f"- CTCVR: ESMM changes ROC-AUC by `{_delta('ctcvr', 'rocAuc'):+.6f}`, PR-AUC by `{_delta('ctcvr', 'prAuc'):+.6f}`, and LogLoss by `{_delta('ctcvr', 'logLoss'):+.6f}` versus the independent baseline.",
            f"- Post-click CVR: ESMM changes ROC-AUC by `{_delta('postClickCvr', 'rocAuc'):+.6f}`, PR-AUC by `{_delta('postClickCvr', 'prAuc'):+.6f}`, and LogLoss by `{_delta('postClickCvr', 'logLoss'):+.6f}`.",
            f"- CTR: ESMM changes ROC-AUC by `{_delta('ctr', 'rocAuc'):+.6f}`, PR-AUC by `{_delta('ctr', 'prAuc'):+.6f}`, and LogLoss by `{_delta('ctr', 'logLoss'):+.6f}`; the CTR task does not improve in this run.",
            "",
            "The conversion-task gains and CTR regression are both retained. These multi-seed means are descriptive; this benchmark does not claim statistical significance or causal online lift.",
            "",
            "These are descriptive public offline results. A metric improvement, if any, is not an online CTR/CVR or revenue claim.",
            "",
            "## Label boundary",
            "",
            "The publisher's raw `conversion` label can include a conversion within 30 days even when the current impression was not clicked. This experiment defines CTCVR as `click × conversion`; it does not relabel view-through conversions as post-click outcomes.",
            "",
            "## Limitations",
            "",
            limitations,
            "",
        ]
    )

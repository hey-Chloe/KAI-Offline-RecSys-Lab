from __future__ import annotations

import csv
import gzip
from pathlib import Path

from kai_recsys_lab.pipelines.cvr_esmm import (
    ATTRIBUTION_COLUMNS,
    CATEGORICAL_NAMES,
    EXCLUDED_OUTCOME_OR_IDENTITY_FEATURES,
    TrainFittedAttributionPreprocessor,
    fixed_temporal_split,
    load_criteo_attribution_impressions,
    run_cvr_esmm_experiment,
)


def _write_attribution_fixture(path: Path, *, rows: int = 600) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ATTRIBUTION_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for index in range(rows):
            clicked = int(index % 3 == 0)
            converted = int(index % 21 == 0)
            writer.writerow(
                {
                    "timestamp": index,
                    "uid": index % 47,
                    "campaign": index % 7,
                    "conversion": converted,
                    "conversion_timestamp": index + 5 if converted else -1,
                    "conversion_id": index // 21 if converted else -1,
                    "attribution": converted,
                    "click": clicked,
                    "click_pos": 0 if clicked else -1,
                    "click_nb": 1 if clicked else -1,
                    "cost": 0.01 + (index % 11) / 100,
                    "cpo": 0.5 if converted else 0.0,
                    "time_since_last_click": index % 31 if index % 5 else -1,
                    **{f"cat{offset}": (index + offset) % 13 for offset in range(1, 10)},
                }
            )


def _config(path: Path) -> dict:
    return {
        "experimentId": "test-criteo-attribution-cvr-esmm",
        "seeds": [3407, 6502, 9109],
        "source": {
            "id": "criteo-attribution-modeling-bidding",
            "officialUrl": "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/",
            "terms": "CC BY-NC-SA 4.0 noncommercial offline research only.",
        },
        "dataset": {
            "rawFile": str(path),
            "rowLimit": 600,
            "samplingRule": "complete synthetic unit fixture representing the public schema",
        },
        "split": {"trainFraction": 0.7, "devFraction": 0.15},
        "preprocessing": {
            "minCategoryCount": 1,
            "maxCategoriesPerFeature": 100,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batchSize": 256,
            "learningRate": 0.001,
            "weightDecay": 0.0,
        },
        "models": {
            "independent": {"embeddingDim": 2, "hiddenDims": [8]},
            "esmm": {"embeddingDim": 2, "hiddenDims": [8]},
        },
        "limitations": ["synthetic unit fixture only"],
    }


def test_loader_constructs_ctcvr_without_relabeling_non_click_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attribution.tsv.gz"
    _write_attribution_fixture(path)
    frame = load_criteo_attribution_impressions(path, row_limit=600)
    assert frame["timestamp"].is_monotonic_increasing
    assert (frame["ctcvr"] == frame["click"] * frame["conversion"]).all()


def test_temporal_split_and_train_only_oov_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "attribution.tsv.gz"
    _write_attribution_fixture(path)
    frame = load_criteo_attribution_impressions(path, row_limit=600)
    first = fixed_temporal_split(frame, train_fraction=0.7, dev_fraction=0.15)
    second = fixed_temporal_split(frame, train_fraction=0.7, dev_fraction=0.15)
    assert (len(first.train), len(first.dev), len(first.test)) == (420, 90, 90)
    assert first.digest == second.digest

    preprocessor = TrainFittedAttributionPreprocessor(
        min_category_count=1,
        max_categories_per_feature=100,
    ).fit(first.train)
    unseen = first.dev.iloc[:1].copy()
    for name in CATEGORICAL_NAMES:
        unseen[name] = 999999999
    transformed = preprocessor.transform(unseen)
    assert transformed.categorical.eq(0).all()
    assert set(EXCLUDED_OUTCOME_OR_IDENTITY_FEATURES).isdisjoint(
        preprocessor.schema.numeric_names + preprocessor.schema.categorical_names
    )


def test_public_pipeline_compares_naive_and_esmm_on_all_funnel_tasks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attribution.tsv.gz"
    _write_attribution_fixture(path)
    report = run_cvr_esmm_experiment(_config(path))

    assert report["status"] == "COMPLETE"
    assert report["dataOrigin"] == "public"
    assert report["claimableOnlinePerformance"] is False
    assert report["protocol"]["countsBySplit"]["test"] == {
        "rows": 90,
        "clickRows": 30,
        "rawConversionRows": 4,
        "ctcvrRows": 4,
        "viewThroughConversionRowsExcluded": 0,
    }
    assert report["protocol"]["testDataUsedForSelection"] is False
    assert set(report["results"]["summary"]) == {"naive_independent", "esmm"}
    assert len(report["results"]["runs"]) == 6
    for run in report["results"]["runs"]:
        assert set(run["testMetrics"]) == {"ctr", "ctcvr", "postClickCvr"}
        for task in run["testMetrics"].values():
            assert set(task) >= {
                "nExamples",
                "positiveRows",
                "rocAuc",
                "prAuc",
                "logLoss",
                "brierScore",
                "ece",
            }

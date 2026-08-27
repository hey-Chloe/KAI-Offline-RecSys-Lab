from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from kai_recsys_lab.pipelines.criteo_ctr_scale import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    ScalableCriteoPreprocessor,
    SparseLinearCTR,
)


def _frame(rows: int, *, category: str | None = None) -> pd.DataFrame:
    payload: dict[str, object] = {"label": [index % 2 for index in range(rows)]}
    payload.update({name: np.arange(rows) + offset for offset, name in enumerate(RAW_NUMERIC_COLUMNS)})
    payload.update({
        name: [category or f"value-{(index + offset) % 3}" for index in range(rows)]
        for offset, name in enumerate(RAW_CATEGORICAL_COLUMNS)
    })
    return pd.DataFrame(payload)


def test_scale_preprocessor_is_train_fitted_and_oov_safe() -> None:
    preprocessor = ScalableCriteoPreprocessor(min_category_count=1, max_categories_per_feature=10).fit(_frame(12))
    transformed = preprocessor.transform(_frame(2, category="dev-only"))
    assert transformed.numeric.shape == (2, 13)
    assert transformed.categorical.shape == (2, 26)
    assert torch.count_nonzero(transformed.categorical) == 0


def test_sparse_linear_ctr_accepts_array_native_batch() -> None:
    preprocessor = ScalableCriteoPreprocessor(min_category_count=1, max_categories_per_feature=10).fit(_frame(12))
    batch = preprocessor.transform(_frame(4))
    logits = SparseLinearCTR(preprocessor.schema)(batch)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()

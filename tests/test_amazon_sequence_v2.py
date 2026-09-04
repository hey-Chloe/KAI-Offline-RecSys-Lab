from __future__ import annotations

import numpy as np

from kai_recsys_lab.pipelines.amazon_sequence_v2 import initialize_item_embeddings
from kai_recsys_lab.sequence.models import DinSequenceScorer, MeanPoolingSequenceScorer


def test_pretrained_sequence_initialization_preserves_padding_and_trainability() -> None:
    values = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    values[0] = 99
    model = DinSequenceScorer(12, 4, 8, 4)
    initialize_item_embeddings(model, values, trainable=False)
    assert model.item_embedding.weight.requires_grad is False
    assert model.item_embedding.weight[0].detach().numpy().tolist() == [0.0] * 4
    assert model.item_embedding.weight[2].detach().numpy().tolist() == values[2].tolist()


def test_pretrained_sequence_initialization_rejects_shape_drift() -> None:
    model = MeanPoolingSequenceScorer(12, 4, 8)
    try:
        initialize_item_embeddings(model, np.zeros((11, 4), dtype=np.float32), trainable=True)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("shape drift must fail closed")

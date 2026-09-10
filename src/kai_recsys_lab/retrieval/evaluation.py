from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from kai_recsys_lab.evaluation import retrieval_metrics_at_k


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    k: int
    recall: float
    hit_rate: float
    mrr: float
    ndcg: float
    query_count: int


def evaluate_rankings(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    relevant_items: Mapping[Hashable, set[Hashable] | frozenset[Hashable]],
    ks: Sequence[int],
) -> tuple[RankingMetrics, ...]:
    """Macro metrics for binary relevance without sampled-negative shortcuts."""

    if not relevant_items:
        raise ValueError("at least one evaluation query is required")
    if any(k < 1 for k in ks):
        raise ValueError("all k values must be positive")
    if any(not truth for truth in relevant_items.values()):
        raise ValueError("every query must have at least one relevant item")

    results: list[RankingMetrics] = []
    for k in ks:
        per_query = []
        for query_id, truth in relevant_items.items():
            ranked = list(recommendations.get(query_id, ()))[:k]
            per_query.append(retrieval_metrics_at_k(ranked, truth, k))
        count = len(relevant_items)
        results.append(
            RankingMetrics(
                k,
                sum(metric.recall for metric in per_query) / count,
                sum(metric.hit_rate for metric in per_query) / count,
                sum(metric.mrr for metric in per_query) / count,
                sum(metric.ndcg for metric in per_query) / count,
                count,
            )
        )
    return tuple(results)

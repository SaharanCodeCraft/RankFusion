"""
Evaluation metrics for the Cranfield Information Retrieval project.

Implements Precision@K, Recall, Average Precision, MAP, and nDCG.
"""

import math


def precision_at_k(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k: int = 10,
) -> float:
    """Calculate Precision@K."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_document_ids:
        return 0.0

    top_k = ranked_document_ids[:k]

    relevant_retrieved = sum(
        1 for document_id in top_k
        if document_id in relevant_document_ids
    )

    return relevant_retrieved / k


def recall(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
) -> float:
    """Calculate recall over the supplied ranked results."""

    if not relevant_document_ids:
        return 0.0

    relevant_retrieved = len(
        set(ranked_document_ids) & relevant_document_ids
    )

    return relevant_retrieved / len(relevant_document_ids)


def average_precision(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
) -> float:
    """Calculate Average Precision for one query."""

    if not relevant_document_ids:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0

    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_document_ids:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    return precision_sum / len(relevant_document_ids)


def mean_average_precision(
    ranked_results_by_query: dict[str, list[str]],
    relevant_by_query: dict[str, set[str]],
) -> float:
    """Calculate Mean Average Precision across queries."""

    if not relevant_by_query:
        return 0.0

    average_precisions = []

    for query_id, relevant_document_ids in relevant_by_query.items():
        ranked_document_ids = ranked_results_by_query.get(query_id, [])

        average_precisions.append(
            average_precision(
                ranked_document_ids,
                relevant_document_ids,
            )
        )

    return sum(average_precisions) / len(average_precisions)


def ndcg_at_k(
    ranked_document_ids: list[str],
    relevance_by_document: dict[str, int],
    k: int = 10,
) -> float:
    """Calculate nDCG@K using graded relevance judgments."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    def gain(relevance: int) -> float:
        # Negative relevance judgments contribute no positive gain.
        return (2 ** max(relevance, 0) - 1)

    actual_dcg = 0.0

    for rank, document_id in enumerate(ranked_document_ids[:k], start=1):
        relevance = relevance_by_document.get(document_id, 0)
        actual_dcg += gain(relevance) / math.log2(rank + 1)

    ideal_relevances = sorted(
        (
            max(relevance, 0)
            for relevance in relevance_by_document.values()
        ),
        reverse=True,
    )[:k]

    ideal_dcg = sum(
        gain(relevance) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg
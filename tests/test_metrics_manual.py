from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    precision_at_k,
    recall,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
)


def main():
    # Example ranking containing relevant and irrelevant documents.
    ranked_ids = ["D1", "D2", "D3", "D4", "D5"]
    relevant_ids = {"D1", "D3", "D5"}

    print("Testing evaluation metrics...\n")

    p_at_5 = precision_at_k(ranked_ids, relevant_ids, k=5)
    print(f"Precision@5: {p_at_5:.4f}")

    recall_score = recall(ranked_ids, relevant_ids)
    print(f"Recall: {recall_score:.4f}")

    ap_score = average_precision(ranked_ids, relevant_ids)
    print(f"Average Precision: {ap_score:.4f}")

    rankings = {
        "Q1": ranked_ids,
        "Q2": ["D3", "D1", "D2"],
    }

    relevance = {
        "Q1": relevant_ids,
        "Q2": {"D1", "D3"},
    }

    map_score = mean_average_precision(rankings, relevance)
    print(f"Mean Average Precision: {map_score:.4f}")

    graded_relevance = {
        "D1": 3,
        "D2": 0,
        "D3": 2,
        "D4": 0,
        "D5": 1,
    }

    ndcg_score = ndcg_at_k(ranked_ids, graded_relevance, k=5)
    print(f"nDCG@5: {ndcg_score:.4f}")

    assert abs(p_at_5 - 0.6) < 1e-9
    assert abs(recall_score - 1.0) < 1e-9
    assert abs(ap_score - 0.7555555556) < 1e-8
    assert 0.0 <= map_score <= 1.0
    assert 0.0 <= ndcg_score <= 1.0

    print("\nAll evaluation metric tests passed!")


if __name__ == "__main__":
    main()

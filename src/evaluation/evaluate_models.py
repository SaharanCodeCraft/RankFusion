"""
Evaluate all retrieval models using Cranfield relevance judgments.

Models:
1. TF-IDF
2. BM25
3. Sentence Transformer
4. Hybrid RRF

Metrics:
- Precision@10
- Recall
- Mean Average Precision (MAP)
- nDCG@10
"""

import json
import csv
from pathlib import Path

from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.sentence_transformer import SentenceTransformerRetriever
from src.retrieval.hybrid import HybridRetriever

from src.evaluation.metrics import (
    precision_at_k,
    recall,
    average_precision,
    ndcg_at_k,
)
from src.visualization.plot_precision_recall import plot_precision_recall


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUERIES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "queries_processed.jsonl"
)

QRELS_PATH = PROJECT_ROOT / "data" / "raw" / "qrels.tsv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"

K = 10


def load_queries() -> list[dict]:
    """Load the processed Cranfield queries."""

    queries = []

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            queries.append(json.loads(line))

    if not queries:
        raise ValueError("No queries were loaded.")

    return queries


def load_qrels() -> dict[str, dict[str, int]]:
    """Load graded relevance judgments from the Cranfield qrels file."""

    relevance_by_query: dict[str, dict[str, int]] = {}

    with QRELS_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        required_columns = {"query_id", "document_id", "relevance"}

        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "The qrels file must contain query_id, document_id, "
                "and relevance columns."
            )

        for row in reader:
            query_id = str(row["query_id"])
            document_id = str(row["document_id"])
            relevance = int(row["relevance"])

            relevance_by_query.setdefault(query_id, {})[
                document_id
            ] = relevance

    return relevance_by_query


def evaluate_one_model(
    model_name: str,
    model,
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
) -> dict:
    """Evaluate one retrieval model across all queries."""

    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    ndcg_scores = []
    precision_recall_data = []

    for query_number, query_data in enumerate(queries, start=1):
        query_id = str(query_data["query_id"])
        query_text = query_data["processed_text"]

        results = model.search(query_text, top_k=K)

        ranked_document_ids = [
            str(result["document_id"]) for result in results
        ]
        precision_recall_data.append(
            {
        "query_id": query_id,
        "ranked_document_ids": ranked_document_ids,
        }
        )

        query_relevance = qrels.get(query_id, {})

        # Explicit convention: positive judgments count as relevant
        # for binary metrics. Negative judgments are not relevant.
        relevant_document_ids = {
            document_id
            for document_id, relevance in query_relevance.items()
            if relevance > 0
        }

        precision_scores.append(
            precision_at_k(
                ranked_document_ids,
                relevant_document_ids,
                k=K,
            )
        )

        recall_scores.append(
            recall(
                ranked_document_ids,
                relevant_document_ids,
            )
        )

        average_precision_scores.append(
            average_precision(
                ranked_document_ids,
                relevant_document_ids,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                ranked_document_ids,
                query_relevance,
                k=K,
            )
        )

        if query_number % 25 == 0 or query_number == len(queries):
            print(
                f"{model_name}: evaluated "
                f"{query_number}/{len(queries)} queries"
            )

    number_of_queries = len(queries)

    return {
        "model": model_name,
        "queries_evaluated": number_of_queries,
        "precision_at_10": sum(precision_scores) / number_of_queries,
        "recall": sum(recall_scores) / number_of_queries,
        "map": sum(average_precision_scores) / number_of_queries,
        "ndcg_at_10": sum(ndcg_scores) / number_of_queries,
        "precision_recall_data": precision_recall_data,
    }


def main() -> None:
    """Run the complete retrieval model evaluation."""

    if not QUERIES_PATH.is_file():
        raise FileNotFoundError(
            f"Processed queries not found: {QUERIES_PATH}"
        )

    if not QRELS_PATH.is_file():
        raise FileNotFoundError(f"Qrels file not found: {QRELS_PATH}")

    queries = load_queries()
    qrels = load_qrels()

    print("=" * 65)
    print("CRANFIELD RETRIEVAL MODEL EVALUATION")
    print("=" * 65)
    print(f"Queries: {len(queries)}")
    print(f"Queries with relevance judgments: {len(qrels)}")
    print(f"Evaluation cutoff: {K}")
    print("Binary relevance convention: relevance > 0")
    print("nDCG uses the original graded relevance judgments.")
    print()

    models = [
        ("TF-IDF", TFIDFRetriever()),
        ("BM25", BM25Retriever()),
        ("Sentence Transformer", SentenceTransformerRetriever()),
        ("Hybrid RRF", HybridRetriever()),
    ]

    evaluation_results = []
    precision_recall_results = {}

    for model_name, model in models:
        print("\n" + "-" * 65)
        print(f"Evaluating: {model_name}")
        print("-" * 65)

        # HybridRetriever.fit() initializes its three component models.
        # The other models each initialize their own index.
        model.fit()

        result = evaluate_one_model(
            model_name,
            model,
            queries,
            qrels,
        )

        evaluation_results.append(result)
        precision_recall_results[model_name] = result.pop(
            "precision_recall_data"
            )

        print(
            f"\n{model_name} results:\n"
            f"Precision@10: {result['precision_at_10']:.4f}\n"
            f"Recall:       {result['recall']:.4f}\n"
            f"MAP:          {result['map']:.4f}\n"
            f"nDCG@10:      {result['ndcg_at_10']:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "model_comparison.csv"

    fieldnames = [
        "model",
        "queries_evaluated",
        "precision_at_10",
        "recall",
        "map",
        "ndcg_at_10",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_results)

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        f"{'Model':<24}"
        f"{'P@10':>10}"
        f"{'Recall':>10}"
        f"{'MAP':>10}"
        f"{'nDCG@10':>11}"
    )

    for result in evaluation_results:
        print(
            f"{result['model']:<24}"
            f"{result['precision_at_10']:>10.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['map']:>10.4f}"
            f"{result['ndcg_at_10']:>11.4f}"
        )

    print(f"\nResults saved to: {output_path}")
    plot_precision_recall(
        precision_recall_results,
        qrels,
        )


if __name__ == "__main__":
    main()
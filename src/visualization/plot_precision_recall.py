"""Plot precision-recall curves for the Cranfield retrieval models."""

from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graphs"


def calculate_precision_recall(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
) -> tuple[list[float], list[float]]:
    """Calculate precision and recall at each ranking position."""

    precisions = []
    recalls = []
    relevant_found = 0
    total_relevant = len(relevant_document_ids)

    if total_relevant == 0:
        return precisions, recalls

    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_document_ids:
            relevant_found += 1

        precisions.append(relevant_found / rank)
        recalls.append(relevant_found / total_relevant)

    return precisions, recalls


def plot_precision_recall(
    precision_recall_results: dict,
    qrels: dict,
) -> None:
    """Generate macro-averaged precision-recall curves for all models."""

    import numpy as np

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recall_levels = np.linspace(0.0, 1.0, 101)
    plt.figure(figsize=(10, 7))

    for model_name, query_results in precision_recall_results.items():
        query_curves = []

        for query_result in query_results:
            query_id = query_result["query_id"]
            ranked_ids = query_result["ranked_document_ids"]

            relevant_ids = {
                document_id
                for document_id, relevance in qrels.get(
                    query_id, {}
                ).items()
                if relevance > 0
            }

            if not relevant_ids:
                continue

            precisions, recalls = calculate_precision_recall(
                ranked_ids,
                relevant_ids,
            )

            if not precisions:
                continue

            # Interpolate precision at each recall level.
            interpolated_precision = []

            for recall_level in recall_levels:
                eligible_precisions = [
                    precision
                    for precision, recall_value in zip(
                        precisions, recalls
                    )
                    if recall_value >= recall_level
                ]

                interpolated_precision.append(
                    max(eligible_precisions, default=0.0)
                )

            query_curves.append(interpolated_precision)

        if not query_curves:
            continue

        # Average precision across queries at each recall level.
        mean_precision = np.mean(query_curves, axis=0)

        plt.plot(
            recall_levels,
            mean_precision,
            label=model_name,
            linewidth=2,
        )

    plt.xlabel("Recall")
    plt.ylabel("Interpolated Precision")
    plt.title("Macro-Averaged Precision-Recall Curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "precision_recall_curve.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf()
    fig.savefig(
        output_path,
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Precision-recall curve saved to: {output_path}")
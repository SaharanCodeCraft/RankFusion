"""
Generate comparison graphs for Cranfield retrieval models.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_comparison.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graphs"
REPORT_FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

METRICS = {
    "precision_at_10": "Precision@10",
    "recall": "Recall",
    "map": "Mean Average Precision (MAP)",
    "ndcg_at_10": "nDCG@10",
}


def load_results() -> list[dict]:
    """Load model evaluation results from the CSV file."""

    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}\n"
            "Run the model evaluation first."
        )

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        results = list(reader)

    if not results:
        raise ValueError("The results CSV contains no model results.")

    required_columns = {"model", *METRICS.keys()}

    if not reader.fieldnames or not required_columns.issubset(
        reader.fieldnames
    ):
        raise ValueError(
            "The results CSV is missing one or more required columns."
        )

    for row in results:
        for metric in METRICS:
            row[metric] = float(row[metric])

    return results


def create_graphs(results: list[dict]) -> None:
    """Create and save one comparison graph for each evaluation metric."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_names = [row["model"] for row in results]

    for metric_key, metric_title in METRICS.items():
        values = [row[metric_key] for row in results]

        fig, ax = plt.subplots(figsize=(9, 5))

        bars = ax.bar(model_names, values)

        ax.set_title(f"Retrieval Model Comparison: {metric_title}")
        ax.set_xlabel("Retrieval Model")
        ax.set_ylabel(metric_title)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax.bar_label(bars, fmt="%.4f", padding=3)

        fig.tight_layout()

        filename = f"{metric_key}_comparison.png"

        output_path = OUTPUT_DIR / filename
        report_path = REPORT_FIGURES_DIR / filename

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(report_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved: {output_path}")
        print(f"Saved: {report_path}")


def main() -> None:
    """Load the evaluation results and generate all comparison graphs."""

    print("Loading retrieval model evaluation results...")

    results = load_results()

    print(f"Models found: {len(results)}")
    print("\nGenerating comparison graphs...")

    create_graphs(results)

    print("\nAll comparison graphs generated successfully.")


if __name__ == "__main__":
    main()
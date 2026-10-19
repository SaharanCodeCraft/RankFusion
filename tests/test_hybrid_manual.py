from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever


def main():
    model = HybridRetriever()

    print("Building the hybrid retrieval system...")
    model.fit()

    query = "aerodynamic performance of aircraft wings"

    print(f"\nSearch query: {query}")
    results = model.search(query, top_k=5)

    print("\nTop 5 hybrid search results:")

    for result in results:
        print(
            "Rank {} | Document {} | RRF Score {:.4f}".format(
                result["rank"],
                result["document_id"],
                result["score"],
            )
        )

    print("\nHybrid retrieval test completed successfully!")


if __name__ == "__main__":
    main()
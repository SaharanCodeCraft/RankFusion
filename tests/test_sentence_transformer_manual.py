from pathlib import Path
import sys

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.sentence_transformer import (
    SentenceTransformerRetriever,
)


def main():
    model = SentenceTransformerRetriever()

    print("Loading documents and building semantic search model...")
    model.fit()

    query = "aerodynamic performance of aircraft wings"

    print(f"\nSearch query: {query}")
    results = model.search(query, top_k=5)

    print("\nTop 5 search results:")

    for result in results:
        print(
            "Rank {} | Document {} | Score {:.4f}".format(
                result["rank"],
                result["document_id"],
                result["score"],
            )
        )


if __name__ == "__main__":
    main()
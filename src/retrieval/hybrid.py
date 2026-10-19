"""
Hybrid retrieval using Reciprocal Rank Fusion (RRF).

Combines ranked results from TF-IDF, BM25, and Sentence Transformers.
"""

from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.sentence_transformer import SentenceTransformerRetriever


class HybridRetriever:
    """Combine multiple retrieval models using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer.")

        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.rrf_k = rrf_k

        self.tfidf_model = TFIDFRetriever()
        self.bm25_model = BM25Retriever()
        self.semantic_model = SentenceTransformerRetriever()

        self.is_fitted = False

    def fit(self) -> None:
        """Build the indexes for all three retrieval models."""

        print("\nBuilding TF-IDF index...")
        self.tfidf_model.fit()

        print("\nBuilding BM25 index...")
        self.bm25_model.fit()

        print("\nBuilding semantic index...")
        self.semantic_model.fit()

        self.is_fitted = True

        print("\nHybrid retrieval models are ready.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 100,
    ) -> list[dict]:
        """Combine the rankings from all three models using RRF."""

        if not self.is_fitted:
            raise RuntimeError(
                "The hybrid model has not been fitted. Call fit() first."
            )

        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
            raise TypeError("candidate_k must be an integer.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        # Retrieve candidate rankings from each model.
        tfidf_results = self.tfidf_model.search(query, candidate_k)
        bm25_results = self.bm25_model.search(query, candidate_k)
        semantic_results = self.semantic_model.search(query, candidate_k)

        # Store each document's accumulated RRF score.
        rrf_scores: dict[str, float] = {}
        document_details: dict[str, dict] = {}

        model_results = [
            tfidf_results,
            bm25_results,
            semantic_results,
        ]

        for results in model_results:
            for rank, result in enumerate(results, start=1):
                document_id = result["document_id"]

                rrf_scores[document_id] = (
                    rrf_scores.get(document_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

                # Keep the document text for displaying search results.
                document_details[document_id] = result

        # Rank documents by their combined RRF scores.
        ranked_document_ids = sorted(
            rrf_scores,
            key=lambda document_id: (
                -rrf_scores[document_id],
                document_id,
            ),
        )[:top_k]

        final_results = []

        for rank, document_id in enumerate(
            ranked_document_ids,
            start=1,
        ):
            final_results.append(
                {
                    "rank": rank,
                    "document_id": document_id,
                    "title": document_details[document_id].get(
                        "title", "Untitled document"
                        ),
                    "score": float(rrf_scores[document_id]),
                    "text": document_details[document_id]["text"],
                }
            )

        return final_results
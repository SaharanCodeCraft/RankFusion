"""
BM25 retrieval model for the Cranfield Information Retrieval project.
"""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.preprocessing.text_preprocessor import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents_processed.jsonl"
)


class BM25Retriever:
    """Retrieve and rank documents using the BM25 algorithm."""

    def __init__(self) -> None:
        self.document_ids: list[str] = []
        self.document_texts: list[str] = []
        self.document_titles: list[str] = []
        self.tokenized_documents: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.is_fitted = False

    def fit(self, documents_path: Path = DOCUMENTS_PATH) -> None:
        """Load processed documents and build the BM25 index."""

        documents_path = Path(documents_path)

        if not documents_path.is_file():
            raise FileNotFoundError(
                f"Processed documents not found: {documents_path}\n"
                "Run the preprocessing pipeline first."
            )

        document_ids = []
        document_texts = []
        document_titles = []
        original_document_texts = []
        tokenized_documents = []

        with documents_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}."
                    ) from exc

                document_id = document.get("document_id")
                processed_text = document.get("processed_text", "")

                if not document_id:
                    raise ValueError(
                        f"Missing document_id on line {line_number}."
                    )

                if not isinstance(processed_text, str):
                    raise ValueError(
                        f"Invalid processed_text for document {document_id}."
                    )

                document_ids.append(str(document_id))
                document_texts.append(processed_text)
                document_titles.append(document.get("title", "Untitled document"))
                original_document_texts.append(document.get("text", ""))
                tokenized_documents.append(processed_text.split())

        if not document_ids:
            raise ValueError("No documents were loaded.")

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Duplicate document IDs found.")

        # BM25 requires at least one document containing tokens.
        if not any(tokenized_documents):
            raise ValueError(
                "All documents are empty after preprocessing."
            )

        self.bm25 = BM25Okapi(tokenized_documents)
        self.document_ids = document_ids
        self.document_texts = document_texts
        self.document_titles = document_titles
        self.original_document_texts = original_document_texts
        self.tokenized_documents = tokenized_documents
        self.is_fitted = True

        print("BM25 model fitted successfully.")
        print(f"Documents indexed: {len(self.document_ids)}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return the top-k documents ranked by BM25 score."""

        if not self.is_fitted or self.bm25 is None:
            raise RuntimeError(
                "The model has not been fitted. Call fit() first."
            )

        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = preprocess_text(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Ignore documents with zero or negative scores.
        ranked_indices = sorted(
            (
                index
                for index, score in enumerate(scores)
                if score > 0
            ),
            key=lambda index: (-scores[index], index),
        )[:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "document_id": self.document_ids[index],
                    "title": self.document_titles[index],
                    "score": float(scores[index]),
                    "text": self.original_document_texts[index],
                }
            )

        return results
"""
Semantic retrieval model using Sentence Transformers.

Encodes documents and queries into dense vectors and ranks
documents by cosine similarity.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_preprocessor import preprocess_to_string


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents_processed.jsonl"
)

# A compact, widely used model for semantic text similarity.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerRetriever:
    """Retrieve documents using semantic vector similarity."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = None
        self.document_ids: list[str] = []
        self.document_texts: list[str] = []
        self.document_titles: list[str] = []
        self.document_embeddings = None
        self.is_fitted = False

    def fit(self, documents_path: Path = DOCUMENTS_PATH) -> None:
        """Load documents and generate their semantic embeddings."""

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

        if not document_ids:
            raise ValueError("No documents were loaded.")

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Duplicate document IDs found.")

        # Keep empty documents in the collection but encode only nonempty ones.
        nonempty_indices = [
            index
            for index, text in enumerate(document_texts)
            if text.strip()
        ]

        if not nonempty_indices:
            raise ValueError(
                "All documents are empty after preprocessing."
            )

        nonempty_texts = [
            document_texts[index] for index in nonempty_indices
        ]

        print(f"Loading semantic model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print("Encoding documents. This may take a little time...")

        nonempty_embeddings = self.model.encode(
            nonempty_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Give empty documents zero vectors so document IDs remain aligned.
        embedding_dimension = nonempty_embeddings.shape[1]

        embeddings = np.zeros(
            (len(document_ids), embedding_dimension),
            dtype=np.float32,
        )

        for embedding_index, document_index in enumerate(nonempty_indices):
            embeddings[document_index] = nonempty_embeddings[embedding_index]

        self.document_ids = document_ids
        self.document_texts = document_texts
        self.document_titles = document_titles
        self.original_document_texts = original_document_texts
        self.document_embeddings = embeddings
        self.is_fitted = True

        print("Sentence Transformer model fitted successfully.")
        print(f"Documents indexed: {len(self.document_ids)}")
        print(f"Embedding matrix shape: {self.document_embeddings.shape}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return the top-k documents ranked by semantic similarity."""

        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "The model has not been fitted. Call fit() first."
            )

        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        processed_query = preprocess_to_string(query)

        if not processed_query.strip():
            return []

        query_embedding = self.model.encode(
            [processed_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_embedding,
            self.document_embeddings,
        ).ravel()

        # Exclude empty documents from search results.
        ranked_indices = sorted(
            (
                index
                for index, text in enumerate(self.document_texts)
                if text.strip()
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
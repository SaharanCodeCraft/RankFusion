"""TF-IDF retrieval model for the Cranfield IR project."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_preprocessor import preprocess_to_string


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "documents_processed.jsonl"
)


class TFIDFRetriever:
    """Retrieve and rank Cranfield documents using TF-IDF."""

    def __init__(self) -> None:
        self.document_ids: list[str] = []
        self.document_texts: list[str] = []
        self.document_titles: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.document_matrix = None
        self.is_fitted = False

    def fit(self, documents_path: Path = DOCUMENTS_PATH) -> None:
        """Load processed documents and build the TF-IDF matrix."""
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
                        f"Invalid JSON on line {line_number} "
                        f"of {documents_path.name}."
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

        # Keep the document collection unchanged. Empty documents
        # receive zero vectors instead of being deleted.
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            norm="l2",
            dtype=np.float64,
        )

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

        nonempty_matrix = self.vectorizer.fit_transform(nonempty_texts)

        # Restore the full document order, including empty documents.
        from scipy.sparse import csr_matrix, vstack

        empty_matrix = csr_matrix(
            (1, nonempty_matrix.shape[1]),
            dtype=nonempty_matrix.dtype,
        )

        matrix_rows = []
        nonempty_row = 0

        for text in document_texts:
            if text.strip():
                matrix_rows.append(nonempty_matrix[nonempty_row])
                nonempty_row += 1
            else:
                matrix_rows.append(empty_matrix)

        self.document_matrix = vstack(
            matrix_rows,
            format="csr",
        )

        self.document_ids = document_ids
        self.document_texts = document_texts
        self.document_titles = document_titles
        self.original_document_texts = original_document_texts
        self.is_fitted = True

        print("TF-IDF model fitted successfully.")
        print(f"Documents indexed: {len(self.document_ids)}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Document matrix shape: {self.document_matrix.shape}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Return the top-k documents ranked by cosine similarity."""
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError(
                "The model has not been fitted. Call fit() first."
            )

        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        processed_query = preprocess_to_string(query)

        if not processed_query.strip():
            return []

        query_vector = self.vectorizer.transform([processed_query])

        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(
            query_vector,
            self.document_matrix,
        ).ravel()

        # Highest score first; use original position for stable ties.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )[: min(top_k, len(scores))]

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
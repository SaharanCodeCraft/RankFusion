"""Preprocess Cranfield documents and queries and save the results."""

import json
import sys
from pathlib import Path

from src.preprocessing.text_preprocessor import (
    preprocess_text,
    preprocess_to_string,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOCUMENTS_INPUT = RAW_DIR / "documents.jsonl"
QUERIES_INPUT = RAW_DIR / "queries.jsonl"

DOCUMENTS_OUTPUT = PROCESSED_DIR / "documents_processed.jsonl"
QUERIES_OUTPUT = PROCESSED_DIR / "queries_processed.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSON Lines file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name}, "
                    f"line {line_number}."
                )

            records.append(record)

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a UTF-8 JSON Lines file."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def preprocess_documents(documents: list[dict]) -> list[dict]:
    """Add processed text and tokens to every document."""
    processed_documents = []

    for document in documents:
        document_id = document.get("document_id")

        if not document_id:
            raise ValueError("A document is missing its document_id.")

        original_text = document.get("text", "")

        if not isinstance(original_text, str):
            raise ValueError(
                f"Document {document_id} has a non-string text field."
            )

        tokens = preprocess_text(original_text)

        processed_documents.append(
            {
                **document,
                "tokens": tokens,
                "processed_text": " ".join(tokens),
            }
        )

    return processed_documents


def preprocess_queries(queries: list[dict]) -> list[dict]:
    """Add processed text and tokens to every query."""
    processed_queries = []

    for query in queries:
        query_id = query.get("query_id")

        if not query_id:
            raise ValueError("A query is missing its query_id.")

        original_text = query.get("query_text", "")

        if not isinstance(original_text, str):
            raise ValueError(
                f"Query {query_id} has a non-string query_text field."
            )

        tokens = preprocess_text(original_text)

        if not tokens:
            raise ValueError(
                f"Query {query_id} produced no tokens after preprocessing."
            )

        processed_queries.append(
            {
                **query,
                "tokens": tokens,
                "processed_text": preprocess_to_string(original_text),
            }
        )

    return processed_queries


def run_preprocessing() -> bool:
    """Run preprocessing and save the resulting files."""
    print("=" * 60)
    print("CRANFIELD DATASET PREPROCESSING")
    print("=" * 60)

    try:
        documents = read_jsonl(DOCUMENTS_INPUT)
        queries = read_jsonl(QUERIES_INPUT)

        if len(documents) != 1400:
            raise ValueError(
                f"Expected 1400 documents, found {len(documents)}."
            )

        if len(queries) != 225:
            raise ValueError(
                f"Expected 225 queries, found {len(queries)}."
            )

        print(f"Loaded documents: {len(documents)}")
        print(f"Loaded queries: {len(queries)}")

        processed_documents = preprocess_documents(documents)
        processed_queries = preprocess_queries(queries)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        write_jsonl(DOCUMENTS_OUTPUT, processed_documents)
        write_jsonl(QUERIES_OUTPUT, processed_queries)

        empty_documents = sum(
            not document["tokens"]
            for document in processed_documents
        )

        print(f"Processed documents: {len(processed_documents)}")
        print(f"Processed queries: {len(processed_queries)}")
        print(f"Documents with no tokens: {empty_documents}")

        print(f"\nSaved documents to:\n{DOCUMENTS_OUTPUT}")
        print(f"\nSaved queries to:\n{QUERIES_OUTPUT}")

        print("\nPREPROCESSING COMPLETED SUCCESSFULLY")
        return True

    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"\nPREPROCESSING FAILED: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = run_preprocessing()
    sys.exit(0 if success else 1)
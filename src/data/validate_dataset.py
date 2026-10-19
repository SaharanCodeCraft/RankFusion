"""Validate the exported Cranfield Information Retrieval dataset."""

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DOCUMENTS_PATH = RAW_DIR / "documents.jsonl"
QUERIES_PATH = RAW_DIR / "queries.jsonl"
QRELS_PATH = RAW_DIR / "qrels.tsv"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSON Lines file and validate every record."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name}, "
                    f"line {line_number}."
                )

            records.append(record)

    return records


def validate_dataset() -> bool:
    """Validate dataset files, counts, required fields, and ID references."""
    print("=" * 60)
    print("CRANFIELD DATASET VALIDATION")
    print("=" * 60)

    # Check that all required files exist.
    required_files = [
        DOCUMENTS_PATH,
        QUERIES_PATH,
        QRELS_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            print(f"FAIL: Missing file: {path}")
            return False

    print("\nPASS: All three dataset files exist.")

    try:
        documents = load_jsonl(DOCUMENTS_PATH)
        queries = load_jsonl(QUERIES_PATH)

        with QRELS_PATH.open(
            "r", encoding="utf-8", newline=""
        ) as file:
            reader = csv.DictReader(file, delimiter="\t")

            expected_columns = {
                "query_id",
                "document_id",
                "relevance",
            }

            if not reader.fieldnames or not expected_columns.issubset(
                set(reader.fieldnames)
            ):
                print("FAIL: The qrels file has an invalid header.")
                print(f"Found columns: {reader.fieldnames}")
                return False

            qrels = list(reader)

    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: Could not read dataset: {exc}")
        return False

    # Validate the expected record counts.
    expected_counts = {
        "Documents": (len(documents), 1400),
        "Queries": (len(queries), 225),
        "Relevance judgments": (len(qrels), 1837),
    }

    for label, (actual, expected) in expected_counts.items():
        if actual != expected:
            print(f"FAIL: {label}: expected {expected}, found {actual}")
            return False

        print(f"PASS: {label}: {actual}")

    # Validate required fields.
    for index, document in enumerate(documents, start=1):
        if not document.get("document_id"):
            print(f"FAIL: Document {index} has no document_id.")
            return False

        if not isinstance(document.get("text"), str):
            print(f"FAIL: Document {index} has invalid text.")
            return False

    for index, query in enumerate(queries, start=1):
        if not query.get("query_id"):
            print(f"FAIL: Query {index} has no query_id.")
            return False

        if not isinstance(query.get("query_text"), str):
            print(f"FAIL: Query {index} has invalid query_text.")
            return False

        if not query["query_text"].strip():
            print(f"FAIL: Query {index} has empty text.")
            return False

    print("PASS: Required document and query fields are valid.")

    # Check that IDs are unique.
    document_ids = [d["document_id"] for d in documents]
    query_ids = [q["query_id"] for q in queries]

    if len(document_ids) != len(set(document_ids)):
        print("FAIL: Duplicate document IDs found.")
        return False

    if len(query_ids) != len(set(query_ids)):
        print("FAIL: Duplicate query IDs found.")
        return False

    print("PASS: Document and query IDs are unique.")

    # Validate relevance judgment references and values.
    document_id_set = set(document_ids)
    query_id_set = set(query_ids)

    allowed_relevance_values = {"-1", "1", "2", "3", "4"}

    for index, row in enumerate(qrels, start=2):
        query_id = row.get("query_id", "")
        document_id = row.get("document_id", "")
        relevance = row.get("relevance", "")

        if query_id not in query_id_set:
            print(
                f"FAIL: Unknown query ID {query_id!r} "
                f"in qrels line {index}."
            )
            return False

        if document_id not in document_id_set:
            print(
                f"FAIL: Unknown document ID {document_id!r} "
                f"in qrels line {index}."
            )
            return False

        if relevance not in allowed_relevance_values:
            print(
                f"FAIL: Unexpected relevance value {relevance!r} "
                f"in qrels line {index}."
            )
            return False

    print("PASS: All relevance judgments reference existing IDs.")
    print("PASS: All relevance values are recognized.")

    # Report incomplete document text without changing the raw data.
    empty_text_documents = [
        d["document_id"]
        for d in documents
        if not d["text"].strip()
    ]

    print(
        "Documents with empty text:",
        len(empty_text_documents),
        empty_text_documents,
    )

    print("\n" + "=" * 60)
    print("DATASET VALIDATION PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
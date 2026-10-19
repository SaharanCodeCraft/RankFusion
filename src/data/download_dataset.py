"""
Download / export the Cranfield dataset into ``data/raw/``.

Uses the ``ir_datasets`` package to obtain the standard Cranfield collection
(1 400 documents, 225 queries, ~1 837 relevance judgements) and writes three
clean, machine-readable files:

    data/raw/documents.jsonl   — one JSON object per line
    data/raw/queries.jsonl     — one JSON object per line
    data/raw/qrels.tsv         — tab-separated values

The script is idempotent: if the three files already exist and pass a quick
size sanity-check the download is skipped.

Usage
-----
    python -m src.data.download_dataset          (from project root)
    python src/data/download_dataset.py          (standalone)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is importable when run as a standalone script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Paths relative to project root
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DOCS_PATH = RAW_DIR / "documents.jsonl"
QUERIES_PATH = RAW_DIR / "queries.jsonl"
QRELS_PATH = RAW_DIR / "qrels.tsv"

# Minimum expected sizes (bytes) — a quick sanity gate to avoid re-downloading
# when valid files already exist.  Actual files are ~600 KB, ~10 KB, ~20 KB.
_MIN_DOCS_BYTES = 100_000
_MIN_QUERIES_BYTES = 2_000
_MIN_QRELS_BYTES = 5_000


def _files_look_present() -> bool:
    """Return True if all three raw files exist and exceed minimum sizes."""
    for path, min_bytes in [
        (DOCS_PATH, _MIN_DOCS_BYTES),
        (QUERIES_PATH, _MIN_QUERIES_BYTES),
        (QRELS_PATH, _MIN_QRELS_BYTES),
    ]:
        if not path.exists() or path.stat().st_size < min_bytes:
            return False
    return True


def download_dataset(force: bool = False) -> None:
    """Fetch the Cranfield collection via *ir_datasets* and export it.

    Args:
        force: If ``True``, overwrite existing files even when they look valid.
    """
    if not force and _files_look_present():
        logger.info("Raw dataset files already present — skipping download.")
        return

    # Late import so the rest of the project does not depend on ir_datasets
    # at import time.
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError(
            "The 'ir_datasets' package is required for dataset acquisition. "
            "Install it with:  pip install ir-datasets"
        ) from exc

    logger.info("Loading Cranfield dataset via ir_datasets …")
    ds = ir_datasets.load("cranfield")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Documents --------------------------------------------------------
    doc_count = 0
    with open(DOCS_PATH, "w", encoding="utf-8") as fh:
        for doc in ds.docs_iter():
            record = {
                "document_id": doc.doc_id,
                "title": getattr(doc, "title", "") or "",
                "text": getattr(doc, "text", "") or "",
                "author": getattr(doc, "author", "") or "",
                "bibliography": getattr(doc, "bib", "") or "",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_count += 1
    logger.info("Exported %d documents → %s", doc_count, DOCS_PATH)

    # ---- Queries ----------------------------------------------------------
    query_count = 0
    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        for query in ds.queries_iter():
            record = {
                "query_id": query.query_id,
                "query_text": getattr(query, "text", "") or "",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            query_count += 1
    logger.info("Exported %d queries  → %s", query_count, QUERIES_PATH)

    # ---- Qrels ------------------------------------------------------------
    qrel_count = 0
    with open(QRELS_PATH, "w", encoding="utf-8") as fh:
        fh.write("query_id\tdocument_id\trelevance\n")
        for qrel in ds.qrels_iter():
            fh.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            qrel_count += 1
    logger.info("Exported %d qrels    → %s", qrel_count, QRELS_PATH)

    logger.info("Dataset acquisition complete.")


# ---------------------------------------------------------------------------
# Allow running as a standalone script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_dataset()

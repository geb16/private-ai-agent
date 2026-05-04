"""Document ingestion for the enterprise private RAG agent."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from private_agent.config import Settings, load_settings
from private_agent.logging_utils import configure_logging
from private_agent.modeling import create_embeddings

logger = logging.getLogger(__name__)


def _list_supported_files(data_dir: Path) -> list[Path]:
    """Collect supported document files from the data directory."""
    supported_extensions = {".txt", ".md", ".pdf"}
    return sorted(
        file_path
        for file_path in data_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions
    )


def _load_file(file_path: Path) -> list[Document]:
    """Load one file into LangChain documents based on extension."""
    if file_path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(file_path)).load()

    return TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True).load()


def _normalize_metadata(documents: list[Document], file_path: Path) -> list[Document]:
    """Normalize metadata keys to keep source attribution stable at runtime."""
    source_name = file_path.name
    source_path = str(file_path.resolve())

    for document in documents:
        document.metadata["source"] = source_name
        document.metadata["source_path"] = source_path

    return documents


def _build_chunk_ids(chunks: list[Document]) -> list[str]:
    """Generate deterministic chunk IDs from content and metadata."""
    chunk_ids: list[str] = []

    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        page = str(chunk.metadata.get("page", "na"))
        digest_input = f"{source}|{page}|{index}|{chunk.page_content}".encode("utf-8")
        chunk_ids.append(hashlib.sha256(digest_input).hexdigest())

    return chunk_ids


def ingest_documents(settings: Settings, rebuild: bool = False) -> int:
    """Ingest supported files into the Chroma vector store.

    Args:
        settings: Runtime settings object.
        rebuild: If True, fully removes any existing vector DB directory first.

    Returns:
        Number of indexed chunks.
    """
    if not settings.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {settings.data_dir}")

    if rebuild and settings.vector_db_dir.exists():
        logger.warning("Rebuild mode enabled. Removing %s", settings.vector_db_dir)
        shutil.rmtree(settings.vector_db_dir)

    files_to_ingest = _list_supported_files(settings.data_dir)
    if not files_to_ingest:
        raise RuntimeError(
            f"No supported documents found in {settings.data_dir}. "
            "Supported extensions: .txt, .md, .pdf"
        )

    all_documents: list[Document] = []
    for file_path in files_to_ingest:
        loaded_docs = _load_file(file_path)
        all_documents.extend(_normalize_metadata(loaded_docs, file_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(all_documents)

    if not chunks:
        raise RuntimeError("Document splitting produced 0 chunks; ingestion aborted.")

    settings.vector_db_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=create_embeddings(settings),
        persist_directory=str(settings.vector_db_dir),
    )

    vector_store.add_documents(chunks, ids=_build_chunk_ids(chunks))

    # Some versions persist automatically; keep explicit persistence when available.
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    logger.info(
        "Ingestion completed: files=%d chunks=%d collection=%s db=%s",
        len(files_to_ingest),
        len(chunks),
        settings.collection_name,
        settings.vector_db_dir,
    )
    return len(chunks)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ingestion command."""
    parser = argparse.ArgumentParser(description="Ingest local documents into Chroma DB.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing vector DB directory before ingesting.",
    )
    return parser


def main() -> None:
    """CLI entry point for document ingestion."""
    args = _build_arg_parser().parse_args()

    settings = load_settings()
    configure_logging(settings.log_level)

    chunk_count = ingest_documents(settings=settings, rebuild=args.rebuild)
    print(f"Indexed {chunk_count} chunks into '{settings.collection_name}'.")


if __name__ == "__main__":
    main()

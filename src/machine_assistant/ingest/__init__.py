"""Ingestion package for machine_assistant."""
from machine_assistant.ingest.base_parser import BaseParser, Chunk, ParserRegistry, SourceDocument
from machine_assistant.ingest import excel_parser, pdf_parser  # noqa: F401

__all__ = [
    "BaseParser",
    "Chunk",
    "ParserRegistry",
    "SourceDocument",
    "excel_parser",
    "pdf_parser",
]

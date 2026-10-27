"""Abstraksi dasar untuk pemrosesan dokumen (ingest) dan registrasi parser.

Mendefinisikan skema data Chunk, SourceDocument, antarmuka BaseParser,
dan ParserRegistry untuk mendukung modularitas ekstensi berkas.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Type

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "excel"]
ChunkKind = Literal[
    "alarm",
    "parameter",
    "io_signal",
    "status_message",
    "prose",
    "history_record",
]


class SourceDocument(BaseModel):
    """Metadata berkas dokumen sumber yang didaftarkan pada config/sources.yaml."""
    doc_id: str
    filename: str
    doc_title: str
    doc_version: str
    source_type: SourceType
    machine_type: str = "ALL"
    profile: str


class Chunk(BaseModel):
    """Representasi unit teks terkecil beranotasi hasil pemotongan dokumen."""
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_version: str
    source_type: SourceType
    page: int | None = None
    section: str = ""
    breadcrumb: str = ""
    kind: ChunkKind = "prose"
    codes: List[str] = Field(default_factory=list)
    text_en: str
    text_id: str | None = None
    related_ids: List[str] = Field(default_factory=list)
    content_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def compute_hash(self) -> str:
        """Menghitung hash SHA-256 dari konten chunk untuk deteksi perubahan incremental."""
        payload = f"{self.doc_id}|{self.page}|{self.section}|{self.kind}|{self.text_en}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class BaseParser(ABC):
    """Kelas abstrak dasar untuk semua parser dokumen (PDF dan Excel)."""

    @abstractmethod
    def parse(
        self,
        file_path: Path,
        source_doc: SourceDocument,
        profile_config: Dict[str, Any],
    ) -> Iterable[Chunk]:
        """Mem-parsing berkas dokumen menjadi kumpulan objek Chunk."""
        raise NotImplementedError


class ParserRegistry:
    """Registry pemetaan ekstensi/tipe sumber ke kelas implementasi parser konkret."""

    _registry: Dict[str, Type[BaseParser]] = {}

    @classmethod
    def register(cls, key: str, parser_cls: Type[BaseParser]) -> None:
        """Mendaftarkan kelas parser untuk kunci/ekstensi tertentu."""
        cls._registry[key.lower()] = parser_cls

    @classmethod
    def get_parser(cls, key: str) -> BaseParser:
        """Mengambil instance parser berdasarkan ekstensi atau tipe sumber."""
        normalized = key.lower().lstrip(".")
        if normalized not in cls._registry:
            raise ValueError(f"Tidak ada parser terdaftar untuk format: '{key}'")
        return cls._registry[normalized]()

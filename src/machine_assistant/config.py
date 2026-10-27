"""Modul konfigurasi dan pengaturan aplikasi Machine Assistant.

Mengelola pemuatan file konfigurasi YAML, penentuan direktori kerja (docs, index, debug),
serta validasi parameter runtime, ingest, retrieval, dan Local LLM.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field


def get_repo_root() -> Path:
    """Mengembalikan path absolut direktori root repositori proyek.

    Struktur folder: <repo_root>/src/machine_assistant/config.py
    sehingga parents[2] menunjuk ke direktori root proyek.
    """
    return Path(__file__).resolve().parents[2]


def get_default_data_dir() -> Path:
    """Menentukan direktori data utama.

    Secara default mengarah ke root repositori proyek tempat folder docs/ dan index/ berada.
    """
    return get_repo_root()


def load_yaml(path: Path) -> Dict[str, Any]:
    """Membaca berkas YAML dan mengembalikannya sebagai dictionary Python.

    Jika berkas tidak ditemukan atau kosong, mengembalikan dictionary kosong {}.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class IngestConfig(BaseModel):
    """Konfigurasi parser dan chunking dokumen teknis (PDF & Excel)."""
    pdf_backend: str = "pymupdf"
    header_footer_threshold: float = 0.50
    prose_chunk_tokens: int = 300
    prose_chunk_overlap: int = 50


class RetrievalConfig(BaseModel):
    """Konfigurasi kuota dan parameter mesin pencarian (retrieval engine)."""
    top_k_per_route: int = 5
    final_top_k: int = 5


class LocalLLMConfig(BaseModel):
    """Konfigurasi koneksi HTTP ke server inferensi lokal (llama-server / llama.cpp)."""
    base_url: str = "http://127.0.0.1:8080"
    model: str = "qwen2.5-1.5b-instruct"
    temperature: float = 0.2
    max_tokens: int = 450
    timeout_s: int = 180


class LLMConfig(BaseModel):
    """Konfigurasi generator jawaban bahasa alami."""
    enabled: bool = True
    provider: str = "local"  # "local" (llama-server) atau "none" (raw retrieval fallback)
    model: str = "qwen2.5-1.5b-instruct"
    timeout_s: int = 180
    max_retries: int = 1
    allowed_source_types: List[str] = Field(default_factory=lambda: ["pdf"])
    local: LocalLLMConfig = Field(default_factory=LocalLLMConfig)


class RuntimeConfig(BaseModel):
    """Konfigurasi lingkungan eksekusi aplikasi."""
    device: str = "cpu"
    offline_mode: bool = True


class AppSettings(BaseModel):
    """Skema pengaturan menyeluruh untuk aplikasi Machine Assistant."""
    pipeline_version: str = "0.2.0"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data_dir: Path = Field(default_factory=get_default_data_dir)

    @property
    def docs_dir(self) -> Path:
        """Direktori tempat dokumen sumber (PDF manual dan Excel laporan)."""
        return self.data_dir / "docs"

    @property
    def index_dir(self) -> Path:
        """Direktori tempat indeks hasil parsing (chunks.jsonl dan manifest.json)."""
        return self.data_dir / "index"

    @property
    def debug_dir(self) -> Path:
        """Direktori artefak pembantu dan laporan analitik ingest."""
        return self.data_dir / "debug"

    def ensure_dirs(self) -> None:
        """Membuat direktori kerja (docs, index, debug) jika belum ada."""
        for directory in (self.docs_dir, self.index_dir, self.debug_dir):
            directory.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Memuat pengaturan aplikasi dari berkas YAML konfigurasi.

    Mencari settings.yaml terlebih dahulu, jika belum dibuat maka memakai settings.example.yaml.
    """
    repo_root = get_repo_root()
    if config_path is None:
        custom_path = repo_root / "config" / "settings.yaml"
        example_path = repo_root / "config" / "settings.example.yaml"
        config_path = custom_path if custom_path.exists() else example_path

    raw_data = load_yaml(config_path)
    settings = AppSettings(**raw_data)
    settings.model_rebuild()
    settings.ensure_dirs()
    if settings.runtime.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return settings

"""Pipeline ingest dan indexing dokumen sumber Machine Assistant.

Membaca sources.yaml, mengecek hash dokumen dan profil konfigurasi (incremental ingest),
memanggil parser PDF dan Excel, serta menuliskan indeks gabungan ke chunks.jsonl
beserta manifest.json untuk pelacakan versi dan integritas data.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from machine_assistant.config import AppSettings, get_repo_root, load_yaml
from machine_assistant.ingest import excel_parser, pdf_parser  # noqa: F401
from machine_assistant.ingest.base_parser import Chunk, ParserRegistry, SourceDocument


def _compute_file_hash(path: Path) -> str:
    """Menghitung hash SHA-256 dari berkas untuk mendeteksi perubahan."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_sources(repo_root: Path) -> List[SourceDocument]:
    """Memuat daftar dokumen sumber yang terdaftar dalam config/sources.yaml."""
    sources_path = repo_root / "config" / "sources.yaml"
    data = load_yaml(sources_path)
    return [SourceDocument(**item) for item in data.get("sources", [])]


def load_profile(repo_root: Path, profile_name: str) -> dict:
    """Memuat konfigurasi profil parsing spesifik dari folder config/profiles/."""
    profile_path = repo_root / "config" / "profiles" / f"{profile_name}.yaml"
    return load_yaml(profile_path)


def compute_profile_hash(repo_root: Path, profile_name: str) -> str:
    """Menghitung hash SHA-256 dari file konfigurasi profil."""
    profile_path = repo_root / "config" / "profiles" / f"{profile_name}.yaml"
    if not profile_path.exists():
        return ""
    return _compute_file_hash(profile_path)


def _write_stats_report(
    stats: dict,
    output_path: Path,
    source_doc: SourceDocument,
) -> None:
    """Write a human-readable statistics report for Excel ingest."""
    lines = [
        "=" * 70,
        f"  LAPORAN STATISTIK INGEST: {source_doc.doc_title}",
        f"  File: {source_doc.filename}",
        "=" * 70,
        "",
        f"  Total record   : {stats.get('total_records', 0)}",
        f"  Penyebab terisi: {stats.get('penyebab_filled', 0)}",
        f"  Penyebab kosong: {stats.get('penyebab_empty', 0)} "
        f"({stats.get('penyebab_empty', 0) * 100 / max(stats.get('total_records', 1), 1):.1f}%)",
        f"  Low-info record: {stats.get('low_info_count', 0)}",
        f"  Tanggal ditandai: {stats.get('flagged_date_count', 0)}",
        "",
        "  --- Per Mesin (top 20) ---",
    ]
    by_machine = sorted(
        stats.get("by_machine", {}).items(), key=lambda x: x[1], reverse=True
    )
    for machine, count in by_machine[:20]:
        lines.append(f"    {count:5d}  {machine}")

    lines.extend(["", "  --- Per Area ---"])
    by_area = sorted(
        stats.get("by_area", {}).items(), key=lambda x: x[1], reverse=True
    )
    for area, count in by_area[:15]:
        lines.append(f"    {count:5d}  {area}")

    lines.extend(["", "  --- Per Status (normalized) ---"])
    for status, count in sorted(
        stats.get("by_status", {}).items(), key=lambda x: x[1], reverse=True
    ):
        lines.append(f"    {count:5d}  {status}")

    lines.extend(["", "  --- Per Jenis Kerusakan (normalized) ---"])
    for jenis, count in sorted(
        stats.get("by_jenis", {}).items(), key=lambda x: x[1], reverse=True
    ):
        lines.append(f"    {count:5d}  {jenis}")

    flagged = stats.get("flagged_rows", [])
    if flagged:
        lines.extend(["", f"  --- Baris dengan tanggal ditandai ({len(flagged)}) ---"])
        for f in flagged[:20]:
            lines.append(
                f"    Baris {f['row']:5d}: date={f['date']}  machine={f['machine']}"
            )
        if len(flagged) > 20:
            lines.append(f"    ... dan {len(flagged) - 20} baris lagi")

    lines.extend(["", "=" * 70])

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    # Also print to console
    print(report_text)


def run_ingest(
    settings: AppSettings,
    rebuild: bool = False,
    parse_only: bool = False,
) -> Path:
    """Execute the ingestion pipeline, write manifest.json, and write chunks.jsonl."""
    settings.ensure_dirs()
    repo_root = get_repo_root()
    sources = load_sources(repo_root)

    manifest_path = settings.index_dir / "manifest.json"
    index_jsonl = settings.index_dir / "chunks.jsonl"

    prev_manifest: Dict[str, Any] | None = None
    cached_chunks_by_doc: Dict[str, List[Chunk]] = {}
    can_incremental = False

    if rebuild:
        print("[INFO] --rebuild active. Re-indexing all sources from scratch.")
    elif manifest_path.exists() and index_jsonl.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
            if not isinstance(prev_manifest, dict):
                raise ValueError("Manifest is not a JSON object")

            prev_version = prev_manifest.get("pipeline_version")
            if prev_version != settings.pipeline_version:
                print(
                    f"[INFO] Pipeline version mismatch ({prev_version} != {settings.pipeline_version}). "
                    "Running full rebuild."
                )
            elif "sources" not in prev_manifest or not isinstance(
                prev_manifest["sources"], dict
            ):
                raise ValueError("Manifest missing valid 'sources' dictionary")
            else:
                # Load existing chunks grouped by doc_id
                with index_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        chunk_data = json.loads(line)
                        chunk = Chunk(**chunk_data)
                        cached_chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
                can_incremental = True
                print(
                    f"[INFO] Found valid manifest (v{prev_version}) and cached chunks. "
                    "Attempting incremental ingest."
                )
        except Exception as e:
            print(
                f"[WARN] Corrupt or incomplete manifest found ({e}). Falling back to full rebuild."
            )
            can_incremental = False
            prev_manifest = None
            cached_chunks_by_doc.clear()

    manifest: Dict[str, Any] = {
        "pipeline_version": settings.pipeline_version,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "total_chunks": 0,
    }

    all_chunks: List[Chunk] = []
    for src in sources:
        file_path = settings.docs_dir / src.filename
        if not file_path.exists():
            print(f"[WARN] Skipping missing source file: {file_path}")
            continue

        file_hash = _compute_file_hash(file_path)
        profile_hash = compute_profile_hash(repo_root, src.profile)

        # Check if source can be reused from cache
        reused = False
        if can_incremental and prev_manifest and src.doc_id in prev_manifest["sources"]:
            prev_src_info = prev_manifest["sources"][src.doc_id]
            if (
                prev_src_info.get("sha256") == file_hash
                and prev_src_info.get("profile_sha256") == profile_hash
                and src.doc_id in cached_chunks_by_doc
                and len(cached_chunks_by_doc[src.doc_id])
                == prev_src_info.get("chunks_count", -1)
            ):
                doc_chunks = cached_chunks_by_doc[src.doc_id]
                print(f"[INFO] Reusing cached chunks for {src.doc_id} ({src.filename})")
                reused = True

        if not reused:
            profile_cfg = load_profile(repo_root, src.profile)
            parser = ParserRegistry.get_parser(src.source_type)
            doc_chunks = list(parser.parse(file_path, src, profile_cfg))
            print(f"[INFO] Parsed {len(doc_chunks)} chunks from {src.filename}")

            # Write Excel-specific stats report and per-source JSONL
            if src.source_type == "excel" and hasattr(parser, "last_stats"):
                stats = parser.last_stats
                if stats:
                    excel_jsonl = settings.debug_dir / "excel_records.jsonl"
                    with excel_jsonl.open("w", encoding="utf-8") as f:
                        for chunk in doc_chunks:
                            f.write(
                                json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
                            )
                    print(f"[INFO] Wrote {len(doc_chunks)} Excel records to {excel_jsonl}")

                    stats_path = settings.debug_dir / "excel_ingest_report.txt"
                    _write_stats_report(stats, stats_path, src)
                    print(f"[INFO] Statistics report: {stats_path}")

        elif src.source_type == "excel":
            excel_jsonl = settings.debug_dir / "excel_records.jsonl"
            if not excel_jsonl.exists():
                with excel_jsonl.open("w", encoding="utf-8") as f:
                    for chunk in doc_chunks:
                        f.write(
                            json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
                        )

        all_chunks.extend(doc_chunks)

        manifest["sources"][src.doc_id] = {
            "filename": src.filename,
            "source_type": src.source_type,
            "sha256": file_hash,
            "profile_sha256": profile_hash,
            "chunks_count": len(doc_chunks),
            "profile": src.profile,
        }

    manifest["total_chunks"] = len(all_chunks)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Updated manifest: {manifest_path}")

    # Write to debug/chunks.jsonl AND index/chunks.jsonl
    output_jsonl = settings.debug_dir / "chunks.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    with index_jsonl.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(all_chunks)} total chunks to {output_jsonl}")
    if parse_only:
        print("[INFO] --parse-only flag active. Skipping translation and embedding.")
    return output_jsonl

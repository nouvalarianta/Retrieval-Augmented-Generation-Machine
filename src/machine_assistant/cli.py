"""Antarmuka Baris Perintah (Command Line Interface - CLI) untuk Machine Assistant.

Menyediakan dua perintah utama:
1. `ingest`: Membaca, mem-parsing, menormalisasi, dan mengindeks dokumen teknis (PDF & Excel).
2. `ask`: Menjawab pertanyaan pemeliharaan teknis menggunakan Dual-Track RAG (Local LLM + Aggregator).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from machine_assistant.config import AppSettings, load_settings
from machine_assistant.ingest.pipeline import run_ingest


def build_parser() -> argparse.ArgumentParser:
    """Membangun parser argumen CLI untuk perintah `ingest` dan `ask`."""
    parser = argparse.ArgumentParser(
        prog="ma",
        description="Asisten Troubleshooting Lapangan Mesin Industri Bühler MEAG berbasis RAG Lokal",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Perintah: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Parsing dan bangun indeks dokumen PDF serta Excel")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Paksa pembuatan ulang seluruh indeks dari awal tanpa menggunakan cache",
    )
    ingest_parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Hanya jalankan ekstraksi dan pembagian chunk tanpa mengubah status indeks produksi",
    )

    # Perintah: ask
    ask_parser = subparsers.add_parser("ask", help="Ajukan pertanyaan kendala teknis atau troubleshooting")
    ask_parser.add_argument("question", type=str, help="Pertanyaan teknis dalam Bahasa Indonesia atau Inggris")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Jumlah kutipan dokumen maksimum yang diambil per sumber (default: 5)",
    )
    ask_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Lewati pemanggilan LLM dan tampilkan kutipan manual asli secara langsung",
    )

    return parser


def _print_raw_pdf_results(results: list) -> None:
    """Menampilkan kutipan manual PDF secara langsung (mode fallback tanpa LLM)."""
    for rank, result in enumerate(results, 1):
        print(f"  [{rank}] {result.doc_title} — Hal. {result.page} (Skor: {result.score:.2f})")
        print(f"       Lokasi: {result.breadcrumb}")
        wrapped = textwrap.fill(
            result.text.strip(),
            width=68,
            initial_indent="       ",
            subsequent_indent="       ",
        )
        print(wrapped)
        print(f"  {'-'*66}")


def run_ask(
    question: str,
    chunks_path: Path,
    top_k: int = 5,
    use_llm: bool = True,
    settings: AppSettings | None = None,
) -> int:
    """Menjalankan alur pencarian dual-source dan menyusun jawaban terstruktur.

    Langkah alur:
    1. Pencarian BM25 dan Entity Detection via RetrievalEngine.
    2. Pengecekan guardrail Out-of-Scope (merek pihak ketiga, PII karyawan, perintah OS).
    3. Ekstraksi Bagian 1 (Manual PDF): Diringkas oleh Local LLM (llama-server) atau raw text fallback.
    4. Ekstraksi Bagian 2 (Riwayat Excel): Dirangkum secara deterministik lokal oleh MaintenanceAggregator.
    """
    from machine_assistant.answer.aggregator import MaintenanceAggregator
    from machine_assistant.retrieval.engine import RetrievalEngine

    if settings is None:
        settings = load_settings()

    if not chunks_path.exists():
        print(
            "[ERROR] Berkas indeks tidak ditemukan. Silakan jalankan ingest terlebih dahulu:\n"
            "  ma ingest --rebuild"
        )
        return 1

    print(f"\n{'='*70}")
    print(f"  PERTANYAAN: {question}")
    print(f"{'='*70}\n")

    print("  [1/2] Mencari relevansi dari manual PDF dan riwayat Excel...")
    engine = RetrievalEngine(chunks_path)
    dual = engine.retrieve_dual(question, top_k_pdf=top_k, top_k_excel=top_k)

    # Tangani penolakan dini untuk pertanyaan out-of-scope / merek pihak ketiga / privasi
    if getattr(dual, "out_of_scope_reason", None):
        print("  Ditemukan 0 kutipan manual dan 0 catatan riwayat maintenance.\n")
        print(f"  {'='*68}")
        print("  [INFORMASI CAKUPAN & KEBIJAKAN PRIVASI SISTEM]")
        print(f"  {'='*68}\n")
        print(f"  {dual.out_of_scope_reason}\n")
        print(f"{'='*70}\n")
        return 0

    pdf_count = len(dual.pdf_results)
    excel_count = len(dual.excel_results)

    if not dual.has_any:
        print("  Tidak ditemukan hasil yang relevan baik di manual maupun riwayat maintenance.\n")
        return 0

    print(f"  Ditemukan {pdf_count} kutipan manual dan {excel_count} catatan riwayat maintenance.\n")

    # =========================================================================
    # BAGIAN 1: DARI MANUAL (BUKU PANDUAN RESMI BÜHLER MEAG)
    # =========================================================================
    if dual.pdf_results:
        print(f"  {'='*68}")
        print("  [DARI MANUAL (BUKU PANDUAN BÜHLER)]")
        print(f"  {'='*68}\n")

        llm_enabled = use_llm and settings.llm.enabled
        show_sources = True

        if llm_enabled and settings.llm.provider.lower() == "local":
            from machine_assistant.answer.local_llm import generate_local_answer
            answer, used_llm = generate_local_answer(
                question=question,
                results=dual.pdf_results,
                config=settings.llm.local,
            )
            for line in answer.splitlines():
                print(f"  {line}" if line.strip() else "")
            if not used_llm and not answer.startswith("Jawaban untuk:"):
                show_sources = False
        else:
            _print_raw_pdf_results(dual.pdf_results)

        if show_sources:
            print(f"\n  Sumber Manual:")
            seen_docs = set()
            for r in dual.pdf_results:
                cite = f"{r.doc_title} — Hal. {r.page}"
                if cite not in seen_docs:
                    print(f"    • {cite}")
                    seen_docs.add(cite)
            print()

    # =========================================================================
    # BAGIAN 2: DARI RIWAYAT PERBAIKAN MAINTENANCE (LAPORAN HARIAN EXCEL)
    # =========================================================================
    if dual.excel_results:
        print(f"  {'='*68}")
        print("  [DARI RIWAYAT PERBAIKAN MAINTENANCE (LAPORAN HARIAN)]")
        print(f"  {'='*68}\n")

        # Riwayat maintenance diagregasi 100% lokal tanpa LLM untuk menjamin kerahasiaan data operasional
        history_report = MaintenanceAggregator.aggregate(dual.excel_results, dual.detected_entities)
        for line in history_report.splitlines():
            print(f"  {line}" if line.strip() else "")

        print("\n  Catatan:")
        print("    • Riwayat dianalisis secara lokal tanpa LLM untuk menjaga kerahasiaan data operasional.")
        print()

    print(f"{'='*70}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Fungsi titik masuk (entrypoint) utama aplikasi CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "ingest":
        run_ingest(settings, rebuild=args.rebuild, parse_only=args.parse_only)
        return 0

    if args.command == "ask":
        chunks_path = settings.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            chunks_path = settings.debug_dir / "chunks.jsonl"
        use_llm = not args.no_llm
        return run_ask(args.question, chunks_path, top_k=args.top_k, use_llm=use_llm, settings=settings)

    return 1


if __name__ == "__main__":
    sys.exit(main())

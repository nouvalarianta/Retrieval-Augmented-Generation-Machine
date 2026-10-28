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

# Lebar tampilan terminal — aman untuk CMD/PowerShell Windows
WIDTH = 72
LINE  = "=" * WIDTH
DASH  = "-" * WIDTH
THIN  = "·" * WIDTH


def _box(title: str) -> None:
    """Cetak header kotak bertingkat untuk setiap bagian utama."""
    inner = f"  {title}  "
    pad   = max(0, WIDTH - len(inner))
    left  = pad // 2
    right = pad - left
    print(LINE)
    print(f"{'':>{left}}{inner}{'':>{right}}")
    print(LINE)


def _section(label: str) -> None:
    """Cetak label sub-bagian dengan garis tipis di bawahnya."""
    print()
    print(f"  ┌─ {label}")
    print(f"  {DASH}")


def _body(text: str, indent: int = 4) -> None:
    """Cetak teks dengan word-wrap dan indentasi konsisten."""
    prefix = " " * indent
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            print()
            continue
        wrapped = textwrap.fill(
            paragraph,
            width=WIDTH - indent,
            initial_indent=prefix,
            subsequent_indent=prefix + "  ",
        )
        print(wrapped)


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
        print(f"\n    [{rank}] {result.doc_title}  —  Halaman {result.page}  (Skor: {result.score:.2f})")
        print(f"        Lokasi : {result.breadcrumb}")
        print(f"    {'- ' * 33}")
        _body(result.text.strip(), indent=8)
    print()


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
        print()
        _box("  MACHINE ASSISTANT  —  ERROR  ")
        print(f"\n  Berkas indeks tidak ditemukan.")
        print(f"  Jalankan perintah berikut terlebih dahulu:\n")
        print(f"      ma ingest --rebuild\n")
        print(LINE)
        return 1

    # ─── HEADER ──────────────────────────────────────────────────────────────
    print()
    _box("ASISTEN TROUBLESHOOTING  BUHLER MEAG")
    print()
    print("  PERTANYAAN")
    print(f"  {THIN}")
    _body(question, indent=4)
    print(f"  {THIN}")

    # ─── PROSES PENCARIAN ─────────────────────────────────────────────────────
    print()
    print("  >> Mencari informasi di manual PDF dan riwayat Excel...")
    engine = RetrievalEngine(chunks_path)
    dual   = engine.retrieve_dual(question, top_k_pdf=top_k, top_k_excel=top_k)

    # Tangani penolakan dini untuk pertanyaan out-of-scope
    if getattr(dual, "out_of_scope_reason", None):
        print()
        print(LINE)
        print("  INFORMASI CAKUPAN SISTEM")
        print(f"  {DASH}")
        print()
        _body(dual.out_of_scope_reason, indent=4)
        print()
        print(LINE)
        return 0

    pdf_count   = len(dual.pdf_results)
    excel_count = len(dual.excel_results)

    if not dual.has_any:
        print()
        print(LINE)
        print("  Tidak ditemukan hasil relevan di manual maupun riwayat maintenance.")
        print(LINE)
        return 0

    print(f"  >> Ditemukan  {pdf_count} kutipan manual  +  {excel_count} catatan riwayat.")

    # =========================================================================
    # BAGIAN 1 — DARI MANUAL RESMI BÜHLER MEAG
    # =========================================================================
    if dual.pdf_results:
        print()
        print()
        print(LINE)
        print("  BAGIAN 1  —  DARI MANUAL RESMI BUHLER MEAG")
        print(LINE)

        llm_enabled  = use_llm and settings.llm.enabled
        show_sources = True

        if llm_enabled and settings.llm.provider.lower() == "local":
            from machine_assistant.answer.local_llm import generate_local_answer
            answer, used_llm = generate_local_answer(
                question=question,
                results=dual.pdf_results,
                config=settings.llm.local,
            )

            # Cetak setiap baris jawaban LLM dengan indentasi rapi
            print()
            for line in answer.splitlines():
                stripped = line.strip()
                if not stripped:
                    print()
                    continue
                # Deteksi label seksi (Kesimpulan:, Tindakan yang disarankan:, dll.)
                if stripped.endswith(":") and len(stripped) < 50 and not stripped[0].isdigit():
                    print(f"\n  >> {stripped}")
                    print(f"  {'- ' * 33}")
                elif stripped[0].isdigit() and stripped[1] in ".)" if len(stripped) > 1 else False:
                    # Langkah bernomor
                    _body(stripped, indent=6)
                else:
                    _body(stripped, indent=4)

            if not used_llm and not answer.startswith("Jawaban untuk:"):
                show_sources = False
        else:
            print()
            _print_raw_pdf_results(dual.pdf_results)

        # Sumber manual
        if show_sources and dual.pdf_results:
            print()
            print(f"  {THIN}")
            print("  SUMBER MANUAL")
            seen_docs: set[str] = set()
            for r in dual.pdf_results:
                cite = f"{r.doc_title}  —  Hal. {r.page}"
                if cite not in seen_docs:
                    print(f"    •  {cite}")
                    seen_docs.add(cite)

    # =========================================================================
    # BAGIAN 2 — RIWAYAT PERBAIKAN MAINTENANCE (LAPORAN HARIAN)
    # =========================================================================
    if dual.excel_results:
        print()
        print()
        print(LINE)
        print("  BAGIAN 2  —  RIWAYAT PERBAIKAN MAINTENANCE")
        print(LINE)

        history_report = MaintenanceAggregator.aggregate(dual.excel_results, dual.detected_entities)
        print()
        for line in history_report.splitlines():
            stripped = line.strip()
            if not stripped:
                print()
                continue
            # Baris yang dimulai dengan bullet point
            if stripped.startswith("•"):
                # Bullet utama: cetak dengan satu spasi ekstra
                _body(stripped, indent=4)
            elif stripped.startswith("-") or stripped.startswith("["):
                # Sub-item / contoh catatan
                _body(stripped, indent=8)
            elif stripped.startswith("Berdasarkan"):
                # Kalimat pembuka laporan
                print()
                print(f"  {stripped}")
            else:
                _body(stripped, indent=6)

    # ─── FOOTER ──────────────────────────────────────────────────────────────
    print()
    print()
    print(LINE)
    print("  Selesai. Ketik pertanyaan berikutnya atau 'keluar' untuk menutup.")
    print(LINE)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Fungsi titik masuk (entrypoint) utama aplikasi CLI."""
    parser = build_parser()
    args   = parser.parse_args(argv)
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

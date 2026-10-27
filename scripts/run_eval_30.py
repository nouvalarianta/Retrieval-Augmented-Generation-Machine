#!/usr/bin/env python3
"""Runner script to sequentially execute 30 benchmark questions from ask.md

and record full outputs into answer.md.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    ask_path = root / "ask.md"
    answer_path = root / "answer.md"
    ma_bin = root / ".venv" / "bin" / "ma"

    if not ask_path.exists():
        print(f"[ERROR] {ask_path} tidak ditemukan!", file=sys.stderr)
        sys.exit(1)

    if not ma_bin.exists():
        print(f"[ERROR] {ma_bin} tidak ditemukan!", file=sys.stderr)
        sys.exit(1)

    with open(ask_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    questions = []
    for line in lines:
        if line.startswith("#"):
            continue
        # Strip numbering like "1. ", "2. ", etc.
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            questions.append(m.group(1).strip())
        else:
            questions.append(line)

    print(f"[*] Ditemukan {len(questions)} pertanyaan di ask.md")
    print(f"[*] Menulis hasil pengujian ke: {answer_path}")
    print(f"[*] Menjalankan satu per satu secara sekuensial...\n")

    # Header for answer.md
    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(f"# Hasil Evaluasi {len(questions)} Pertanyaan Uji — Machine Assistant\n\n")
        f.write(f"- **Waktu Pengujian:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("- **Model Generator:** `Qwen2.5-1.5B-Instruct Q4_K_M GGUF` via `llama-server`\n")
        f.write("- **Endpoint Local LLM:** `http://127.0.0.1:8080/v1/chat/completions`\n")
        f.write("- **Metode Retrieval:** Dual-Source BM25 + Entity Detection\n")
        f.write("- **Mode Riwayat Maintenance:** 100% Pemrosesan Lokal (Python Deterministik Tanpa LLM)\n\n")
        f.write("---\n\n")

    total_start = time.time()

    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Menjalankan: \"{q[:60]}...\"")
        t0 = time.time()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            res = subprocess.run(
                [str(ma_bin), "ask", q],
                cwd=str(root),
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
            raw_output = res.stdout.strip()
            if not raw_output and res.stderr:
                raw_output = f"[STDERR]\n{res.stderr.strip()}"
        except subprocess.TimeoutExpired:
            raw_output = "[ERROR] Eksekusi timeout melebihi 180 detik."
        except Exception as e:
            raw_output = f"[ERROR] Gagal mengeksekusi: {e}"

        elapsed = time.time() - t0
        print(f"    -> Selesai dalam {elapsed:.1f} detik\n")

        # Append to answer.md immediately
        with open(answer_path, "a", encoding="utf-8") as f:
            f.write(f"## Pertanyaan {idx}\n\n")
            f.write(f"> **\"{q}\"**\n\n")
            f.write(f"*Waktu respon: {elapsed:.1f} detik*\n\n")
            f.write("```text\n")
            f.write(raw_output + "\n")
            f.write("```\n\n")
            f.write("---\n\n")
            f.flush()

    total_time = time.time() - total_start
    print(f"[✓] Seluruh {len(questions)} pertanyaan selesai dalam {total_time:.1f} detik ({total_time/60:.2f} menit).")
    print(f"[✓] Seluruh output tersimpan rapi di {answer_path}")


if __name__ == "__main__":
    main()

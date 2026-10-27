"""Local LLM answer generator using llama-server (OpenAI-compatible HTTP API).

Takes retrieved PDF chunks from retrieval engine and formats a structured,
natural Indonesian-language troubleshooting answer using a local LLM instance.
"""
from __future__ import annotations

import json
import re
import socket
import textwrap
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, List, Optional

from machine_assistant.retrieval.engine import check_out_of_scope

if TYPE_CHECKING:
    from machine_assistant.config import LocalLLMConfig
    from machine_assistant.retrieval.engine import RetrievalResult

# Minimum BM25 score required to send PDF chunks to the LLM.
# Prevents feeding random noise (like matching number "18" on out-of-scope queries).
MIN_LLM_PDF_SCORE = 3.5

# Regex for queries asking for personal data / HR / non-machine admin
_OUT_OF_SCOPE_QUERY_RE = re.compile(
    r"\b(?:siapa\s+nama|sebutkan\s+nik|nomor\s+induk\s+karyawan|"
    r"serah\s+terima\s+user|surat\s+laporan.*hrd|oli\s+gardan.*forklift)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """\
Kamu adalah asisten teknisi senior untuk mesin industri Bühler MEAG.
Tugasmu adalah menjelaskan isi kutipan manual bahasa Inggris ke dalam Bahasa Indonesia yang natural, lugas, dan mudah dipahami teknisi lapangan.

ATURAN WAJIB:
1. Gunakan Bahasa Indonesia sepenuhnya untuk penjelasan dan langkah tindakan. Terjemahkan instruksi bahasa Inggris ke Bahasa Indonesia (contoh: "Switch device off and on" -> "Matikan lalu nyalakan kembali perangkat", "Replace main circuit board" -> "Ganti papan sirkuit utama / main circuit board", "Check wiring" -> "Periksa pengkabelan").
2. Pertahankan semua kode teknis, kode alarm, nama parameter, nilai angka/satuan (seperti 500 Ohm, +20%/-15%, 24 V DC), dan nomor terminal/jumper persis seperti yang tertulis di kutipan manual.
3. Jawab HANYA berdasarkan fakta di dalam "Kutipan Manual" yang diberikan. DILARANG mengarang prosedur di luar kutipan.
4. Perhatikan tanda positif (+) untuk batas berlebih (overweight, contoh: REC.T1+, REC.T2+, REC.TOL+) dan tanda negatif (-) untuk batas kurang (underweight, contoh: REC.T1-, REC.T2-, REC.TOL-). Jangan tertukar.
5. FILTER MEREK KETAT: Jika pertanyaan operator secara eksplisit menyebutkan spesifikasi, konfigurasi, atau perbaikan dari MEREK, SOFTWARE, atau PROTOKOL LAIN di luar Bühler / MEAG (contoh: Danfoss, Siemens, Wonderware, Atlas Copco, Modbus, TCP/IP eksternal), kamu WAJIB MENOLAK pertanyaan tersebut. Jawab HANYA dengan: "Maaf, saya hanya asisten teknis untuk kontroler dan peralatan Bühler MEAG. Pertanyaan mengenai perangkat pihak ketiga berada di luar cakupan saya."
6. Jangan pernah mengulang-ulang kalimat atau langkah yang sama.

Format jawaban yang wajib diikuti:
Kesimpulan:
[1-3 kalimat penjelasan ringkas mengenai arti alarm, fungsi parameter, atau spesifikasi teknis dalam Bahasa Indonesia yang natural]

Tindakan yang disarankan:
1. [Langkah pemeriksaan atau pengaturan pertama dalam Bahasa Indonesia sesuai kutipan]
2. [Langkah pemeriksaan berikutnya dalam Bahasa Indonesia sesuai kutipan]"""


ANSWER_PROMPT_TEMPLATE = """\
Kutipan Manual Bühler MEAG:
{context}

Pertanyaan Teknisi:
{question}

Tugas:
Berdasarkan Kutipan Manual di atas, tuliskan "Kesimpulan:" dan "Tindakan yang disarankan:" dalam Bahasa Indonesia yang natural dan akurat:"""


def _deduplicate_lines(text: str) -> str:
    """Remove repetitive looping lines if a small model gets stuck."""
    lines = text.splitlines()
    cleaned: List[str] = []
    seen_normalized: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        # Normalize numbered prefix like "7. Set TCON..." -> "set tcon..."
        norm = re.sub(r"^\d+[\.\)]\s*", "", stripped).lower()
        if len(norm) > 12 and norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _build_context(results: List["RetrievalResult"], max_chars: int = 2800) -> str:
    """Format high-confidence PDF retrieval results into a context string for the local LLM."""
    parts = []
    current_chars = 0
    for i, result in enumerate(results, 1):
        if getattr(result, "source_type", None) != "pdf" and result.chunk.get("source_type") != "pdf":
            continue
        text_content = result.text.strip()
        part = (
            f"[Kutipan {i} — {result.doc_title}, Hal. {result.page}]\n"
            f"{text_content}"
        )
        if current_chars + len(part) > max_chars and parts:
            break
        parts.append(part)
        current_chars += len(part)

    return "\n\n".join(parts)


def _format_raw_fallback(question: str, results: List["RetrievalResult"]) -> str:
    """Fallback: format retrieval results as clean structured text without LLM."""
    lines = [
        f"Jawaban untuk: {question}",
        "",
        "Berikut informasi relevan yang ditemukan dari manual Bühler:",
        "",
    ]
    rank = 1
    for result in results:
        if getattr(result, "source_type", None) != "pdf" and result.chunk.get("source_type") != "pdf":
            continue
        lines.append(f"{rank}. [{result.doc_title} — Hal. {result.page}]")
        wrapped = textwrap.fill(result.text.strip(), width=72)
        lines.append(wrapped)
        lines.append("")
        rank += 1
    return "\n".join(lines)


def is_local_llm_available(base_url: str = "http://127.0.0.1:8080", timeout_s: float = 2.0) -> bool:
    """Quick non-blocking health check to determine if llama-server is responding."""
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "MachineAssistant/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        models_url = f"{base_url.rstrip('/')}/v1/models"
        try:
            req = urllib.request.Request(models_url, headers={"User-Agent": "MachineAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False


def call_llama_server(
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 450,
    timeout_s: int = 180,
) -> str:
    """Send chat completion request to llama-server OpenAI-compatible API."""
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "repeat_penalty": 1.15,
        "frequency_penalty": 0.2,
        "max_tokens": max_tokens,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data_bytes,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "MachineAssistant/1.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        if response.status != 200:
            raise ValueError(f"HTTP status {response.status}")
        raw_body = response.read().decode("utf-8")
        data = json.loads(raw_body)
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Respons llama-server tidak memuat pilihan jawaban (choices kosong)")
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise ValueError("Respons jawaban dari Local LLM kosong")
        return _deduplicate_lines(content)


def generate_local_answer(
    question: str,
    results: List["RetrievalResult"],
    config: Optional["LocalLLMConfig"] = None,
) -> tuple[str, bool]:
    """Generate answer from PDF manual retrieval results using Local LLM (llama-server)."""
    # 1. Hard Guardrail for PII / Out-of-scope administrative queries
    oos_reason = check_out_of_scope(question)
    if oos_reason:
        return oos_reason, False
    if _OUT_OF_SCOPE_QUERY_RE.search(question):
        return (
            "Maaf, informasi mengenai identitas personel (Nama/NIK), administrasi HRD, "
            "atau peralatan di luar sistem kontrol mesin tidak tercakup dalam buku panduan manual Bühler "
            "dan dibatasi oleh kebijakan privasi data operasional.",
            False,
        )

    # 2. Strict Privacy & Relevance Score Filter:
    # Only keep PDF chunks that pass MIN_LLM_PDF_SCORE (or top-3 chunks if top score is strong).
    all_pdf = [
        r for r in results
        if getattr(r, "source_type", None) == "pdf" or r.chunk.get("source_type") == "pdf"
    ]

    if not all_pdf:
        return (
            "Tidak ditemukan informasi yang relevan dalam buku panduan (manual) "
            "untuk menjawab pertanyaan ini.",
            False,
        )

    top_score = max(getattr(r, "score", 0.0) for r in all_pdf)
    if top_score < MIN_LLM_PDF_SCORE:
        return (
            "Informasi spesifik mengenai hal ini tidak ditemukan di dalam buku panduan (manual) "
            "kontroler Bühler MEAG yang tersedia.",
            False,
        )

    # Filter out noisy low-score tail chunks (keep chunks with score >= 35% of top_score)
    pdf_results = [
        r for r in all_pdf
        if getattr(r, "score", 0.0) >= max(MIN_LLM_PDF_SCORE, top_score * 0.35)
    ][:4]

    # Resolve configuration
    base_url = "http://127.0.0.1:8080"
    model = "qwen2.5-1.5b-instruct"
    temperature = 0.2
    max_tokens = 450
    timeout_s = 180

    if config is not None:
        base_url = getattr(config, "base_url", base_url)
        model = getattr(config, "model", model)
        temperature = getattr(config, "temperature", temperature)
        max_tokens = getattr(config, "max_tokens", max_tokens)
        timeout_s = getattr(config, "timeout_s", timeout_s)

    context = _build_context(pdf_results)
    user_prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)

    try:
        answer = call_llama_server(
            base_url=base_url,
            model=model,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        return answer, True
    except urllib.error.HTTPError as exc:
        print(
            f"\n  [WARN] Local LLM mengembalikan error HTTP {exc.code}: {exc.reason}. "
            "Menampilkan hasil retrieval langsung sebagai fallback.\n"
        )
        raw = _format_raw_fallback(question, pdf_results)
        return raw, False
    except (urllib.error.URLError, ConnectionRefusedError) as exc:
        print(
            f"\n  [WARN] Local LLM tidak dapat dihubungi di {base_url} ({exc.reason if hasattr(exc, 'reason') else exc}). "
            "Menampilkan hasil retrieval langsung sebagai fallback.\n"
        )
        raw = _format_raw_fallback(question, pdf_results)
        return raw, False
    except (socket.timeout, TimeoutError):
        print(
            f"\n  [WARN] Local LLM timeout setelah {timeout_s} detik. "
            "Menampilkan hasil retrieval langsung sebagai fallback.\n"
        )
        raw = _format_raw_fallback(question, pdf_results)
        return raw, False
    except Exception as exc:
        print(
            f"\n  [WARN] Gagal menghasilkan jawaban dari Local LLM ({exc}). "
            "Menampilkan hasil retrieval langsung sebagai fallback.\n"
        )
        raw = _format_raw_fallback(question, pdf_results)
        return raw, False
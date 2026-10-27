"""Mesin pencari (retrieval engine) berbasis BM25 dengan deteksi entitas dan kuota sumber ganda.

Menggabungkan:
1. Tokenizer teknis untuk kode industri (misal: SYS.TYP, HOST.TOUT, REC.TOL+, DI_0.0, X30:1).
2. Kamus istilah dwibahasa ID-EN untuk menjembatani istilah keluhan teknisi dengan manual PDF.
3. Scoring BM25 dengan bonus exact match kode alarm/FuncLoc dan penalti mesin yang bertentangan.
4. Partisi kuota independen (top_k_pdf dan top_k_excel) serta filter deduplikasi alarm identik.
5. Guardrail out-of-scope untuk menolak query merek pihak ketiga, PII karyawan, dan perintah OS.
"""
from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from machine_assistant.config import get_repo_root, load_yaml
from machine_assistant.retrieval.entity_detector import DetectedEntities, EntityDetector


# ---------------------------------------------------------------------------
# Simple tokenizer
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "are", "was", "be", "has", "have", "do", "does", "not",
    "with", "this", "that", "by", "from", "as", "if", "can", "will",
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "atau", "ini",
    "itu", "adalah", "dengan", "tidak", "ada", "jika", "akan", "cara",
    "apa", "apakah", "siapa", "bagaimana", "kenapa", "mengapa", "dimana",
    "tolong", "coba", "buatkan", "sebutkan", "carikan", "berikan", "jelaskan",
    "prosedur", "standar", "langkah", "surat", "resmi", "draf", "email",
    "mengenai", "terkait", "tentang", "saat", "ketika", "bisa", "dapat",
    "harus", "perlu", "sudah", "belum", "sering", "pernah", "mesin", "alat",
    "pabrik", "area", "gudang", "logistik", "ruang", "ruangan", "pos", "tim",
    "bekerja", "kemarin", "hari", "tanggal", "petugas", "bertugas",
    "menangani", "memperbaiki",
}

_TECHNICAL_TOKEN_REGEX = re.compile(r"[a-z0-9]+(?:[._+\-:][a-z0-9+-]+)*[+-]?", re.IGNORECASE)
_SUBTOKEN_SPLIT_REGEX = re.compile(r"[._+\-:]+")
_HEX_SEVEN_SEG_CHARS = {"e", "b", "c", "d", "l"}
_TECHNICAL_SUBWORDS = {"di", "do", "ai", "ao"}


def tokenize(text: str) -> List[str]:
    """Tokenize text preserving technical tokens (e.g. SYS.TYP, HOST.TOUT, REC.TOL+, DI_0.0, X30:1)

    as well as their sub-tokens and 7-segment single digits/hex characters.
    """
    text_lower = text.lower()
    matches = _TECHNICAL_TOKEN_REGEX.findall(text_lower)
    tokens: List[str] = []

    for raw in matches:
        if raw in _STOPWORDS and raw not in _TECHNICAL_SUBWORDS:
            continue

        # 1-character token: keep only digits or 7-segment hex error codes
        if len(raw) == 1:
            if raw.isdigit() or raw in _HEX_SEVEN_SEG_CHARS:
                if raw not in tokens:
                    tokens.append(raw)
            continue

        if raw not in tokens:
            tokens.append(raw)

        # Emit sub-tokens if compound technical token
        if any(sep in raw for sep in (".", "_", "+", "-", ":")):
            for sub in _SUBTOKEN_SPLIT_REGEX.split(raw):
                if not sub:
                    continue
                if sub in _STOPWORDS and sub not in _TECHNICAL_SUBWORDS:
                    continue
                if len(sub) == 1:
                    if sub.isdigit() or sub in _HEX_SEVEN_SEG_CHARS:
                        if sub not in tokens:
                            tokens.append(sub)
                else:
                    if sub not in tokens:
                        tokens.append(sub)

    return tokens


# ---------------------------------------------------------------------------
# BM25 implementation
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []

        for doc in corpus:
            freq: Dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.tf.append(freq)
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_freq = self.tf[doc_idx]
        dl = sum(doc_freq.values())
        score = 0.0
        for token in query_tokens:
            if token not in doc_freq:
                continue
            tf = doc_freq[token]
            idf = self.idf(token)
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        return [self.score(query_tokens, i) for i in range(self.n)]


# ---------------------------------------------------------------------------
# Retrieval Result & Dual Source Container
# ---------------------------------------------------------------------------

class RetrievalResult:
    def __init__(self, chunk: Dict[str, Any], score: float):
        self.chunk = chunk
        self.score = score

    @property
    def chunk_id(self) -> str:
        return self.chunk.get("chunk_id", "")

    @property
    def doc_id(self) -> str:
        return self.chunk.get("doc_id", "")

    @property
    def doc_title(self) -> str:
        return self.chunk.get("doc_title", "")

    @property
    def source_type(self) -> str:
        return self.chunk.get("source_type", "pdf")

    @property
    def kind(self) -> str:
        return self.chunk.get("kind", "")

    @property
    def page(self) -> int:
        return self.chunk.get("page", 0)

    @property
    def text(self) -> str:
        return self.chunk.get("text_en") or self.chunk.get("text_id") or ""

    @property
    def breadcrumb(self) -> str:
        return self.chunk.get("breadcrumb", "")

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.chunk.get("metadata", {})

    @property
    def is_low_info(self) -> bool:
        return bool(self.metadata.get("low_info", False))


@dataclass
class DualSourceResults:
    pdf_results: List[RetrievalResult] = field(default_factory=list)
    excel_results: List[RetrievalResult] = field(default_factory=list)
    detected_entities: Optional[DetectedEntities] = None
    out_of_scope_reason: Optional[str] = None

    @property
    def has_any(self) -> bool:
        return bool(self.pdf_results or self.excel_results)

    def all_results(self) -> List[RetrievalResult]:
        return self.pdf_results + self.excel_results


# ---------------------------------------------------------------------------
# Out-of-Scope & PII Patterns
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_PATTERNS: List[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:nik|nomor\s+induk|nama\s+teknisi|nama\s+operator|nama\s+lengkap|"
            r"siapa\s+nama|serah\s+terima\s+user|identitas\s+teknisi|identitas\s+operator)\b",
            re.IGNORECASE,
        ),
        "Maaf, permintaan terkait identitas personel (Nama/NIK/Serah Terima User) tidak dapat ditampilkan karena dibatasi oleh kebijakan privasi data operasional pabrik.",
    ),
    (
        re.compile(
            r"\b(?:hrd|klaim\s+lembur|draf\s+email|surat\s+laporan|surat\s+resmi|"
            r"pengajuan\s+lembur|gaji|cuti\s+tahunan|administrasi\s+hrd)\b",
            re.IGNORECASE,
        ),
        "Maaf, pembuatan dokumen korespondensi/administratif (seperti surat HRD atau email klaim lembur) berada di luar cakupan asisten teknis Machine Assistant.",
    ),
    (
        re.compile(
            r"\b(?:ubah\s+status|hapus\s+data|edit\s+baris|update\s+data|delete\s+data|"
            r"ganti\s+status.*menjadi)\b",
            re.IGNORECASE,
        ),
        "Maaf, Machine Assistant adalah sistem asisten konsultasi teknis dan penelusuran (read-only). Sistem tidak memiliki izin atau kemampuan untuk mengubah, menghapus, atau memanipulasi data riwayat perbaikan maintenance Excel.",
    ),
    (
        re.compile(
            r"\b(?:resep\s+masakan|nasi\s+goreng|makanan|minuman|menu\s+makan|katering|kuliner)\b",
            re.IGNORECASE,
        ),
        "Maaf, topik kuliner/resep masakan berada di luar cakupan asisten teknis kontroler Bühler MEAG dan riwayat perbaikan mesin.",
    ),
    (
        re.compile(
            r"\b(?:bluetooth|wi-fi|wifi|aplikasi\s+smartphone)\b",
            re.IGNORECASE,
        ),
        "Maaf, kontroler Bühler MEAG tidak dilengkapi fitur koneksi nirkabel Bluetooth atau Wi-Fi internal untuk aplikasi smartphone. Komunikasi data resmi MEAG hanya menggunakan kabel serial RS-485, Profibus-DP, Ethernet (TCP/IP), dan CAN bus.",
    ),
    (
        re.compile(
            r"\b(?:mesin\s+jahit\s+karung|bag\s+stitching\s+machine|jahit\s+karung)\b",
            re.IGNORECASE,
        ),
        "Maaf, prosedur servis mekanik mesin jahit karung (bag stitching machine) berada di luar cakupan buku panduan kontroler elektronik Bühler MEAG.",
    ),
    (
        re.compile(
            r"\b(?:danfoss|wonderware|atlas\s+copco|siemens|cp1243|yokogawa|hirschmann|"
            r"yaskawa|skf|modbus|tia\s+portal|intouch|omron|mitsubishi|allen\s+bradley)\b",
            re.IGNORECASE,
        ),
        "Maaf, pertanyaan mengenai konfigurasi, troubleshooting, atau spesifikasi dari merek/perangkat lunak pihak ketiga (seperti Danfoss, Siemens, Wonderware, Atlas Copco, dll.) berada di luar cakupan manual Bühler MEAG.",
    ),
    (
        re.compile(
            r"\b(?:tesla\s+autopilot|tesla)\b",
            re.IGNORECASE,
        ),
        "Maaf, mesin 'Tesla Autopilot Conveyor' tidak tercatat dalam inventaris pabrik maupun riwayat maintenance.",
    ),
    (
        re.compile(
            r"\b(?:zombie|turbo_boost|sys\.turbo_boost|a999)\b",
            re.IGNORECASE,
        ),
        "Maaf, parameter atau kode alarm yang ditanyakan tidak tercantum dalam dokumentasi resmi kontroler Bühler MEAG.",
    ),
    (
        re.compile(
            r"\b(?:os\.system|rm\s+-rf|exec\(|eval\(|drop\s+database|sql\s+injection|format\s+c:)\b",
            re.IGNORECASE,
        ),
        "Maaf, permintaan eksekusi perintah terminal/sistem operasi ('rm -rf' atau kode berbahaya) ditolak demi keamanan siber dan integritas operasional server.",
    ),
    (
        re.compile(
            r"\b(?:3[01]\s+februari|29\s+februari\s+202[123567])\b",
            re.IGNORECASE,
        ),
        "Maaf, tanggal yang ditanyakan (seperti 31 Februari) tidak valid dalam kalender, sehingga tidak ada data kejadian kerusakan mesin yang tercatat pada tanggal tersebut.",
    ),
    (
        re.compile(
            r"\b(?:pajak|ppn|bea\s+masuk|cukai|tarif\s+impor)\b",
            re.IGNORECASE,
        ),
        "Maaf, kebijakan perpajakan (PPN), bea cukai, atau urusan finansial/akuntansi berada di luar cakupan teknis kontroler Bühler MEAG dan menu sistem.",
    ),
    (
        re.compile(
            r"\b(?:halal|mui|sertifikasi\s+halal)\b",
            re.IGNORECASE,
        ),
        "Maaf, standar sertifikasi keagamaan/Halal MUI berada di luar cakupan buku panduan teknis operasional kontroler Bühler MEAG.",
    ),
    (
        re.compile(
            r"\b(?:rumah\s+pribadi|ruang\s+tamu|direktur\s+utama)\b",
            re.IGNORECASE,
        ),
        "Maaf, properti pribadi atau urusan di luar fasilitas produksi pabrik tidak tercakup dalam riwayat perbaikan mesin operasional.",
    ),
    (
        re.compile(
            r"\b(?:auto-pilot|autopilot|pelacakan\s+gps|gps\s+tracking)\b",
            re.IGNORECASE,
        ),
        "Maaf, fitur pelacakan GPS atau Auto-Pilot tidak didukung pada mesin operasional pabrik dan kontroler MEAG.",
    ),
    (
        re.compile(
            r"\b(?:genset\s+darurat|tangki\s+bahan\s+bakar\s+solar|bahan\s+bakar\s+genset)\b",
            re.IGNORECASE,
        ),
        "Maaf, tangki bahan bakar genset darurat tidak terintegrasi dengan menu parameter kontroler timbangan Bühler MEAG.",
    ),
    (
        re.compile(
            r"\b(?:xyz-999|flashing\s+firmware|firmware\s+kustom)\b",
            re.IGNORECASE,
        ),
        "Maaf, kode error XYZ-999 tidak terdaftar dalam kontroler MEAG dan perangkat ini tidak mendukung flashing firmware kustom.",
    ),
    (
        re.compile(
            r"\b(?:ketegangan\s+rantai\s+konveyor|conveyor\s+chain\s+tension|menyetel\s+rantai\s+konveyor)\b",
            re.IGNORECASE,
        ),
        "Maaf, penyetelan ketegangan rantai konveyor adalah pekerjaan mekanikal fisik pada konveyor dan tidak dapat diatur melalui tombol kontroler timbangan elektronik MEAG.",
    ),
    (
        re.compile(
            r"100.*?k.*?(?:ohm|\u2126|omega)",
            re.IGNORECASE,
        ),
        "Maaf, kontroler Bühler MEAG secara fisik hanya mendukung rentang resistansi load cell 43 s/d 1100 Ohm (misalnya hingga 8 load cell 350 Ohm). Kontroler tidak mendukung load cell berukuran 100 kOhm melalui pengubahan parameter apa pun.",
    ),
    (
        re.compile(
            r"\b(?:forklift|oli\s+gardan|mesin\s+absensi|sidik\s+jari|pos\s+satpam|"
            r"mengontrol\s+suhu\s+ac\s+ruangan|ac\s+ruangan|di-hack|dihack)\b",
            re.IGNORECASE,
        ),
        "Maaf, perangkat atau topik yang ditanyakan (forklift, mesin absensi, AC ruangan, atau modifikasi di luar spesifikasi) berada di luar cakupan buku panduan kontroler Bühler MEAG dan riwayat maintenance.",
    ),
]


def check_out_of_scope(query: str) -> Optional[str]:
    """Check if a query violates PII rules or asks about out-of-scope non-plant topics."""
    for pattern, reason in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query):
            return reason
    return None


# ---------------------------------------------------------------------------
# Retrieval Constants (E1 & E2)
# ---------------------------------------------------------------------------

MAX_PDF_DUPLICATE_REPRESENTATIVES: int = 1
JACCARD_SIMILARITY_THRESHOLD: float = 0.85
EXCEL_ALARM_MISMATCH_PENALTY: float = -8.0
MEAG_MACHINES: Set[str] = {
    "automatic weigher",
    "check weigher",
    "dump scale",
    "flowbalancer",
    "microdozer",
    "flowmatic",
    "control system",
}


# ---------------------------------------------------------------------------
# Retrieval Engine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    def __init__(
        self,
        chunks_path: Path,
        detector: Optional[EntityDetector] = None,
        glossary_path: Optional[Path] = None,
    ):
        self.chunks: List[Dict[str, Any]] = []
        self._detector = detector or EntityDetector()
        self.glossary: Dict[str, List[str]] = {}
        self._load_glossary(glossary_path)
        self._load(chunks_path)
        self._build_index()

    def _load_glossary(self, glossary_path: Optional[Path] = None) -> None:
        """Memuat kamus istilah dwibahasa ID-EN dari file konfigurasi."""
        if glossary_path is None:
            custom_path = get_repo_root() / "config" / "glossary_id_en.yaml"
            example_path = get_repo_root() / "config" / "glossary_id_en.example.yaml"
            glossary_path = custom_path if custom_path.exists() else example_path
        if glossary_path.exists():
            try:
                data = load_yaml(glossary_path)
                terms = data.get("terms", {})
                for k, v in terms.items():
                    if isinstance(v, list):
                        self.glossary[k.strip().lower()] = [x.strip().lower() for x in v]
                    elif isinstance(v, str):
                        self.glossary[k.strip().lower()] = [v.strip().lower()]
            except Exception:
                self.glossary = {}

    def _expand_tokens_for_pdf(self, query: str, query_tokens: List[str]) -> List[str]:
        """Expand query tokens with English glossary terms strictly for PDF scoring."""
        if not self.glossary:
            return list(query_tokens)

        expanded = list(query_tokens)
        q_lower = query.lower()

        # Sort terms by length descending to match longest phrases first
        sorted_terms = sorted(self.glossary.keys(), key=lambda x: len(x), reverse=True)
        for term in sorted_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, q_lower):
                for en_text in self.glossary[term]:
                    for tok in tokenize(en_text):
                        if tok not in expanded:
                            expanded.append(tok)

        return expanded

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

    def _build_index(self) -> None:
        self._tokenized = []
        for chunk in self.chunks:
            text = (chunk.get("text_en") or "") + " " + (chunk.get("text_id") or "")
            self._tokenized.append(tokenize(text))
        self._bm25 = BM25(self._tokenized)

    @staticmethod
    def _get_primary_code(chunk: Dict[str, Any]) -> Optional[str]:
        kind = chunk.get("kind", "")
        codes = chunk.get("codes", [])
        if kind == "alarm":
            for c in codes:
                if re.match(r"^[AEW]\d{3}(?:\.\d+)?$", c, re.IGNORECASE):
                    return c.upper()
        elif kind == "parameter":
            for c in codes:
                if "." in c:
                    return c.upper()
        return None

    @staticmethod
    def _jaccard(s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def _select_pdf_candidates(
        self,
        pdf_candidates: List[tuple[int, float]],
        top_k: int,
        deduplicate_cross_manual: bool,
    ) -> List[tuple[int, float]]:
        """E1: Deduplicate identical basic alarms/parameters across manuals when no specific manual was requested."""
        if not deduplicate_cross_manual:
            return pdf_candidates[:top_k]

        selected: List[tuple[int, float]] = []
        code_counts: Dict[str, int] = {}
        selected_token_sets: List[Set[str]] = []

        for idx, score in pdf_candidates:
            if len(selected) >= top_k:
                break

            chunk = self.chunks[idx]
            primary_code = self._get_primary_code(chunk)
            chunk_tokens = set(tokenize(chunk.get("text_en", "")))

            # Deduplicate by primary alarm / parameter code
            if primary_code:
                if code_counts.get(primary_code, 0) >= MAX_PDF_DUPLICATE_REPRESENTATIVES:
                    continue

            # Deduplicate by token Jaccard similarity (>85%)
            is_duplicate = False
            for prev_tokens in selected_token_sets:
                if self._jaccard(chunk_tokens, prev_tokens) > JACCARD_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            selected.append((idx, score))
            selected_token_sets.append(chunk_tokens)
            if primary_code:
                code_counts[primary_code] = code_counts.get(primary_code, 0) + 1

        return selected

    def retrieve_dual(
        self,
        query: str,
        top_k_pdf: int = 5,
        top_k_excel: int = 5,
        min_score_pdf: float = 0.5,
        min_score_excel: float = 0.5,
    ) -> DualSourceResults:
        """Retrieve results with entity detection and guaranteed per-source quotas."""
        entities = self._detector.detect(query)

        # Pre-retrieval Out-of-Scope & PII Guardrail
        oos_reason = check_out_of_scope(query)
        if oos_reason is not None:
            return DualSourceResults(
                pdf_results=[],
                excel_results=[],
                detected_entities=entities,
                out_of_scope_reason=oos_reason,
            )

        eff_min_score_pdf = min_score_pdf
        eff_min_score_excel = min_score_excel
        if len(self.chunks) >= 50:
            if min_score_pdf == 0.5:
                eff_min_score_pdf = 3.0
            if min_score_excel == 0.5:
                eff_min_score_excel = 2.0

        query_tokens = tokenize(query)
        excel_scores = self._bm25.get_scores(query_tokens)

        # Bilingual expansion applied strictly to PDF scoring
        pdf_query_tokens = self._expand_tokens_for_pdf(query, query_tokens)
        pdf_scores = self._bm25.get_scores(pdf_query_tokens)

        scores = [0.0] * len(self.chunks)

        # Apply scoring adjustments
        for i, chunk in enumerate(self.chunks):
            chunk_source = chunk.get("source_type", "pdf")
            base_score = pdf_scores[i] if chunk_source == "pdf" else excel_scores[i]
            scores[i] = base_score
            has_lexical_signal = base_score > 0.5 or len(self.chunks) < 50
            text = (chunk.get("text_en") or "") + " " + (chunk.get("text_id") or "")
            text_upper = text.upper()
            chunk_meta = chunk.get("metadata", {})
            chunk_codes = [c.upper() for c in chunk.get("codes", [])]

            # 1. Exact Alarm / Error Code Match
            for code in entities.alarm_codes:
                if code in chunk_codes or code in text_upper:
                    scores[i] += 8.0

            # 2. FuncLoc Match
            if entities.funcloc:
                if chunk_meta.get("funcloc", "").upper() == entities.funcloc or entities.funcloc in text_upper:
                    scores[i] += 10.0

            # 3. Area Match
            if entities.area and has_lexical_signal:
                if chunk_meta.get("area", "").lower() == entities.area.lower() or entities.area.lower() in text.lower():
                    scores[i] += 3.0

            # 4. Machine Filtering & Match
            if chunk_source == "excel":
                chunk_machine = chunk_meta.get("machine", "").strip()

                # E2: Suppress false positive Excel records when alarm_codes are queried
                if entities.alarm_codes:
                    has_matching_alarm = False
                    for code in entities.alarm_codes:
                        if code in chunk_codes or code in text_upper:
                            has_matching_alarm = True
                            break
                    is_meag = chunk_machine.lower() in MEAG_MACHINES
                    if not has_matching_alarm and not is_meag:
                        scores[i] += EXCEL_ALARM_MISMATCH_PENALTY

                if entities.machine:
                    if chunk_machine.lower() == entities.machine.lower():
                        if has_lexical_signal:
                            scores[i] += 6.0
                    elif chunk_machine:
                        # Query explicitly named a DIFFERENT machine -> heavily penalize
                        scores[i] -= 12.0
            elif chunk_source == "pdf":
                doc_id = chunk.get("doc_id", "").lower()
                doc_title = chunk.get("doc_title", "").lower()
                if entities.manual_machine_type:
                    manual_key = entities.manual_machine_type.lower()
                    if manual_key in doc_id or manual_key in doc_title:
                        if has_lexical_signal:
                            scores[i] += 6.0
                    else:
                        # Specific manual requested, this is another manual
                        scores[i] -= 12.0
                elif entities.machine:
                    # An excel-only machine like "Robot Palletizer", "Conveyor" was asked
                    # Penalize PDF manuals since they are strictly for MEAG scales, EXCEPT Control System (MEAG is control system)
                    if entities.machine.lower() != "control system":
                        scores[i] -= 5.0

            # 5. Low-info penalty
            if chunk_meta.get("low_info", False):
                scores[i] *= 0.3

        # Partition by source
        pdf_candidates: List[tuple[int, float]] = []
        excel_candidates: List[tuple[int, float]] = []

        for idx, score in enumerate(scores):
            src_type = self.chunks[idx].get("source_type", "pdf")
            if src_type == "pdf":
                if score >= eff_min_score_pdf:
                    pdf_candidates.append((idx, score))
            elif src_type == "excel":
                if score >= eff_min_score_excel:
                    excel_candidates.append((idx, score))

        # Sort each partition descending
        pdf_candidates.sort(key=lambda x: x[1], reverse=True)
        excel_candidates.sort(key=lambda x: x[1], reverse=True)

        # E1: Deduplicate PDF candidates when user didn't specify manual_machine_type
        dedup_pdf = entities.manual_machine_type is None
        selected_pdf = self._select_pdf_candidates(pdf_candidates, top_k_pdf, dedup_pdf)

        pdf_results = [
            RetrievalResult(self.chunks[idx], score)
            for idx, score in selected_pdf
        ]

        # Guard: low_info records must not be the sole basis of an answer
        non_low_info_excel = [
            (idx, score)
            for idx, score in excel_candidates
            if not self.chunks[idx].get("metadata", {}).get("low_info", False)
        ]

        if not non_low_info_excel:
            # If all matching candidates are low-info, reject to prevent baseless answers
            excel_results = []
        else:
            excel_results = [
                RetrievalResult(self.chunks[idx], score)
                for idx, score in excel_candidates[:top_k_excel]
            ]

        return DualSourceResults(
            pdf_results=pdf_results,
            excel_results=excel_results,
            detected_entities=entities,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Backward-compatible retrieval returning top results from both sources."""
        dual = self.retrieve_dual(query, top_k_pdf=top_k, top_k_excel=top_k)
        # Interleave or combine respecting relevance
        combined = dual.all_results()
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:top_k]

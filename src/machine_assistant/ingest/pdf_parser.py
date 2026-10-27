"""Parser dokumen PDF manual teknis Bühler MEAG.

Melakukan ekstraksi teks dengan PyMuPDF, pembersihan header/footer berulang,
penyambungan kata terpenggal tanda hubung (dehyphenation), decoding font khusus MEAG DUMP,
ekstraksi kode alarm dan parameter, serta chunking berbasis struktur halaman dan tabel.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pymupdf

from machine_assistant.ingest.base_parser import (
    BaseParser,
    Chunk,
    ChunkKind,
    ParserRegistry,
    SourceDocument,
)

_UNITS: Set[str] = {
    "V", "A", "MA", "HZ", "KG", "KW", "MM", "MS", "S", "MIN", "H", "BAR", "VAC", "VDC", "OHM", "BIT", "BITS",
}

_PARAM_FAMILIES = (
    "USER", "MAIN", "REC", "TCON", "SYS", "ADC", "HOST", "ENET", "TCP", "PBDP", "SERV", "PRN",
)
_PARAM_FAMILY_PREFIXES = tuple(f"{p}." for p in _PARAM_FAMILIES)

_DEFAULT_CODE_PATTERNS: Dict[str, str] = {
    "alarm": r"\b([AEW]\d{2,4}(?:\.\d+)?)\b",
    "parameter": r"\b([A-Z]{2,5}\.[A-Z0-9_\+\-]+(?:\.[x0-9])?)\b",
    "io_signal": r"\b((?:DI|DO|AI|AO)[_\s]?\d+\.\d+)\b",
    "terminal": r"\b(X\d{1,3}(?::\d{1,2})?)\b",
    "seven_seg_error": r"\b(E\s?[0-9A-Za-z])\b|\b(no\s+comm\s+an)\b",
}

_HEADER_FOOTER_PATTERNS: List[re.Pattern] = [
    re.compile(r"©\s*Copyright\s+.*Bühler\s+AG.*", re.IGNORECASE),
    re.compile(r"(?:©\s*)?Copyright\s+(?:by\s+|\d{4}\s+)?Bühler\s+AG.*", re.IGNORECASE),
    re.compile(r"^Bühler\s+AG(?:\s+MEAG-[A-Za-z0-9\-]+)?.*", re.IGNORECASE),
    re.compile(r"^MEAG-[A-Za-z0-9\-]+.*", re.IGNORECASE),
    re.compile(r"^Chapter\s+\d+\b.*", re.IGNORECASE),
    re.compile(r"^Table\s+of\s+contents\b.*", re.IGNORECASE),
    re.compile(r"^\d{1,3}$"),  # Standalone page number
    re.compile(
        r"^(?:Checkweigher\s*\(CHECK\)|Differential\s+dosing\s+scale\s*\(DIFF\)|Automatic\s+volume\s+controller\s*\(FBAL\)).*",
        re.IGNORECASE,
    ),
    re.compile(r"^\d{1,3}\s+(?:Bühler\s+AG\s+)?MEAG-[A-Za-z0-9\-]+.*", re.IGNORECASE),
    re.compile(r"^\d{1,3}\s+Alarms\s+and\s+messages.*", re.IGNORECASE),
    re.compile(
        r"^Differential\s+dosing\s+scale\s*\(DIFF\)\s+Alarms\s+and\s+messages\s+\d+.*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\d{1,3}\s+Bühler\s+AG\s+MEAG-[A-Za-z0-9\-]+(?:\s+Chapter\s+\d+\s+Alarms\s+and\s+messages)?.*",
        re.IGNORECASE,
    ),
]


def decode_dump_text(text: str, force: bool = False) -> str:
    """Decode font-shifted text found in MEAG_DUMP PDF (Identity-H ASCII offset +29)."""
    if not (force or "0($*" in text or "2SHUDWLQJ" in text or "FRQWURO" in text):
        return text
    char_map = {
        "ಘ": " ", "ತ": "-", "ದ": "±", "ಯ": '"', "ರ": '"',
        "ಹ": " ... ", "Ⴋ": "- ", "\x03": " ", "¾": "ü",
    }
    res: List[str] = []
    for ch in text:
        if ch in char_map:
            res.append(char_map[ch])
        elif ch in "\n\r\t ":
            res.append(ch)
        elif ord(ch) < 128:
            res.append(chr(ord(ch) + 29))
        else:
            res.append(ch)
    return "".join(res)


def dehyphenate(text: str) -> str:
    """Connect lowercase word parts split across line breaks.

    Preserves technical codes and compound terms like RS-485, DC-SCALE, DMS-T, slow-blow.
    """
    def _repl(m: re.Match) -> str:
        w1 = m.group(1)
        w2 = m.group(2)
        # Never join if first part is all-uppercase (e.g. IN, OUT, RS, DC)
        if w1.isupper():
            return m.group(0)
        return w1 + w2

    # Words split across line break
    text = re.sub(r"\b([a-zA-Z]{2,})-\s*\n\s*([a-z]{2,})\b", _repl, text)
    # Words split across space (from line breaks turned into spaces)
    text = re.sub(r"\b([a-zA-Z]{2,})-\s+([a-z]{2,})\b", _repl, text)
    return text


def detect_repeating_margin_lines(
    doc: pymupdf.Document,
    top_ratio: float = 0.07,
    bottom_ratio: float = 0.07,
    threshold: float = 0.40,
    min_pages: int = 3,
    is_shifted: bool = False,
) -> Set[str]:
    """Detect lines repeating across page margins (>=40% pages, min 3 pages, ignoring pure digits)."""
    line_page_counts: Dict[str, Set[int]] = {}
    total_pages = len(doc)
    for pno in range(total_pages):
        page = doc[pno]
        h = page.rect.height
        top_y = h * top_ratio
        bot_y = h * (1.0 - bottom_ratio)
        blocks = page.get_text("blocks")
        for b in blocks:
            y0, y1 = b[1], b[3]
            if y0 <= top_y or y1 >= bot_y:
                b_text = decode_dump_text(b[4], force=is_shifted)
                for line in b_text.splitlines():
                    s = line.strip().lower()
                    if not s or s.isdigit():
                        continue
                    if s not in line_page_counts:
                        line_page_counts[s] = set()
                    line_page_counts[s].add(pno)

    repeating: Set[str] = set()
    req_pages = max(min_pages, int(total_pages * threshold))
    for s, pages in line_page_counts.items():
        if len(pages) >= req_pages:
            repeating.add(s)
    return repeating


def is_margin_block_to_remove(
    block: tuple,
    top_y: float,
    bot_y: float,
    repeating_lines: Set[str],
    is_shifted: bool = False,
) -> bool:
    """Layer 1: discard text block in margin only if matching header/footer patterns or repeating lines."""
    y0, y1 = block[1], block[3]
    in_margin = (y1 <= top_y) or (y0 >= bot_y)
    if not in_margin:
        return False
    b_text = decode_dump_text(block[4], force=is_shifted).strip()
    if not b_text:
        return True
    for line in b_text.splitlines():
        s = line.strip().lower()
        if s in repeating_lines:
            return True
        for pat in _HEADER_FOOTER_PATTERNS:
            if pat.search(line.strip()):
                return True
    return False


def clean_line_layer2(line: str) -> bool:
    """Layer 2: return True if line should be kept, False if it is a header/footer line."""
    s = line.strip()
    if not s:
        return False
    for pat in _HEADER_FOOTER_PATTERNS:
        if pat.search(s):
            return False
    return True


def detect_section(lines: List[str]) -> str | None:
    """Detect section heading e.g. '9.2 Basic alarms' or '8.3.3 Parameter group HOST'.

    Avoids false positives like '24 V inputs'.
    """
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or "..." in s or "Table of contents" in s:
            continue
        # Case 1: single line e.g. "9.2 Basic alarms"
        m1 = re.match(r"^(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z0-9\s/\-\(\)\.,_]+)$", s)
        if m1:
            num = m1.group(1)
            title = m1.group(2).strip()
            first_word = title.split()[0].upper()
            if first_word not in _UNITS and not title.endswith("."):
                return f"{num} {title}"

        # Case 2: two-line heading: line i is "9.2", line i+1 is "Basic alarms"
        m2 = re.match(r"^(\d+(?:\.\d+)+)$", s)
        if m2 and i + 1 < len(lines):
            next_s = lines[i + 1].strip()
            if next_s and "..." not in next_s and len(next_s) <= 60:
                m_next = re.match(r"^([A-Z][A-Za-z0-9\s/\-\(\)\.,_]{2,})$", next_s)
                if m_next:
                    title = m_next.group(1).strip()
                    first_word = title.split()[0].upper()
                    if first_word not in _UNITS and not title.endswith("."):
                        return f"{m2.group(1)} {title}"

        # Case 3: chapter number e.g. "9" followed by "Alarms and messages"
        m3 = re.match(r"^(?:Chapter\s+)?(\d{1,2})$", s)
        if m3 and i + 1 < len(lines):
            next_s = lines[i + 1].strip()
            if next_s and "..." not in next_s and len(next_s) <= 60:
                m_next = re.match(r"^([A-Z][A-Za-z0-9\s/\-\(\)\.,_]{3,})$", next_s)
                if m_next:
                    title = m_next.group(1).strip()
                    first_word = title.split()[0].upper()
                    if first_word not in _UNITS and not title.endswith("."):
                        return f"{m3.group(1)} {title}"
    return None


def extract_and_normalize_codes(
    text: str,
    profile_config: Dict[str, Any] | None = None,
    extra_codes: List[str] | None = None,
) -> List[str]:
    """Extract and normalize codes (uppercase, without internal spaces, deduplicated)."""
    patterns = dict(_DEFAULT_CODE_PATTERNS)
    if profile_config and "code_patterns" in profile_config:
        patterns.update(profile_config["code_patterns"])

    found: List[str] = []

    def _add(code: str) -> None:
        norm = re.sub(r"\s+", "", code.upper().strip())
        if norm and norm not in found:
            found.append(norm)

    if extra_codes:
        for ec in extra_codes:
            _add(ec)

    for pat in patterns.values():
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw = m.group(0)
            if raw:
                _add(raw)

    return found


class PDFParser(BaseParser):
    """Bühler MEAG PDF Parser with two-layer header/footer removal, de-hyphenation,

    running section tracking, and granular alarm/parameter chunking.
    """

    def parse(
        self,
        file_path: Path,
        source_doc: SourceDocument,
        profile_config: Dict[str, Any],
    ) -> Iterable[Chunk]:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        chunks: List[Chunk] = []
        with pymupdf.open(file_path) as doc:
            # Check if document has font-shifted text (MEAG_DUMP)
            doc_sample = ""
            for s_idx in range(min(3, len(doc))):
                doc_sample += doc[s_idx].get_text("text")
            is_shifted = ("0($*" in doc_sample or "2SHUDWLQJ" in doc_sample or "FRQWURO" in doc_sample)

            # Pre-pass: Detect repeating margin lines across document
            repeating_margin_lines = detect_repeating_margin_lines(doc, is_shifted=is_shifted)

            current_section = ""
            alarm_start_regex = re.compile(
                r"^([AEW]\d{3}(?:\.\d+)?)(?:\s+([A-Za-z0-9_\+\-\s]{2,}))?",
            )
            param_start_regex = re.compile(
                r"^((?:USER|MAIN|REC|TCON|SYS|ADC|HOST|ENET|TCP|PBDP|SERV|PRN)\.[A-Z0-9_\+\-]+(?:\.[x0-9])?)\b",
            )

            for page_index in range(len(doc)):
                page_num = page_index + 1
                page = doc[page_index]
                h = page.rect.height
                top_y = h * 0.07
                bot_y = h * 0.93

                # Layer 1: Filter out margin blocks that match headers/footers
                blocks = page.get_text("blocks")
                valid_blocks_text: List[str] = []
                for b in blocks:
                    if is_margin_block_to_remove(b, top_y, bot_y, repeating_margin_lines, is_shifted=is_shifted):
                        continue
                    b_decoded = decode_dump_text(b[4], force=is_shifted)
                    valid_blocks_text.append(b_decoded)

                raw_page_text = "\n".join(valid_blocks_text)
                if not raw_page_text.strip():
                    continue

                # Layer 2: Filter out header/footer lines and apply de-hyphenation
                cleaned_lines: List[str] = []
                for line in raw_page_text.splitlines():
                    if clean_line_layer2(line):
                        cleaned_lines.append(line)

                page_text_before_dehyphen = "\n".join(cleaned_lines)
                page_text = dehyphenate(page_text_before_dehyphen)
                lines = [l.strip() for l in page_text.splitlines() if l.strip()]
                if not lines:
                    continue

                # Initial section check on page start
                detected_sec = detect_section(lines[:10])
                if detected_sec:
                    current_section = detected_sec

                # Check if page is an alarm page (2+ alarm starts at beginning of lines)
                alarm_starts: List[Tuple[int, str, str | None]] = []
                for i, l in enumerate(lines):
                    m = alarm_start_regex.match(l)
                    if m:
                        a_code = m.group(1).upper()
                        raw_name = m.group(2)
                        a_name = raw_name.strip() if raw_name else None
                        alarm_starts.append((i, a_code, a_name))

                # Check if page is a parameter page (2+ parameter starts)
                param_starts: List[Tuple[int, str]] = []
                if len(alarm_starts) < 2:
                    for i, l in enumerate(lines):
                        m = param_start_regex.match(l)
                        if m:
                            p_code = m.group(1).upper()
                            param_starts.append((i, p_code))

                # --- 1. Alarm Page Chunking ---
                if len(alarm_starts) >= 2:
                    # Optional intro before first alarm
                    first_alarm_line = alarm_starts[0][0]
                    intro_lines = lines[:first_alarm_line]
                    intro_text = " ".join(intro_lines).strip()
                    if len(intro_text.split()) > 15:
                        chunk_id = f"{source_doc.doc_id}_p{page_num:03d}_intro"
                        b_crumb = (
                            f"{source_doc.doc_title} > {current_section} > Hal. {page_num}"
                            if current_section
                            else f"{source_doc.doc_title} > Hal. {page_num}"
                        )
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            doc_id=source_doc.doc_id,
                            doc_title=source_doc.doc_title,
                            doc_version=source_doc.doc_version,
                            source_type="pdf",
                            page=page_num,
                            section=current_section,
                            breadcrumb=b_crumb,
                            kind="prose",
                            codes=extract_and_normalize_codes(intro_text, profile_config),
                            text_en=intro_text,
                        )
                        chunk.content_hash = chunk.compute_hash()
                        chunks.append(chunk)

                    seen_alarm_ids: Dict[str, int] = {}
                    for k in range(len(alarm_starts)):
                        idx, a_code, a_name = alarm_starts[k]
                        # Check if a new section heading was passed before this alarm
                        prev_line = alarm_starts[k - 1][0] if k > 0 else 0
                        sec_in_between = detect_section(lines[prev_line:idx])
                        if sec_in_between:
                            current_section = sec_in_between

                        next_idx = alarm_starts[k + 1][0] if k + 1 < len(alarm_starts) else len(lines)
                        a_lines = lines[idx:next_idx]
                        a_text = "\n".join(a_lines).strip()

                        # If alarm name wasn't on the same line, check line 2
                        if not a_name and len(a_lines) > 1:
                            cand = a_lines[1].strip()
                            if re.match(r"^[A-Z0-9_\+\-]{3,15}$", cand):
                                a_name = cand

                        extra = [a_code]
                        if a_name:
                            extra.append(a_name)
                        codes = extract_and_normalize_codes(a_text, profile_config, extra_codes=extra)

                        seen_alarm_ids[a_code] = seen_alarm_ids.get(a_code, 0) + 1
                        suffix = f"_{seen_alarm_ids[a_code]:02d}" if seen_alarm_ids[a_code] > 1 else ""
                        chunk_id = f"{source_doc.doc_id}_p{page_num:03d}_{a_code.lower()}{suffix}"

                        b_crumb = (
                            f"{source_doc.doc_title} > {current_section} > Hal. {page_num} [{a_code}]"
                            if current_section
                            else f"{source_doc.doc_title} > Hal. {page_num} [{a_code}]"
                        )

                        chunk = Chunk(
                            chunk_id=chunk_id,
                            doc_id=source_doc.doc_id,
                            doc_title=source_doc.doc_title,
                            doc_version=source_doc.doc_version,
                            source_type="pdf",
                            page=page_num,
                            section=current_section,
                            breadcrumb=b_crumb,
                            kind="alarm",
                            codes=codes,
                            text_en=a_text,
                        )
                        chunk.content_hash = chunk.compute_hash()
                        chunks.append(chunk)

                # --- 2. Parameter Page Chunking ---
                elif len(param_starts) >= 2:
                    first_p_line = param_starts[0][0]
                    intro_lines = lines[:first_p_line]
                    intro_text = " ".join(intro_lines).strip()
                    if len(intro_text.split()) > 15:
                        chunk_id = f"{source_doc.doc_id}_p{page_num:03d}_intro"
                        b_crumb = (
                            f"{source_doc.doc_title} > {current_section} > Hal. {page_num}"
                            if current_section
                            else f"{source_doc.doc_title} > Hal. {page_num}"
                        )
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            doc_id=source_doc.doc_id,
                            doc_title=source_doc.doc_title,
                            doc_version=source_doc.doc_version,
                            source_type="pdf",
                            page=page_num,
                            section=current_section,
                            breadcrumb=b_crumb,
                            kind="prose",
                            codes=extract_and_normalize_codes(intro_text, profile_config),
                            text_en=intro_text,
                        )
                        chunk.content_hash = chunk.compute_hash()
                        chunks.append(chunk)

                    seen_param_ids: Dict[str, int] = {}
                    for k in range(len(param_starts)):
                        idx, p_code = param_starts[k]
                        next_idx = param_starts[k + 1][0] if k + 1 < len(param_starts) else len(lines)
                        p_lines = lines[idx:next_idx]
                        p_text = "\n".join(p_lines).strip()

                        codes = extract_and_normalize_codes(p_text, profile_config, extra_codes=[p_code])

                        norm_code = p_code.lower().replace(".", "_")
                        seen_param_ids[norm_code] = seen_param_ids.get(norm_code, 0) + 1
                        suffix = f"_{seen_param_ids[norm_code]:02d}" if seen_param_ids[norm_code] > 1 else ""
                        chunk_id = f"{source_doc.doc_id}_p{page_num:03d}_{norm_code}{suffix}"

                        b_crumb = (
                            f"{source_doc.doc_title} > {current_section} > Hal. {page_num} [{p_code}]"
                            if current_section
                            else f"{source_doc.doc_title} > Hal. {page_num} [{p_code}]"
                        )

                        chunk = Chunk(
                            chunk_id=chunk_id,
                            doc_id=source_doc.doc_id,
                            doc_title=source_doc.doc_title,
                            doc_version=source_doc.doc_version,
                            source_type="pdf",
                            page=page_num,
                            section=current_section,
                            breadcrumb=b_crumb,
                            kind="parameter",
                            codes=codes,
                            text_en=p_text,
                        )
                        chunk.content_hash = chunk.compute_hash()
                        chunks.append(chunk)

                # --- 3. Prose / I/O Signal Page Chunking ---
                else:
                    full_p_text = "\n".join(lines).strip()
                    io_signals = re.findall(r"\b(?:DI|DO|AI|AO)[_\s]?\d+\.\d+\b", full_p_text, re.IGNORECASE)
                    kind: ChunkKind = "io_signal" if len(set(io_signals)) >= 4 else "prose"

                    words = full_p_text.split()
                    if len(words) <= 300:
                        chunk_id = f"{source_doc.doc_id}_p{page_num:03d}_c01"
                        b_crumb = (
                            f"{source_doc.doc_title} > {current_section} > Hal. {page_num}"
                            if current_section
                            else f"{source_doc.doc_title} > Hal. {page_num}"
                        )
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            doc_id=source_doc.doc_id,
                            doc_title=source_doc.doc_title,
                            doc_version=source_doc.doc_version,
                            source_type="pdf",
                            page=page_num,
                            section=current_section,
                            breadcrumb=b_crumb,
                            kind=kind,
                            codes=extract_and_normalize_codes(full_p_text, profile_config),
                            text_en=full_p_text,
                        )
                        chunk.content_hash = chunk.compute_hash()
                        chunks.append(chunk)
                    else:
                        window_size = 300
                        step = 250
                        c_idx = 1
                        for start_w in range(0, len(words), step):
                            chunk_words = words[start_w : start_w + window_size]
                            if not chunk_words:
                                break
                            chunk_text = " ".join(chunk_words)
                            chunk_id = f"{source_doc.doc_id}_p{page_num:03d}_c{c_idx:02d}"
                            b_crumb = (
                                f"{source_doc.doc_title} > {current_section} > Hal. {page_num}"
                                if current_section
                                else f"{source_doc.doc_title} > Hal. {page_num}"
                            )
                            chunk = Chunk(
                                chunk_id=chunk_id,
                                doc_id=source_doc.doc_id,
                                doc_title=source_doc.doc_title,
                                doc_version=source_doc.doc_version,
                                source_type="pdf",
                                page=page_num,
                                section=current_section,
                                breadcrumb=b_crumb,
                                kind=kind,
                                codes=extract_and_normalize_codes(chunk_text, profile_config),
                                text_en=chunk_text,
                            )
                            chunk.content_hash = chunk.compute_hash()
                            chunks.append(chunk)
                            c_idx += 1
                            if start_w + window_size >= len(words):
                                break

        return chunks


ParserRegistry.register("pdf", PDFParser)

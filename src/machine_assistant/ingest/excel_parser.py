"""Excel parser for daily maintenance reports (Laporan Harian).

Reads an XLSX file with openpyxl in read_only mode, detects new-record vs
continuation rows, merges tools into parent records, applies configurable
normalization (status, jenis kerusakan, typo corrections), and yields
Chunk objects ready for indexing.

Privacy: Name, NIK, and pelaksana columns are never read.
"""
from __future__ import annotations

import re
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import openpyxl
from openpyxl.utils import column_index_from_string

from machine_assistant.ingest.base_parser import (
    BaseParser,
    Chunk,
    ParserRegistry,
    SourceDocument,
)


def _col_letter_to_index(letter: str) -> int:
    """Convert Excel column letter (A, B, ..., AG) to 0-based index."""
    return column_index_from_string(letter) - 1


def _safe_str(value: Any) -> str:
    """Convert a cell value to stripped string, handling None and datetime."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    """Check if a cell value is blank or a dash placeholder."""
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s == "-"


class ExcelParser(BaseParser):
    """Parser for daily maintenance report XLSX files."""

    def parse(
        self,
        file_path: Path,
        source_doc: SourceDocument,
        profile_config: Dict[str, Any],
    ) -> Iterable[Chunk]:
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        # Load profile configuration
        col_map = profile_config.get("column_mapping", {})
        privacy_excluded = set(profile_config.get("privacy_excluded_columns", []))
        new_rec_fields = profile_config.get("new_record_columns", ["machine", "permasalahan"])
        continuation_fields = profile_config.get("continuation_merge_fields", ["tools"])
        empty_threshold = profile_config.get("empty_row_stop_threshold", 2000)
        data_start = profile_config.get("data_start_row", 5)
        sheet_name = profile_config.get("sheet_name", "Notification")

        # Normalization config
        status_map_raw = profile_config.get("status_mapping", {})
        status_map = {k.strip().lower(): v for k, v in status_map_raw.items()}
        jenis_map_raw = profile_config.get("jenis_kerusakan_mapping", {})
        jenis_map = {k.strip().lower(): v for k, v in jenis_map_raw.items()}
        typo_corrections = profile_config.get("typo_corrections", {})
        machine_aliases = profile_config.get("machine_aliases", {})
        low_info_patterns = [
            p.lower() for p in profile_config.get("low_info_patterns", [])
        ]
        year_range = profile_config.get("valid_year_range", {"min": 2020, "max": 2027})

        # Build column index lookup (0-based)
        col_idx: Dict[str, int] = {}
        for logical_name, col_letter in col_map.items():
            col_idx[logical_name] = _col_letter_to_index(col_letter)

        # Privacy check: make sure excluded columns are not in col_map
        for excl_letter in privacy_excluded:
            excl_idx = _col_letter_to_index(excl_letter)
            for name, idx in col_idx.items():
                if idx == excl_idx:
                    raise ValueError(
                        f"Privacy violation: column {excl_letter} ({name}) "
                        f"is in both column_mapping and privacy_excluded_columns"
                    )

        # Parse the Excel file
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        try:
            ws = wb[sheet_name]
        except KeyError:
            wb.close()
            raise ValueError(f"Sheet '{sheet_name}' not found in {file_path}")

        records: List[Dict[str, Any]] = []
        current_record: Optional[Dict[str, Any]] = None
        empty_streak = 0
        flagged_rows: List[Dict[str, Any]] = []

        for row_num, row in enumerate(
            ws.iter_rows(min_row=data_start, values_only=True), data_start
        ):
            # Extract values by logical name
            row_data: Dict[str, Any] = {}
            for logical_name, idx in col_idx.items():
                row_data[logical_name] = row[idx] if idx < len(row) else None

            # Check if this is a new record
            is_new_record = any(
                not _is_blank(row_data.get(field))
                for field in new_rec_fields
            )

            if is_new_record:
                # Finalize previous record
                if current_record is not None:
                    records.append(current_record)

                empty_streak = 0
                current_record = {
                    "_row_start": row_num,
                    "_row_end": row_num,
                }
                for logical_name in col_idx:
                    current_record[logical_name] = _safe_str(row_data[logical_name])

                # Tools as list: combine tools (O) and tools_n (N) without duplicates
                tools_vals = []
                for tf in ("tools", "tools_n", "tools_o"):
                    tv = _safe_str(row_data.get(tf))
                    if tv and tv not in tools_vals:
                        tools_vals.append(tv)
                current_record["_tools_list"] = tools_vals

            elif current_record is not None:
                # Continuation row: merge specified fields
                current_record["_row_end"] = row_num
                for field in continuation_fields:
                    val = _safe_str(row_data.get(field))
                    if val and val not in current_record["_tools_list"]:
                        current_record["_tools_list"].append(val)

                # Check if row is completely empty (for stop detection)
                all_empty = all(_is_blank(row_data.get(f)) for f in col_idx)
                if all_empty:
                    empty_streak += 1
                else:
                    empty_streak = 0
            else:
                # No current record yet, check empty streak
                all_empty = all(_is_blank(row_data.get(f)) for f in col_idx)
                if all_empty:
                    empty_streak += 1
                else:
                    empty_streak = 0

            if empty_streak >= empty_threshold:
                break

        # Finalize last record
        if current_record is not None:
            records.append(current_record)

        wb.close()

        # Normalize and convert to chunks
        chunks: List[Chunk] = []
        stats = {
            "total_records": len(records),
            "by_machine": {},
            "by_area": {},
            "by_status": {},
            "by_jenis": {},
            "penyebab_filled": 0,
            "penyebab_empty": 0,
            "low_info_count": 0,
            "flagged_date_count": 0,
            "flagged_rows": [],
        }

        for record in records:
            # --- Normalization ---

            # Status normalization (case-insensitive + canonical mapping)
            raw_status = record.get("status", "").strip()
            raw_status_clean = raw_status.lower()
            if "program" in raw_status_clean:
                norm_status = "selesai_program"
            elif raw_status_clean in status_map:
                norm_status = status_map[raw_status_clean]
            elif "selesai" in raw_status_clean and "belum" not in raw_status_clean:
                norm_status = "selesai"
            elif "belum" in raw_status_clean:
                norm_status = "belum_selesai"
            else:
                norm_status = raw_status_clean
            record["status"] = norm_status
            record["status_normalized"] = norm_status

            # Jenis kerusakan normalization (case-insensitive + canonical mapping)
            raw_jenis = record.get("jenis_kerusakan", "").strip()
            raw_jenis_clean = raw_jenis.lower()
            if raw_jenis_clean in jenis_map:
                norm_jenis = jenis_map[raw_jenis_clean]
            elif "mech" in raw_jenis_clean or raw_jenis_clean.startswith("m"):
                norm_jenis = "mechanical"
            elif "elect" in raw_jenis_clean or raw_jenis_clean.startswith("e"):
                norm_jenis = "electrical"
            else:
                norm_jenis = raw_jenis_clean
            record["jenis_kerusakan"] = norm_jenis
            record["jenis_normalized"] = norm_jenis

            # Machine normalization (aliases)
            raw_machine = record.get("machine", "").strip()
            norm_machine = machine_aliases.get(raw_machine.lower(), raw_machine)
            record["machine"] = norm_machine

            # Typo corrections on text fields
            for text_field in ("permasalahan", "penyebab", "kegiatan_perbaikan"):
                val = record.get(text_field, "")
                if val:
                    corrected = val
                    for typo, fix in typo_corrections.items():
                        if typo.isalnum():
                            pattern = r"\b" + re.escape(typo) + r"\b"
                        else:
                            pattern = re.escape(typo)
                        corrected = re.sub(
                            pattern, fix, corrected, flags=re.IGNORECASE
                        )
                    record[text_field] = corrected

            # Merge tools
            tools_list = record.get("_tools_list", [])
            record["tools_merged"] = ", ".join(t for t in tools_list if t)
            record["tools"] = record["tools_merged"]

            # Date validation
            date_flagged = False
            raw_date = record.get("date", "").strip()
            if raw_date and raw_date != "tanggal tidak tercatat":
                try:
                    parsed = datetime.strptime(raw_date, "%Y-%m-%d")
                    yr = parsed.year
                    if yr < year_range["min"] or yr > year_range["max"]:
                        date_flagged = True
                    if parsed > datetime.now():
                        date_flagged = True
                except (ValueError, TypeError):
                    date_flagged = True
            else:
                raw_date = "tanggal tidak tercatat"
                record["date"] = raw_date
            record["date_flagged"] = date_flagged

            # Low-info detection
            penyebab = record.get("penyebab", "").strip()
            kegiatan = record.get("kegiatan_perbaikan", "").strip()
            is_low_info = (
                (not penyebab or penyebab == "-")
                and kegiatan.lower() in low_info_patterns
            )
            record["low_info"] = is_low_info

            # --- Stats collection ---
            machine = record.get("machine", "").strip()
            area = record.get("area", "").strip()
            if machine:
                stats["by_machine"][machine] = stats["by_machine"].get(machine, 0) + 1
            if area:
                stats["by_area"][area] = stats["by_area"].get(area, 0) + 1
            stats["by_status"][norm_status] = stats["by_status"].get(norm_status, 0) + 1
            if norm_jenis:
                stats["by_jenis"][norm_jenis] = stats["by_jenis"].get(norm_jenis, 0) + 1
            if penyebab and penyebab != "-":
                stats["penyebab_filled"] += 1
            else:
                stats["penyebab_empty"] += 1
            if is_low_info:
                stats["low_info_count"] += 1
            if date_flagged:
                stats["flagged_date_count"] += 1
                stats["flagged_rows"].append({
                    "row": record["_row_start"],
                    "date": raw_date,
                    "machine": machine,
                })

            # Skip date-flagged records from indexing
            if date_flagged:
                continue

            # --- Build chunk ---
            row_start = record["_row_start"]
            chunk_id = f"{source_doc.doc_id}:Notification:{row_start}"

            # Build the embedded text representation
            parts = []
            if machine:
                parts.append(f"Mesin: {machine}")
            if area:
                parts.append(f"Area: {area}")
            funcloc = record.get("funcloc", "").strip()
            if funcloc:
                parts.append(f"FuncLoc: {funcloc}")
            masalah = record.get("permasalahan", "").strip()
            if masalah:
                parts.append(f"Masalah: {masalah}")
            if penyebab and penyebab != "-":
                parts.append(f"Penyebab: {penyebab}")
            if kegiatan and kegiatan != "-":
                parts.append(f"Tindakan: {kegiatan}")
            if norm_jenis:
                parts.append(f"Jenis: {norm_jenis}")
            if norm_status:
                parts.append(f"Status: {norm_status}")
            rework = record.get("rework", "").strip()
            if rework and rework.lower() not in ("", "-", "tidak"):
                parts.append(f"Rework: {rework}")

            text_en = " | ".join(parts)

            # Build breadcrumb for citation
            breadcrumb = (
                f"{source_doc.doc_title} > {sheet_name} > "
                f"Baris {row_start}"
            )
            if raw_date and raw_date != "tanggal tidak tercatat":
                breadcrumb += f" ({raw_date})"
            else:
                breadcrumb += " (tanggal tidak tercatat)"

            # Codes: extract alarm/error codes from text fields using expanded regex (A5 & D3)
            all_text = f"{masalah} {penyebab} {kegiatan}"
            extracted_codes = set()
            # Standard [AEW] 2-4 digits (e.g. A012, W003, E024, E001)
            for m in re.finditer(r"\b([AEW]\d{2,4}(?:\.\d+)?)\b", all_text, re.IGNORECASE):
                extracted_codes.add(m.group(1).upper())

            # 7-segment display codes: E 1 .. E 6, Eb, EC, Ed, EL
            for m in re.finditer(r"\b(E\s?[1-6BbCcDdLl])\b", all_text):
                extracted_codes.add(re.sub(r"\s+", "", m.group(1).upper()))
            for m in re.finditer(r"(?:error|eror|kode|layar|display)\s+(e\s?[1-6bbcddll])\b", all_text, re.IGNORECASE):
                extracted_codes.add(re.sub(r"\s+", "", m.group(1).upper()))

            # 'no comm an'
            if re.search(r"\bno\s+comm\s+an\b", all_text, re.IGNORECASE):
                extracted_codes.add("NO COMM AN")

            # 'error 30'
            for m in re.finditer(r"\b(?:error|eror)\s?(\d{1,4})\b", all_text, re.IGNORECASE):
                extracted_codes.add(f"error {m.group(1)}")

            codes = sorted(extracted_codes)

            # Build metadata for raw field access and filtering
            metadata = {
                "date": raw_date,
                "shift": record.get("shift", ""),
                "area": area,
                "machine": machine,
                "funcloc": funcloc,
                "permasalahan": masalah,
                "penyebab": penyebab,
                "kegiatan_perbaikan": kegiatan,
                "jenis_kerusakan": norm_jenis,
                "status": norm_status,
                "rework": rework,
                "tools": record.get("tools_merged", ""),
                "spare_part_name": record.get("spare_part_name", ""),
                "spare_part_qty": record.get("spare_part_qty", ""),
                "low_info": is_low_info,
                "row_start": row_start,
                "row_end": record.get("_row_end", row_start),
            }

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=source_doc.doc_id,
                doc_title=source_doc.doc_title,
                doc_version=source_doc.doc_version,
                source_type="excel",
                page=row_start,  # Use row number as "page" for citation
                section="Notification",
                breadcrumb=breadcrumb,
                kind="history_record",
                codes=codes,
                text_en=text_en,
                metadata=metadata,
            )
            chunk.content_hash = chunk.compute_hash()
            chunks.append(chunk)

        # Store stats for reporting (accessed by pipeline)
        self._last_stats = stats

        return chunks

    @property
    def last_stats(self) -> Dict[str, Any]:
        """Return statistics from the last parse run."""
        return getattr(self, "_last_stats", {})


ParserRegistry.register("excel", ExcelParser)

"""Maintenance history aggregator.

Performs rule-based aggregation of maintenance records from Excel
WITHOUT using any LLM, adhering strictly to privacy and cautious phrasing:
- 'tercatat N kali'
- 'biasanya ditangani dengan'
- Status 'selesai' counted as successful actions
- 'belum_selesai' and rework tracked separately as unresolved
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from machine_assistant.retrieval.engine import RetrievalResult
    from machine_assistant.retrieval.entity_detector import DetectedEntities


def _format_date(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' or timestamp to readable Indonesian format."""
    if not date_str:
        return ""
    try:
        clean = date_str.split()[0]
        dt = datetime.strptime(clean, "%Y-%m-%d")
        months = [
            "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"
        ]
        return f"{dt.day} {months[dt.month]} {dt.year}"
    except Exception:
        return str(date_str)


class MaintenanceAggregator:
    """Aggregates retrieved maintenance chunks into a clear, cautious Indonesian report."""

    @classmethod
    def aggregate(
        cls,
        results: List["RetrievalResult"],
        entities: Optional["DetectedEntities"] = None,
    ) -> str:
        # Filter only excel results
        excel_results = [r for r in results if r.chunk.get("source_type") == "excel"]
        if not excel_results:
            return ""

        total_records = len(excel_results)

        # Collect metadata
        machines = Counter()
        areas = Counter()
        funclocs = Counter()
        dates: List[str] = []
        causes = Counter()
        actions_success = Counter()
        unresolved_notes: List[str] = []
        rework_count = 0

        for r in excel_results:
            meta = r.chunk.get("metadata", {})
            m = meta.get("machine", "").strip()
            if m:
                machines[m] += 1
            a = meta.get("area", "").strip()
            if a:
                areas[a] += 1
            fl = meta.get("funcloc", "").strip()
            if fl:
                funclocs[fl] += 1
            d = meta.get("date", "").strip()
            if d:
                dates.append(d)

            cause = meta.get("penyebab", "").strip()
            if cause and cause != "-":
                causes[cause.lower()] += 1

            status = meta.get("status", "").strip().lower()
            action = meta.get("kegiatan_perbaikan", "").strip()
            if status == "selesai":
                if action and action != "-":
                    actions_success[action] += 1
            elif status in ("belum_selesai", "selesai_program"):
                unresolved_notes.append(
                    f"Baris {meta.get('row_start')}: {meta.get('permasalahan', '')} "
                    f"({status})"
                )

            rework = meta.get("rework", "").strip().lower()
            if rework and rework not in ("tidak", "-", "no", ""):
                rework_count += 1

        # Date range calculation
        clean_dates = sorted([d.split()[0] for d in dates if d])
        if clean_dates:
            earliest = _format_date(clean_dates[0])
            latest = _format_date(clean_dates[-1])
            if earliest == latest:
                date_range_str = f"pada {earliest}"
            else:
                date_range_str = f"rentang {earliest} s/d {latest}"
        else:
            date_range_str = "tanggal tidak tercatat"

        # Machine label
        primary_machine = ""
        if entities and entities.machine:
            primary_machine = entities.machine
        elif len(machines) == 1:
            primary_machine = machines.most_common(1)[0][0]

        lines: List[str] = []
        if primary_machine:
            lines.append(f"Berdasarkan riwayat perbaikan pada mesin **{primary_machine}**:")
        else:
            lines.append("Berdasarkan riwayat perbaikan maintenance yang tercatat:")

        lines.append(f"• **Frekuensi kejadian**: Tercatat **{total_records} kali** ({date_range_str}).")

        # Areas & FuncLocs
        area_list = [f"{a} ({cnt}x)" if cnt > 1 else a for a, cnt in areas.most_common(3)]
        if area_list:
            lines.append(f"• **Lokasi / Area**: {', '.join(area_list)}")
        floc_list = [fl for fl, _ in funclocs.most_common(3) if fl]
        if floc_list:
            lines.append(f"• **FuncLoc terkait**: {', '.join(floc_list)}")

        # Penyebab terbanyak
        lines.append("")
        lines.append("• **Penyebab yang sering tercatat**:")
        if causes:
            for cause_text, cnt in causes.most_common(3):
                lines.append(f"  - {cause_text.capitalize()} ({cnt}x kejadian)")
        else:
            lines.append("  - Belum ada deskripsi penyebab spesifik yang terdokumentasi.")

        # Tindakan penanganan yang pernah berhasil (status selesai)
        lines.append("")
        lines.append("• **Tindakan penanganan yang pernah berhasil (status selesai)**:")
        if actions_success:
            lines.append("  Biasanya ditangani dengan langkah berikut:")
            for act_text, cnt in actions_success.most_common(3):
                count_str = f" ({cnt}x berhasil)" if cnt > 1 else ""
                lines.append(f"  - {act_text}{count_str}")
        else:
            lines.append("  - Belum ada tindakan berstatus 'selesai' yang tercatat untuk kendala ini.")

        # Status belum selesai / rework
        if unresolved_notes or rework_count > 0:
            lines.append("")
            lines.append("• **Catatan kendala belum tuntas / rework**:")
            if unresolved_notes:
                lines.append(
                    f"  - Terdapat {len(unresolved_notes)} catatan pekerjaan berstatus belum selesai "
                    "atau memerlukan tindak lanjut program."
                )
            if rework_count > 0:
                lines.append(f"  - Terdapat {rework_count} pekerjaan dengan catatan rework/perbaikan ulang.")

        # Contoh record riwayat (maks 3 contoh)
        sep = "- " * 34
        lines.append("")
        lines.append("• **Contoh Riwayat Tercatat** (maks. 3 entri):")
        for idx, r in enumerate(excel_results[:3], 1):
            meta = r.chunk.get("metadata", {})
            row      = meta.get("row_start", r.page)
            date_str = _format_date(meta.get("date", ""))
            masalah  = meta.get("permasalahan", "-")
            penyebab = meta.get("penyebab", "-")
            tindakan = meta.get("kegiatan_perbaikan", "-")
            status   = meta.get("status", "-")
            fl       = meta.get("funcloc", "")
            loc_str  = f"  |  FuncLoc: {fl}" if fl else ""

            lines.append(f"")
            lines.append(f"  [{idx}] Laporan Harian  —  {date_str}  (Baris {row}{loc_str})")
            lines.append(f"  {sep}")
            lines.append(f"      Masalah        : {masalah}")
            lines.append(f"      Penyebab       : {penyebab}")
            lines.append(f"      Tindakan       : {tindakan}")
            lines.append(f"      Status         : {status.upper()}")

        return "\n".join(lines)

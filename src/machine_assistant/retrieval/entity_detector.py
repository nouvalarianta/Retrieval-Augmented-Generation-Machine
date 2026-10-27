"""Entity detector for extracting machine, area, FuncLoc, and alarm codes from queries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from machine_assistant.config import get_repo_root, load_yaml


@dataclass
class DetectedEntities:
    machine: Optional[str] = None
    manual_machine_type: Optional[str] = None  # e.g. "CHECK", "BAG", "DUMP", "DIFF", "FBAL"
    area: Optional[str] = None
    funcloc: Optional[str] = None
    alarm_codes: List[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.machine or self.manual_machine_type or self.area or self.funcloc or self.alarm_codes)


# Mapping from normalized machine name to manual machine_type code
_MACHINE_TO_MANUAL_TYPE: Dict[str, str] = {
    "Check Weigher": "CHECK",
    "CHECK": "CHECK",
    "Automatic Weigher": "BAG",
    "BAG": "BAG",
    "Dump Scale": "DUMP",
    "DUMP": "DUMP",
    "Differential Dosing Scale": "DIFF",
    "DIFF": "DIFF",
    "Microdozer": "DIFF",
    "Flowmatic": "DIFF",
    "Flowbalancer": "FBAL",
    "FBAL": "FBAL",
}

# Reverse mapping: manual code to common names
_MANUAL_TYPE_TO_MACHINES: Dict[str, List[str]] = {
    "CHECK": ["Check Weigher", "Checkweigher"],
    "BAG": ["Automatic Weigher", "Bagging Scale", "Bagger"],
    "DUMP": ["Dump Scale"],
    "DIFF": ["Differential Dosing Scale", "Microdozer", "Flowmatic"],
    "FBAL": ["Flowbalancer", "Automatic Volume Controller"],
}

_KNOWN_AREAS = [
    "Worksop Automation", "Workshop Automation",
    "Packing 25 Kg", "Pellet Silo Baru", "Pellet Silo Lama",
    "Wheat Silo Lama", "Wheat Silo Baru",
    "Mill A", "Mill B", "Mill C", "Mill D", "Mill E", "Mill F", "Mill G", "Mill H",
    "Jetty", "BPP", "Bulk", "Boiler", "CP",
]


class EntityDetector:
    """Detect machine, area, FuncLoc, and alarm/error codes in query text."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = get_repo_root()
        self.repo_root = repo_root
        self._load_config()

    def _load_config(self) -> None:
        profile_path = self.repo_root / "config" / "profiles" / "laporan_harian.yaml"
        profile_data = load_yaml(profile_path)
        aliases = profile_data.get("machine_aliases", {})

        # Build alias lookup: lowercase string -> canonical machine name
        self.alias_map: Dict[str, str] = {}
        for alias, canonical in aliases.items():
            self.alias_map[alias.lower()] = canonical

        # Add explicit multi-word and unambiguous manual-specific aliases (D1 & D2)
        # Standalone generic 1-word terms ("check", "bag", "dump", "diff") are explicitly excluded.
        self.alias_map.update({
            # Checkweigher
            "meag check": "Check Weigher",
            "meag-check": "Check Weigher",
            "tipe check": "Check Weigher",
            "check weigher": "Check Weigher",
            "checkweigher": "Check Weigher",

            # Bagging scale / Automatic weigher
            "meag bag": "Automatic Weigher",
            "meag-bag": "Automatic Weigher",
            "bagging scale": "Automatic Weigher",
            "auto weigher": "Automatic Weigher",
            "automatic weigher": "Automatic Weigher",
            "bagger": "Automatic Weigher",

            # Dump scale
            "meag dump": "Dump Scale",
            "meag-dump": "Dump Scale",
            "dump scale": "Dump Scale",

            # Differential dosing scale
            "meag diff": "Differential Dosing Scale",
            "meag-diff": "Differential Dosing Scale",
            "differential dosing scale": "Differential Dosing Scale",
            "dosing scale": "Differential Dosing Scale",
            "microdozer": "Microdozer",
            "flowmatic": "Flowmatic",

            # Flowbalancer
            "flowbalancer": "Flowbalancer",
            "fbal": "Flowbalancer",
            "automatic volume controller": "Flowbalancer",

            # Control system (MEAG controller itself)
            "control system": "Control System",
        })

        # Ensure single generic words are never present in alias_map
        for generic in ("check", "bag", "dump", "diff"):
            self.alias_map.pop(generic, None)

        # Sort alias keys by length descending to match longest phrases first
        self._sorted_aliases = sorted(self.alias_map.keys(), key=lambda x: len(x), reverse=True)

    def detect(self, query: str) -> DetectedEntities:
        q_lower = query.lower()
        detected_machine: Optional[str] = None
        manual_type: Optional[str] = None

        # 1. Detect Machine (longest phrase match first with word boundaries)
        for alias in self._sorted_aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, q_lower):
                detected_machine = self.alias_map[alias]
                manual_type = _MACHINE_TO_MANUAL_TYPE.get(detected_machine)
                break

        # 2. Detect Area
        detected_area: Optional[str] = None
        for area in _KNOWN_AREAS:
            if re.search(r"\b" + re.escape(area.lower()) + r"\b", q_lower):
                detected_area = area
                break

        # 3. Detect FuncLoc (pattern like 1253-FLC0-BG202 or 1253-CP00-CP11)
        floc_match = re.search(r"\b\d{4}-[A-Z0-9]+-[A-Z0-9]+\b", query, re.IGNORECASE)
        detected_funcloc = floc_match.group(0).upper() if floc_match else None

        # 4. Detect Alarm / Error codes (D3)
        alarm_codes: List[str] = []

        # 4a. Expanded alarm codes: [AEW] + 2-4 digits (e.g. A012, W003, E024, E001)
        for code in re.findall(r"\b([AEW]\d{2,4}(?:\.\d+)?)\b", query, re.IGNORECASE):
            norm_code = code.upper()
            if norm_code not in alarm_codes:
                alarm_codes.append(norm_code)

        # 4b. Pattern for 'error 30' or 'eror 30'
        for err in re.findall(r"\b(?:error|eror)\s?(\d{1,4})\b", query, re.IGNORECASE):
            code = f"error {err}"
            if code not in alarm_codes:
                alarm_codes.append(code)

        # 4c. Pattern for 'no comm an' (case-insensitive)
        if re.search(r"\bno\s+comm\s+an\b", query, re.IGNORECASE):
            if "no comm an" not in alarm_codes:
                alarm_codes.append("no comm an")

        # 4d. 7-segment error code (E 1..E 6, Eb, EC, Ed, EL)
        # Context guard against false positives like "vitamin E 1", "tipe e 1"
        has_error_context = bool(
            re.search(
                r"\b(?:error|eror|display|layar|kode|code|indikator|indicator|h1|7-seg|segmen|segment|tampil|muncul|pesan|message|alarm|rusak|fault)\b",
                query,
                re.IGNORECASE,
            )
        )

        if has_error_context:
            for match in re.finditer(r"\b([Ee]\s?[1-6bBcCdDlL])\b", query):
                raw = match.group(1)
                norm = re.sub(r"\s+", " ", raw).upper().strip()
                if norm not in alarm_codes:
                    alarm_codes.append(norm)
        else:
            # Without error context, require uppercase E and guard against common non-error phrases
            for match in re.finditer(r"\b(E\s?[1-6bBcCdDlL])\b", query):
                raw = match.group(1)
                start_pos = match.start()
                preceding = query[:start_pos].strip().split()
                last_word = preceding[-1].lower() if preceding else ""
                if last_word in {"vitamin", "vit", "lantai", "tipe", "kelas", "grade", "seksi", "group"}:
                    continue
                norm = re.sub(r"\s+", " ", raw).upper().strip()
                if norm not in alarm_codes:
                    alarm_codes.append(norm)

        return DetectedEntities(
            machine=detected_machine,
            manual_machine_type=manual_type,
            area=detected_area,
            funcloc=detected_funcloc,
            alarm_codes=alarm_codes,
        )

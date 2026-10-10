"""Strict validation for the source manifest.

Only rows explicitly marked approved and using a supported extraction mode
can be fetched. Candidate and deferred rows are metadata-only.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_CONDITIONS = {"asthma", "type_2_diabetes"}
ALLOWED_STAGES = {
    "understanding_condition",
    "appointment_preparation",
    "ongoing_care_information",
}
ALLOWED_STATUSES = {"approved", "candidate", "deferred_permission"}
ALLOWED_EXTRACTION_MODES = {
    "none",
    "medlineplus_topic_summary",
    "medlineplus_medical_test",
}
REQUIRED_COLUMNS = [
    "source_id", "condition", "stage_scope", "title", "publisher", "url",
    "source_type", "checked_date", "status", "extraction_mode", "reuse_basis", "notes",
]
SOURCE_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,63}$")


class ManifestValidationError(ValueError):
    """Raised when one or more manifest rows violate the project policy."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Source manifest validation failed:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    condition: str
    stage_scope: tuple[str, ...]
    title: str
    publisher: str
    url: str
    source_type: str
    checked_date: date
    status: str
    extraction_mode: str
    reuse_basis: str
    notes: str

    @property
    def approved_for_ingestion(self) -> bool:
        return self.status == "approved"


def _is_allowed_host(host: str) -> bool:
    host = host.casefold().rstrip(".")
    return (
        host == "medlineplus.gov"
        or host.endswith(".medlineplus.gov")
        or host == "nhs.uk"
        or host.endswith(".nhs.uk")
        or host == "who.int"
        or host.endswith(".who.int")
    )


def _mode_matches_host(mode: str, host: str) -> bool:
    host = host.casefold().rstrip(".")
    if mode in {"medlineplus_topic_summary", "medlineplus_medical_test"}:
        return host == "medlineplus.gov" or host.endswith(".medlineplus.gov")
    return mode == "none"


def load_source_manifest(path: str | Path) -> list[SourceRecord]:
    """Load and validate a CSV manifest; raise with every validation error found."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    errors: list[str] = []
    records: list[SourceRecord] = []
    source_ids: set[str] = set()
    urls: set[str] = set()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise ManifestValidationError(
                [f"Missing required columns: {', '.join(missing_columns)}"]
            )

        for line_number, raw_row in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw_row.items() if key}
            row_error_count = len(errors)

            source_id = row.get("source_id", "")
            if not SOURCE_ID_RE.fullmatch(source_id):
                errors.append(f"Line {line_number}: invalid source_id {source_id!r}.")
            if source_id in source_ids:
                errors.append(f"Line {line_number}: duplicate source_id {source_id!r}.")
            source_ids.add(source_id)

            for field in ("condition", "title", "publisher", "url", "source_type",
                          "checked_date", "status", "extraction_mode", "reuse_basis"):
                if not row.get(field):
                    errors.append(f"Line {line_number}: {field} must not be empty.")

            condition = row.get("condition", "")
            if condition not in ALLOWED_CONDITIONS:
                errors.append(f"Line {line_number}: unsupported condition {condition!r}.")

            stages = tuple(dict.fromkeys(
                part.strip() for part in row.get("stage_scope", "").split(";") if part.strip()
            ))
            if not stages:
                errors.append(f"Line {line_number}: stage_scope must contain at least one stage.")
            unknown_stages = sorted(set(stages) - ALLOWED_STAGES)
            if unknown_stages:
                errors.append(
                    f"Line {line_number}: unknown stage(s): {', '.join(unknown_stages)}."
                )

            status = row.get("status", "")
            if status not in ALLOWED_STATUSES:
                errors.append(f"Line {line_number}: invalid status {status!r}.")

            mode = row.get("extraction_mode", "")
            if mode not in ALLOWED_EXTRACTION_MODES:
                errors.append(f"Line {line_number}: invalid extraction_mode {mode!r}.")

            try:
                checked_date = date.fromisoformat(row.get("checked_date", ""))
            except ValueError:
                checked_date = date.min
                errors.append(f"Line {line_number}: checked_date must use YYYY-MM-DD.")

            parsed = urlparse(row.get("url", ""))
            host = (parsed.hostname or "").casefold()
            if parsed.scheme != "https" or not host or not _is_allowed_host(host):
                errors.append(
                    f"Line {line_number}: URL must be HTTPS on an approved source domain."
                )
            if parsed.username or parsed.password:
                errors.append(f"Line {line_number}: credentials are not permitted in source URLs.")
            if parsed.query or parsed.fragment:
                errors.append(f"Line {line_number}: source URL must not contain query or fragment.")
            if row.get("url", "") in urls:
                errors.append(f"Line {line_number}: duplicate URL {row.get('url')!r}.")
            urls.add(row.get("url", ""))

            if status == "approved":
                if mode == "none":
                    errors.append(
                        f"Line {line_number}: approved source must have a supported extraction mode."
                    )
                if not _mode_matches_host(mode, host):
                    errors.append(
                        f"Line {line_number}: extraction mode {mode!r} does not match host {host!r}."
                    )
                if not row.get("reuse_basis"):
                    errors.append(f"Line {line_number}: approved source requires a reuse_basis.")

            if len(errors) == row_error_count:
                records.append(SourceRecord(
                    source_id=source_id,
                    condition=condition,
                    stage_scope=stages,
                    title=row["title"],
                    publisher=row["publisher"],
                    url=row["url"],
                    source_type=row["source_type"],
                    checked_date=checked_date,
                    status=status,
                    extraction_mode=mode,
                    reuse_basis=row["reuse_basis"],
                    notes=row.get("notes", ""),
                ))

    if errors:
        raise ManifestValidationError(errors)
    return records

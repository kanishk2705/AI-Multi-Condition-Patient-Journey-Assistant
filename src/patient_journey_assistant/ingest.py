"""Explicitly opt-in, robots-aware ingestion of approved manifest sources."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from .chunking import chunk_source_text, sha256_text
from .manifest import ManifestValidationError, SourceRecord, load_source_manifest
from .text_processing import extract_source_text

USER_AGENT = (
    "PatientJourneyAssistantPortfolio/0.1 "
    "(educational research prototype; source attribution retained)"
)
DEFAULT_MANIFEST = Path("data/source_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("data/local/processed")
REQUEST_TIMEOUT_SECONDS = 20
MAX_HTML_BYTES = 4_000_000


class IngestionError(RuntimeError):
    """Raised when a source fails access-policy or extraction checks."""


def _host_family(host: str) -> str:
    host = host.casefold().rstrip(".")
    for base in ("medlineplus.gov", "nhs.uk", "who.int"):
        if host == base or host.endswith("." + base):
            return base
    return host


def _check_robots(url: str, session: requests.Session) -> None:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = session.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise IngestionError(f"Could not check robots.txt for {parsed.netloc}: {exc}") from exc

    if response.status_code in {404, 410}:
        return
    if response.status_code != 200:
        raise IngestionError(
            f"Cannot establish robots policy for {parsed.netloc} "
            f"(HTTP {response.status_code}); refusing to fetch."
        )

    robot_parser = RobotFileParser()
    robot_parser.set_url(robots_url)
    robot_parser.parse(response.text.splitlines())
    if not robot_parser.can_fetch(USER_AGENT, url):
        raise IngestionError(f"robots.txt disallows this page: {url}")


def fetch_and_extract(source: SourceRecord, session: requests.Session | None = None) -> tuple[str, str]:
    """Fetch one approved source, enforce host/robots checks, and extract its allowed text."""
    if not source.approved_for_ingestion:
        raise IngestionError(f"{source.source_id} is not approved for ingestion.")
    if source.extraction_mode == "none":
        raise IngestionError(f"{source.source_id} has no approved extraction mode.")

    client = session or requests.Session()
    _check_robots(source.url, client)
    try:
        response = client.get(
            source.url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise IngestionError(f"Fetch failed for {source.source_id}: {exc}") from exc

    if response.status_code != 200:
        raise IngestionError(f"{source.source_id}: HTTP {response.status_code}.")
    final_url = urlparse(response.url or source.url)
    original_url = urlparse(source.url)
    if final_url.scheme != "https" or _host_family(final_url.hostname or "") != _host_family(
        original_url.hostname or ""
    ):
        raise IngestionError(f"{source.source_id}: redirect left the approved source domain.")
    content_type = response.headers.get("Content-Type", "").casefold()
    if "text/html" not in content_type:
        raise IngestionError(
            f"{source.source_id}: expected HTML but received {content_type or 'unknown content type'}."
        )
    if len(response.content) > MAX_HTML_BYTES:
        raise IngestionError(f"{source.source_id}: page exceeds the configured size limit.")

    extracted = extract_source_text(response.text, source.extraction_mode)
    return extracted, response.url or source.url


def ingest_approved_sources(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    fetch: bool = False,
    session: requests.Session | None = None,
    delay_seconds: float = 1.0,
) -> dict:
    records = load_source_manifest(manifest_path)
    approved = [record for record in records if record.approved_for_ingestion]
    skipped = [record for record in records if not record.approved_for_ingestion]

    report: dict = {
        "manifest": str(manifest_path),
        "manifest_rows": len(records),
        "approved_rows": len(approved),
        "skipped_rows": [{"source_id": r.source_id, "status": r.status} for r in skipped],
        "fetched_sources": [],
        "failed_sources": [],
        "chunks_written": 0,
        "fetch_enabled": fetch,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    if not fetch:
        return report

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    all_chunks: list[dict] = []
    last_host = None

    for source in approved:
        host = urlparse(source.url).netloc.casefold()
        if last_host == host and delay_seconds > 0:
            time.sleep(delay_seconds)
        last_host = host
        try:
            text, final_url = fetch_and_extract(source, client)
            fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            chunks = chunk_source_text(source, text, fetched_at=fetched_at)
            if not chunks:
                raise IngestionError(f"{source.source_id}: extraction produced no chunks.")
            for chunk in chunks:
                chunk["source_url"] = final_url
            all_chunks.extend(chunks)
            report["fetched_sources"].append({
                "source_id": source.source_id,
                "final_url": final_url,
                "text_sha256": sha256_text(text),
                "character_count": len(text),
                "chunk_count": len(chunks),
                "fetched_at": fetched_at,
            })
        except (IngestionError, requests.RequestException, ValueError) as exc:
            report["failed_sources"].append({
                "source_id": source.source_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })

    output.mkdir(parents=True, exist_ok=True)
    chunks_path = output / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as stream:
        for chunk in all_chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")

    report["chunks_written"] = len(all_chunks)
    report["chunks_path"] = str(chunks_path)
    report_path = output / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fetch-approved",
        action="store_true",
        help="Fetch approved sources after robots.txt and domain checks. Network access is off by default.",
    )
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()

    try:
        report = ingest_approved_sources(
            args.manifest,
            args.output_dir,
            fetch=args.fetch_approved,
            delay_seconds=max(0.0, args.delay_seconds),
        )
    except (FileNotFoundError, ManifestValidationError) as exc:
        print(str(exc))
        return 2

    print(f"Manifest rows: {report['manifest_rows']}")
    print(f"Approved for ingestion: {report['approved_rows']}")
    print(f"Skipped by policy: {len(report['skipped_rows'])}")
    if not args.fetch_approved:
        print("Dry run only. No network requests made. Use --fetch-approved to ingest approved sources.")
        return 0

    print(f"Sources fetched: {len(report['fetched_sources'])}")
    print(f"Source failures: {len(report['failed_sources'])}")
    print(f"Chunks written: {report['chunks_written']}")
    print(f"Report: {args.output_dir / 'ingestion_report.json'}")
    if report["failed_sources"]:
        for failure in report["failed_sources"]:
            print(f"- {failure['source_id']}: {failure['message']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Source-page text extraction with restrictive, auditable extraction modes."""
from __future__ import annotations

import re
from bs4 import BeautifulSoup, Tag


class TextExtractionError(ValueError):
    """Raised when a source cannot be extracted using its approved mode."""


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation while preserving paragraph boundaries."""
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n\n".join(lines)


def _blocks_from_tags(container: Tag) -> list[str]:
    """Extract article blocks and stop before bibliography sections."""
    blocks: list[str] = []
    selectors = ["h1", "h2", "h3", "h4", "p", "li", "dd"]
    heading_tags = {"h1", "h2", "h3", "h4"}
    stop_headings = {
        "references",
        "references and resources",
        "bibliography",
    }

    for node in container.find_all(selectors):
        if node.name in heading_tags:
            heading_text = node.get_text(" ", strip=True).casefold().rstrip(":")
            if heading_text in stop_headings:
                break

        # Avoid duplicate text from nested list items and paragraphs.
        if node.name == "li" and node.find(["p", "ul", "ol", "li"]):
            continue

        text = node.get_text(" ", strip=True)
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)

    return blocks

def extract_medlineplus_topic_summary(html: str | bytes) -> str:
    """Extract only the health-topic page's Summary section.

    MedlinePlus states that summaries on health-topic pages are public domain.
    This deliberately avoids separately copyrighted encyclopedia, drug, image,
    video, and non-summary sections.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = None
    for candidate in soup.find_all(["h2", "h3", "h4"]):
        if candidate.get_text(" ", strip=True).casefold() == "summary":
            heading = candidate
            break
    if heading is None:
        raise TextExtractionError("MedlinePlus page has no identifiable Summary heading.")

    current_level = int(heading.name[1])
    blocks: list[str] = []
    for node in heading.find_all_next(["h1", "h2", "h3", "h4", "p", "li", "dd"]):
        if node.name in {"h1", "h2", "h3", "h4"} and int(node.name[1]) <= current_level:
            break
        if node.name == "li" and node.find(["p", "ul", "ol", "li"]):
            continue
        text = node.get_text(" ", strip=True)
        if text and text.casefold() != "summary" and (not blocks or blocks[-1] != text):
            blocks.append(text)

    result = normalize_text("\n".join(blocks))
    if len(result) < 120:
        raise TextExtractionError("Extracted MedlinePlus Summary is unexpectedly short.")
    return result


def extract_medlineplus_medical_test(html: str | bytes) -> str:
    """Extract main text of a MedlinePlus medical-test page, excluding site chrome/media."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("main") or soup.find("article") or soup.body
    if not isinstance(container, Tag):
        raise TextExtractionError("No main page content was found.")

    for element in container.find_all([
        "script", "style", "nav", "aside", "footer", "form", "button",
        "svg", "img", "figure", "picture", "iframe", "noscript",
    ]):
        element.decompose()

    result = normalize_text("\n".join(_blocks_from_tags(container)))
    if len(result) < 200:
        raise TextExtractionError("Extracted medical-test text is unexpectedly short.")
    return result


def extract_source_text(html: str | bytes, extraction_mode: str) -> str:
    if extraction_mode == "medlineplus_topic_summary":
        return extract_medlineplus_topic_summary(html)
    if extraction_mode == "medlineplus_medical_test":
        return extract_medlineplus_medical_test(html)
    raise TextExtractionError(f"Unsupported extraction mode: {extraction_mode!r}")

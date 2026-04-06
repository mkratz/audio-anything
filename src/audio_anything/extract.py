"""PDF text extraction via PyMuPDF4LLM, with image and table extraction."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pymupdf4llm

log = logging.getLogger(__name__)


@dataclass
class PageImage:
    page_number: int
    image_bytes: bytes
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class PageChunk:
    page_number: int
    text: str
    images: list[PageImage] = field(default_factory=list)
    has_tables: bool = False


def _extract_images(pdf_path: Path) -> dict[int, list[PageImage]]:
    """Extract images from each page, keyed by 1-based page number."""
    images_by_page: dict[int, list[PageImage]] = {}
    doc = pymupdf.open(str(pdf_path))

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_images: list[PageImage] = []

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                img_data = doc.extract_image(xref)
                if not img_data or not img_data.get("image"):
                    continue

                # Skip tiny images (icons, bullets, decorations)
                rect = rects[0]
                width = rect.x1 - rect.x0
                height = rect.y1 - rect.y0
                if width < 50 or height < 50:
                    log.debug("Skipping tiny image on page %d (%dx%d)", page_num, width, height)
                    continue

                # Skip low-information images (backgrounds, watermarks, solid fills)
                pixel_count = width * height
                byte_count = len(img_data["image"])
                if pixel_count > 0 and byte_count / pixel_count < 0.1:
                    log.debug("Skipping low-info image on page %d (%d bytes, %.0fx%.0f)", page_num, byte_count, width, height)
                    continue

                page_images.append(PageImage(
                    page_number=page_num,
                    image_bytes=img_data["image"],
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                ))
            except Exception:
                log.debug("Could not extract image xref %d on page %d", xref, page_num, exc_info=True)

        if page_images:
            images_by_page[page_num] = page_images
            log.debug("Page %d: extracted %d images", page_num, len(page_images))

    doc.close()
    return images_by_page


def _format_table_for_narration(cells: list[list[str | None]]) -> str:
    """Format extracted table cells as a tagged block for LLM narrativization."""
    rows: list[list[str]] = []
    for row in cells:
        clean = [(c.replace("\n", " ").strip() if c else "") for c in row]
        if any(clean):
            rows.append(clean)

    if len(rows) < 2:
        return ""

    max_cell_len = max(len(c) for row in rows for c in row)
    lines = ["[TABLE]"]

    header = rows[0]
    if max_cell_len > 100:
        # Expanded format for paragraph-length cells
        for i, row in enumerate(rows[1:], 1):
            for j, cell in enumerate(row):
                col_name = header[j] if j < len(header) and header[j] else f"Column {j + 1}"
                if cell:
                    lines.append(f"  {col_name}: {cell}")
            lines.append("")
    else:
        # Compact format for short cells
        lines.append("Header: " + " | ".join(header))
        for row in rows[1:]:
            lines.append("Row: " + " | ".join(row))

    lines.append("[/TABLE]")
    return "\n".join(lines)


def _extract_tables_raw(pdf_path: str) -> dict[int, list[list[list[str | None]]]]:
    """Extract raw table cells from each page. Runs in a subprocess to avoid
    pymupdf4llm's side effect of corrupting find_tables() text extraction."""
    import json
    import subprocess
    import sys

    script = '''
import json, sys, pymupdf
doc = pymupdf.open(sys.argv[1])
result = {}
for page_idx in range(len(doc)):
    page = doc[page_idx]
    page_num = page_idx + 1
    try:
        finder = page.find_tables()
    except Exception:
        continue
    if not finder.tables:
        continue
    page_tables = []
    for tab in finder.tables:
        try:
            cells = tab.extract()
            if len(cells) < 2:
                continue
            non_empty = sum(1 for row in cells for c in row if c and c.strip())
            if non_empty < 4:
                continue
            page_tables.append(cells)
        except Exception:
            pass
    if page_tables:
        result[str(page_num)] = page_tables
doc.close()
json.dump(result, sys.stdout)
'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(pdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            log.warning("Table extraction subprocess failed: %s", proc.stderr[:200])
            return {}
        raw: dict[str, list] = json.loads(proc.stdout)
        return {int(k): v for k, v in raw.items()}
    except Exception:
        log.warning("Table extraction subprocess error", exc_info=True)
        return {}


def _extract_tables(pdf_path: Path) -> dict[int, list[str]]:
    """Extract tables from each page, formatted as [TABLE] blocks."""
    raw_tables = _extract_tables_raw(str(pdf_path))
    if not raw_tables:
        return {}

    tables_by_page: dict[int, list[str]] = {}
    for page_num, page_cell_lists in raw_tables.items():
        page_tables: list[str] = []
        for cells in page_cell_lists:
            formatted = _format_table_for_narration(cells)
            if formatted:
                page_tables.append(formatted)
        if page_tables:
            tables_by_page[page_num] = page_tables
            log.debug("Page %d: extracted %d tables", page_num, len(page_tables))

    return tables_by_page


_TOC_PATTERN = re.compile(r"\*\*\d+\*\*")


def _is_toc_page(text: str) -> bool:
    """Detect table-of-contents pages by looking for repeated bold page numbers."""
    if "CONTENTS" in text.upper()[:200]:
        return True
    # TOC pages have many bold page-number references like **3**, **45**
    matches = _TOC_PATTERN.findall(text)
    return len(matches) >= 5


def extract_pages(pdf_path: Path, extract_images: bool = False) -> list[PageChunk]:
    """Extract markdown text (and optionally images/tables) from each page of a PDF."""
    log.info("Extracting text from %s", pdf_path)
    try:
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    except Exception:
        log.exception("Failed to extract PDF")
        raise

    images_by_page: dict[int, list[PageImage]] = {}
    if extract_images:
        log.info("Extracting images from %s", pdf_path)
        images_by_page = _extract_images(pdf_path)

    # Extract structured tables
    tables_by_page = _extract_tables(pdf_path)
    table_count = sum(len(v) for v in tables_by_page.values())
    if table_count:
        log.info("Detected %d tables across %d pages", table_count, len(tables_by_page))

    chunks = []
    for page in pages:
        page_num = page.get("metadata", {}).get("page", len(chunks) + 1)
        text = page.get("text", "")
        if text.strip():
            if _is_toc_page(text):
                log.info("Skipping TOC page %d", page_num)
                continue

            # Inject structured table blocks
            page_tables = tables_by_page.get(page_num, [])
            if page_tables:
                text = text.rstrip() + "\n\n" + "\n\n".join(page_tables)

            chunks.append(PageChunk(
                page_number=page_num,
                text=text,
                images=images_by_page.get(page_num, []),
                has_tables=bool(page_tables),
            ))

    img_count = sum(len(c.images) for c in chunks)
    log.info(
        "Extracted %d non-empty pages from %d total (%d images, %d tables)",
        len(chunks), len(pages), img_count, table_count,
    )
    return chunks

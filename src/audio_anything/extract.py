"""PDF text extraction via PyMuPDF4LLM, with image extraction."""

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


_TOC_PATTERN = re.compile(r"\*\*\d+\*\*")


def _is_toc_page(text: str) -> bool:
    """Detect table-of-contents pages by looking for repeated bold page numbers."""
    if "CONTENTS" in text.upper()[:200]:
        return True
    # TOC pages have many bold page-number references like **3**, **45**
    matches = _TOC_PATTERN.findall(text)
    return len(matches) >= 5


def extract_pages(pdf_path: Path, extract_images: bool = False) -> list[PageChunk]:
    """Extract markdown text (and optionally images) from each page of a PDF."""
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

    chunks = []
    for page in pages:
        page_num = page.get("metadata", {}).get("page", len(chunks) + 1)
        text = page.get("text", "")
        if text.strip():
            if _is_toc_page(text):
                log.info("Skipping TOC page %d", page_num)
                continue
            chunks.append(PageChunk(
                page_number=page_num,
                text=text,
                images=images_by_page.get(page_num, []),
            ))

    img_count = sum(len(c.images) for c in chunks)
    log.info("Extracted %d non-empty pages from %d total (%d images)", len(chunks), len(pages), img_count)
    return chunks

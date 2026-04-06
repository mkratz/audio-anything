"""Transcript cleaning: mechanical pre-processing + LLM semantic cleaning."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import Config
from .extract import PageChunk

log = logging.getLogger(__name__)

# --- Mechanical pre-processing (reliable, no LLM needed) ---

# Markdown heading → structural cue
_HEADING_RE = re.compile(r"^#{1,3}\s+\**(.+?)\**\s*$", re.MULTILINE)
# Bold / italic / code
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"[_*](.+?)[_*]")
_CODE_RE = re.compile(r"`(.+?)`")
# Links
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]+\)")
# Standalone URLs
_URL_RE = re.compile(r"https?://\S+")
# Image placeholder from pymupdf4llm
_IMG_PLACEHOLDER_RE = re.compile(r"\*\*==>.*?<==\*\*")
# Footer pattern: page number followed by title, or title followed by page number
# Handles mixed case like "58 Biosecurity Victory" and ALL-CAPS
_FOOTER_RE = re.compile(
    r"^\s*(?:\d{1,3}\s+[A-Z][A-Za-z\s\-]+|[A-Z][A-Za-z\s\-]+\s+\d{1,3})\s*$",
    re.MULTILINE,
)
# Generic footer: a line that is ONLY a number (page number)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
# Figure/source attribution lines
_FIGURE_LABEL_RE = re.compile(
    r"^\s*\**(FIGURE \d+|Figure \d+|Source:)\**.*$", re.MULTILINE
)
# Citation numbers like [1], [22]
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
# Source attribution lines (e.g. "Toxin," ScienceDirect Topics, accessed...)
_SOURCE_LINE_RE = re.compile(r"^.*accessed\s+\w+\s+\d{1,2},\s+\d{4}.*$", re.MULTILINE)
# Bibliographic reference lines (e.g. '"DNA Sequencing," Talking Glossary..., May 7, 2025,')
_BIBLIO_RE = re.compile(
    r'^"[^"]+,"\s+.*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4},?\s*$',
    re.MULTILINE,
)
# Numbered endnote lines (e.g. "1. Tony Blair and William Hague, ...")
_ENDNOTE_RE = re.compile(
    r"^\s*\d{1,3}[\.\)]\s+\u200b?[A-Z].*(?:(?:19|20)\d{2}|doi\.org|https?://).*$",
    re.MULTILINE,
)
# Markdown table artifacts like [|]
_TABLE_ARTIFACT_RE = re.compile(r"\[\|]")
# Markdown table rows: 3+ consecutive lines each containing 2+ pipe characters.
# Only stripped on pages where structured [TABLE] blocks are available.
_MD_TABLE_RE = re.compile(
    r"(?:^[^\n]*\|[^\n]*\|[^\n]*$\n?){3,}",
    re.MULTILINE,
)
# Markdown table separator rows like |---|---|---|
_MD_TABLE_SEP_RE = re.compile(r"^\|?[\s\-:]+\|[\s\-:|]+\|?\s*$", re.MULTILINE)
# Repeating header lines (all-caps or title-case short lines at page tops)
_HEADER_LINE_RE = re.compile(
    r"^\s*(?:A PUBLICATION OF\b.*|[A-Z][A-Z\s\-&]{4,50})\s*$",
    re.MULTILINE,
)


_SIDEBAR_HEADINGS = {"definitions", "glossary", "key terms", "box", "sidebar", "notes"}

# Headings that mark the start of back-matter to be entirely removed
_BACKMATTER_HEADINGS = re.compile(
    r"^(?:Chapter|Section):\s*(?:references|endnotes|bibliography|works cited|notes and references)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Marker used to strip sidebar blocks (heading + body) in _preprocess
_SIDEBAR_MARKER = "[@@SIDEBAR@@]"


def _classify_heading(text: str) -> str:
    """Decide if a heading is a chapter or section based on content."""
    stripped = text.strip()
    # Sidebar / definition-box headings — mark for removal with content
    if stripped.lower() in _SIDEBAR_HEADINGS:
        return _SIDEBAR_MARKER
    # Numbered chapters like "1. Worst-Case Scenarios"
    if re.match(r"^\d+[\.\)]\s", stripped):
        return f"Chapter: {stripped}"
    # Named major sections
    if stripped.lower() in ("preface", "introduction", "conclusion", "appendix", "acknowledgments"):
        return f"Chapter: {stripped}"
    return f"Section: {stripped}"


def _strip_sidebar_blocks(text: str) -> str:
    """Remove sidebar marker and everything until the next structural cue or end."""
    result_lines: list[str] = []
    in_sidebar = False
    for line in text.split("\n"):
        if _SIDEBAR_MARKER in line:
            in_sidebar = True
            continue
        if in_sidebar:
            # End sidebar at next structural cue or substantial body paragraph
            if line.startswith(("Chapter:", "Section:", "Image description:", "[TABLE]")):
                in_sidebar = False
                result_lines.append(line)
            # Skip sidebar content
            continue
        result_lines.append(line)
    return "\n".join(result_lines)


def _preprocess(text: str) -> str:
    """Mechanical cleaning: strip markdown, footers, URLs, etc."""
    # Convert headings to structural cues
    def heading_replace(m):
        return _classify_heading(m.group(1))
    text = _HEADING_RE.sub(heading_replace, text)

    # Remove sidebar blocks (heading marker + all content until next structural cue)
    text = _strip_sidebar_blocks(text)

    # Remove image placeholders (we handle images separately)
    text = _IMG_PLACEHOLDER_RE.sub("", text)

    # Remove figure labels / source attributions
    text = _FIGURE_LABEL_RE.sub("", text)

    # Strip markdown formatting
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)

    # Remove table artifacts
    text = _TABLE_ARTIFACT_RE.sub("", text)

    # Strip markdown table rows when structured [TABLE] blocks are present
    if "[TABLE]" in text:
        text = _MD_TABLE_RE.sub("", text)
        text = _MD_TABLE_SEP_RE.sub("", text)

    # Remove orphan markdown asterisks (not part of bold/italic pairs)
    text = re.sub(r"(?<!\*)\*(?!\*)", "", text)

    # Remove source attribution lines (before URL removal, since they contain URLs)
    text = _SOURCE_LINE_RE.sub("", text)

    # Remove bibliographic reference lines
    text = _BIBLIO_RE.sub("", text)
    text = _ENDNOTE_RE.sub("", text)

    # Remove URLs (run after markdown stripping so bare URLs are caught)
    text = _URL_RE.sub("", text)

    # Remove citation numbers
    text = _CITATION_RE.sub("", text)

    # Remove footers and page numbers
    text = _FOOTER_RE.sub("", text)
    text = _PAGE_NUM_RE.sub("", text)

    # Remove repeating header lines (e.g. "A PUBLICATION OF THE HOOVER INSTITUTION")
    text = _HEADER_LINE_RE.sub("", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --- LLM semantic cleaning ---

SYSTEM_PROMPT = """\
You are an audiobook transcript editor. You receive partially cleaned text from a PDF.

Your job:
1. Fix broken words and hyphenation artifacts from PDF line breaks.
2. Remove any remaining headers, footers, or page numbers you notice.
3. Keep lines starting with "Chapter:", "Section:", or "Image description:" exactly as they are.
4. Remove definition boxes, sidebars, or footnotes that interrupt the narrative flow.
5. When you encounter a [TABLE]...[/TABLE] block, convert it into natural spoken prose \
that conveys the same information. Organize by rows or themes, whichever reads more \
naturally for a listener. Remove the [TABLE] and [/TABLE] markers. If there is nearby \
text that appears to be a garbled version of the same table, remove the garbled text.
6. Preserve every body paragraph. Do not summarize or condense.
7. Output clean plain text only. No markdown. No commentary."""

SYSTEM_PROMPT_RAW = """\
You are an audiobook transcript editor. You receive raw text extracted from a PDF.
The text may contain markdown formatting, sidebars, footnotes, headers, footers,
page numbers, citation numbers, figure labels, and other non-narrative artifacts.

Your job:
1. Remove all markdown formatting (headings, bold, italic, links, code spans).
2. Remove headers, footers, page numbers, and running titles.
3. Remove sidebars, definition boxes, glossary sections, and footnotes.
4. Remove figure/table labels, citation numbers, and bibliographic references.
5. Remove URLs and source attribution lines.
6. Fix broken words and hyphenation artifacts from PDF line breaks.
7. Classify major sections: output "Chapter: <title>" for chapter-level headings
   and "Section: <title>" for sub-sections. Keep "Image description:" lines as-is.
8. When you encounter a [TABLE]...[/TABLE] block, convert it into natural spoken prose \
that conveys the same information. Organize by rows or themes, whichever reads more \
naturally for a listener. Remove the [TABLE] and [/TABLE] markers. If there is nearby \
text that appears to be a garbled version of the same table, remove the garbled text.
9. Preserve every body paragraph. Do not summarize or condense.
10. Output clean plain text only. No markdown. No commentary."""


def _build_chunks(pages: list[PageChunk], max_chars: int) -> list[list[PageChunk]]:
    """Group consecutive pages into chunks not exceeding max_chars."""
    chunks: list[list[PageChunk]] = []
    current: list[PageChunk] = []
    current_len = 0

    for page in pages:
        page_len = len(page.text)
        if current and current_len + page_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(page)
        current_len += page_len

    if current:
        chunks.append(current)

    return chunks


def _clean_chunk(text: str, config: Config) -> str:
    """Send a chunk to Ollama for cleaning."""
    prompt = SYSTEM_PROMPT if config.preprocessing else SYSTEM_PROMPT_RAW
    response = config.get_ollama_client().chat(
        model=config.ollama_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        options={
            "temperature": config.llm_temperature,
            "num_ctx": config.llm_num_ctx,
        },
        think=False,
        keep_alive="30m",
    )
    result = response.message.content

    # Safeguard: if cleaned text is much shorter than input, the LLM dropped content
    if len(result) < len(text) * 0.6:
        log.warning(
            "LLM output suspiciously short (%d chars vs %d input) — using raw text",
            len(result), len(text),
        )
        return text

    return result


def _merge_chunk_boundaries(parts: list[str]) -> str:
    """Join cleaned chunks, merging sentences split across chunk boundaries."""
    if not parts:
        return ""

    merged = parts[0]
    for part in parts[1:]:
        # Check if previous chunk ends mid-sentence
        # Look past any trailing image descriptions
        prev_stripped = merged.rstrip()
        if not prev_stripped:
            merged += "\n\n" + part
            continue

        # Find last body line (skip trailing image descriptions)
        prev_lines = prev_stripped.split("\n")
        body_end = prev_stripped
        for line in reversed(prev_lines):
            line_s = line.strip()
            if line_s and not line_s.startswith("Image description:"):
                body_end = line_s
                break

        last_char = body_end[-1] if body_end else ""
        ends_mid_sentence = last_char not in ".!?\"'\u201d"

        # Check if next chunk starts with lowercase or continuation
        next_stripped = part.lstrip()
        first_char = next_stripped[0] if next_stripped else ""
        starts_continuation = next_stripped and (
            first_char.islower()
            or first_char == "("
            or next_stripped.startswith("and ")
            or next_stripped.startswith("or ")
            or next_stripped.startswith("but ")
        )

        if ends_mid_sentence and starts_continuation:
            # Merge directly — the sentence was split at the boundary
            merged = prev_stripped + " " + next_stripped
        else:
            merged += "\n\n" + part

    return merged


def _postprocess_chunk(text: str) -> str:
    """Apply per-chunk post-processing (regex cleanup safe to run independently)."""
    text = _URL_RE.sub("", text)
    text = _SOURCE_LINE_RE.sub("", text)
    text = _BIBLIO_RE.sub("", text)
    text = _ENDNOTE_RE.sub("", text)
    # Strip any [TABLE]/[/TABLE] markers the LLM failed to remove
    text = text.replace("[TABLE]", "").replace("[/TABLE]", "")
    text = re.sub(
        r"^(?:Chapter|Section):\s*(?:" + "|".join(_SIDEBAR_HEADINGS) + r")\s*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    _major = ("preface", "introduction", "conclusion", "appendix", "acknowledgments")
    for name in _major:
        text = re.sub(
            rf"^{name}\s*$",
            f"Chapter: {name.capitalize()}",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_and_yield(pages: list[PageChunk], config: Config):
    """Streaming variant of clean_transcript — yields cleaned text chunks.

    Does NOT apply cross-chunk operations (back-matter stripping, mid-sentence
    fix, front-matter stripping). Those require the full transcript and should
    be handled by the consumer or applied after collection.
    """
    # Phase 1: Mechanical pre-processing
    if config.preprocessing:
        for page in pages:
            page.text = _preprocess(page.text)
        pages = _merge_page_breaks(pages)

    chunks = _build_chunks(pages, config.max_chunk_chars)
    log.info("Streaming clean: %d pages in %d chunks via %s", len(pages), len(chunks), config.ollama_model)

    def _clean_one(i, chunk):
        raw_text = "\n\n".join(p.text for p in chunk)
        page_range = f"{chunk[0].page_number}-{chunk[-1].page_number}"
        log.info("Cleaning chunk %d/%d (pages %s, %d chars)", i + 1, len(chunks), page_range, len(raw_text))
        try:
            return _clean_chunk(raw_text, config)
        except Exception:
            log.warning("LLM cleaning failed for chunk %d, using raw", i + 1, exc_info=True)
            return raw_text

    if config.ollama_parallel > 1:
        # Submit all, yield in order (futures list preserves submission order)
        with ThreadPoolExecutor(max_workers=config.ollama_parallel) as pool:
            futures = [pool.submit(_clean_one, i, chunk) for i, chunk in enumerate(chunks)]
            for future in tqdm(futures, desc="Cleaning", unit="chunk"):
                text = future.result()  # blocks until this chunk done, preserves order
                yield _postprocess_chunk(text)
    else:
        for i, chunk in enumerate(tqdm(chunks, desc="Cleaning", unit="chunk")):
            text = _clean_one(i, chunk)
            yield _postprocess_chunk(text)


def _is_structural(text: str) -> bool:
    """Check if a paragraph is a structural cue (not body text)."""
    return text.startswith(("Chapter:", "Section:", "Image description:", "[TABLE]"))


def _fix_mid_sentence_breaks(text: str) -> str:
    """Merge paragraphs that were split mid-sentence (from PDF page breaks)."""
    paragraphs = re.split(r"\n\n+", text)
    merged: list[str] = []

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if not merged:
            merged.append(stripped)
            continue

        # Don't merge into or out of structural lines
        if _is_structural(stripped):
            merged.append(stripped)
            continue

        # Find the nearest preceding body paragraph (skip image descriptions)
        body_idx = len(merged) - 1
        while body_idx >= 0 and _is_structural(merged[body_idx]):
            body_idx -= 1
        if body_idx < 0:
            merged.append(stripped)
            continue

        prev_body = merged[body_idx]
        last_char = prev_body.rstrip()[-1] if prev_body.rstrip() else ""
        ends_mid = last_char not in ".!?\"'\u201d:)"

        first_word = stripped.split()[0] if stripped.split() else ""
        starts_continuation = (
            first_word[:1].islower()
            or first_word[:1] == "("
            or first_word in ("and", "or", "but", "nor", "yet", "so", "many", "resources")
        )

        if ends_mid and starts_continuation:
            merged[body_idx] = prev_body.rstrip() + " " + stripped
        else:
            merged.append(stripped)

    return "\n\n".join(merged)


def _split_body_and_images(text: str) -> tuple[str, str]:
    """Split page text into body content and trailing image descriptions."""
    # Find the first image description marker
    marker = "\nImage description:"
    pos = text.find(marker)
    if pos == -1:
        marker = "\n\nImage description:"
        pos = text.find(marker)
    if pos == -1:
        return text, ""
    return text[:pos].rstrip(), text[pos:]


def _merge_page_breaks(pages: list[PageChunk]) -> list[PageChunk]:
    """Merge text across page boundaries where sentences are split."""
    if len(pages) < 2:
        return pages

    for i in range(len(pages) - 1):
        # Separate body text from trailing image descriptions so image
        # descriptions don't mask a mid-sentence break
        body, trailing_imgs = _split_body_and_images(pages[i].text)
        body = body.rstrip()
        next_text = pages[i + 1].text.lstrip()
        if not body or not next_text:
            continue

        last_char = body[-1]
        ends_mid = last_char not in ".!?\"'\u201d:)"

        first_word = next_text.split()[0] if next_text.split() else ""
        starts_cont = first_word[:1].islower() or first_word.lower() in (
            "and", "or", "but", "nor", "yet", "so", "many", "resources",
        )

        if ends_mid and starts_cont:
            # Move continuation text from next page to current page
            # Find the end of the continued sentence in the next page
            sent_end = None
            for j, ch in enumerate(next_text):
                if ch in ".!?" and (j + 1 >= len(next_text) or next_text[j + 1] in " \n\"'\u201d"):
                    sent_end = j + 1
                    break

            if sent_end is not None:
                continuation = next_text[:sent_end].strip()
                remainder = next_text[sent_end:].strip()
                pages[i].text = body + " " + continuation + trailing_imgs
                pages[i + 1].text = remainder
            else:
                # Whole next page is continuation — merge entirely
                pages[i].text = body + " " + next_text + trailing_imgs
                pages[i + 1].text = ""
        elif trailing_imgs:
            # Reassemble unchanged (body was stripped, so restore)
            pages[i].text = body + trailing_imgs

    # Remove empty pages
    return [p for p in pages if p.text.strip()]


def clean_transcript(pages: list[PageChunk], config: Config) -> str:
    """Clean extracted pages into a narration-ready transcript."""
    # Phase 1: Mechanical pre-processing (skipped when preprocessing is off)
    if config.preprocessing:
        for page in pages:
            page.text = _preprocess(page.text)

        # Merge sentences split across page boundaries
        pages = _merge_page_breaks(pages)

    # Phase 2: LLM semantic cleaning
    chunks = _build_chunks(pages, config.max_chunk_chars)
    log.info("Cleaning %d pages in %d chunks via %s", len(pages), len(chunks), config.ollama_model)

    cleaned_parts: list[str] = [None] * len(chunks)

    if config.ollama_parallel > 1:
        with ThreadPoolExecutor(max_workers=config.ollama_parallel) as pool:
            futures = {}
            for i, chunk in enumerate(chunks):
                raw_text = "\n\n".join(p.text for p in chunk)
                page_range = f"{chunk[0].page_number}-{chunk[-1].page_number}"
                log.info("Submitting chunk %d/%d (pages %s, %d chars)", i + 1, len(chunks), page_range, len(raw_text))
                futures[pool.submit(_clean_chunk, raw_text, config)] = i

            for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning", unit="chunk"):
                idx = futures[future]
                try:
                    cleaned_parts[idx] = future.result()
                except Exception:
                    chunk = chunks[idx]
                    page_range = f"{chunk[0].page_number}-{chunk[-1].page_number}"
                    raw_text = "\n\n".join(p.text for p in chunk)
                    log.warning("LLM cleaning failed for chunk %d (pages %s), using raw text", idx + 1, page_range, exc_info=True)
                    cleaned_parts[idx] = raw_text
    else:
        for i, chunk in enumerate(tqdm(chunks, desc="Cleaning", unit="chunk"), 0):
            raw_text = "\n\n".join(p.text for p in chunk)
            page_range = f"{chunk[0].page_number}-{chunk[-1].page_number}"
            log.info("Cleaning chunk %d/%d (pages %s, %d chars)", i + 1, len(chunks), page_range, len(raw_text))
            try:
                cleaned_parts[i] = _clean_chunk(raw_text, config)
            except Exception:
                log.warning("LLM cleaning failed for chunk %d (pages %s), using raw text", i + 1, page_range, exc_info=True)
                cleaned_parts[i] = raw_text

    result = "\n\n".join(cleaned_parts)

    # Phase 3: Post-process to catch anything the LLM re-introduced
    result = _URL_RE.sub("", result)
    result = _SOURCE_LINE_RE.sub("", result)
    result = _BIBLIO_RE.sub("", result)
    result = _ENDNOTE_RE.sub("", result)
    # Strip any [TABLE]/[/TABLE] markers the LLM failed to remove
    result = result.replace("[TABLE]", "").replace("[/TABLE]", "")
    # Remove any sidebar headings the LLM re-labelled
    result = re.sub(
        r"^(?:Chapter|Section):\s*(?:" + "|".join(_SIDEBAR_HEADINGS) + r")\s*$",
        "",
        result,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Restore "Chapter:" prefix on major sections the LLM may have stripped
    _major = ("preface", "introduction", "conclusion", "appendix", "acknowledgments")
    for name in _major:
        result = re.sub(
            rf"^{name}\s*$",
            f"Chapter: {name.capitalize()}",
            result,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # Strip back-matter: everything from References/Endnotes heading onward
    bm_match = _BACKMATTER_HEADINGS.search(result)
    if bm_match:
        log.info("Stripping back-matter from position %d (heading: %s)",
                 bm_match.start(), bm_match.group().strip())
        result = result[:bm_match.start()]

    # Strip front-matter metadata lines before first Chapter/Section cue
    first_cue = re.search(r"^(?:Chapter|Section):", result, re.MULTILINE)
    if first_cue and first_cue.start() > 0:
        preamble = result[:first_cue.start()]
        # Only strip if preamble is short (< 500 chars) — likely metadata, not content
        if len(preamble.strip()) < 500:
            log.info("Stripping %d chars of front-matter metadata", len(preamble.strip()))
            result = result[first_cue.start():]

    result = re.sub(r"\n{3,}", "\n\n", result)

    # Fix paragraphs split mid-sentence by page/chunk breaks
    result = _fix_mid_sentence_breaks(result)

    return result.strip()

"""Image description via Ollama vision model."""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import Config
from .extract import PageChunk

log = logging.getLogger(__name__)

IMAGE_PROMPT = (
    "Briefly describe this image from a PDF for an audiobook listener who cannot see it. "
    "Keep your description under 40 words. "
    "For charts: state the chart type and key trend. "
    "For photos: state the subject. "
    "For diagrams: state what it shows. "
    "Be factual. No preamble."
)


def _describe_image(image_bytes: bytes, config: Config) -> str:
    """Send an image to the vision model and get a description."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # minicpm-v needs a small context; larger vision models can use the full context
    num_ctx = 4096 if "minicpm" in config.vision_model else config.llm_num_ctx
    response = config.get_ollama_client().chat(
        model=config.vision_model,
        messages=[{
            "role": "user",
            "content": IMAGE_PROMPT,
            "images": [b64],
        }],
        options={
            "temperature": 0.3,
            "num_ctx": num_ctx,
        },
        think=False,
        keep_alive="30m",
    )
    return response.message.content.strip()


def describe_images(pages: list[PageChunk], config: Config) -> list[PageChunk]:
    """Add image descriptions into page text for pages that contain images."""
    if not config.vision_model:
        return pages

    total_images = sum(len(p.images) for p in pages)
    if total_images == 0:
        return pages

    log.info("Describing %d images via %s", total_images, config.vision_model)

    page_image_pairs = [
        (page, img)
        for page in pages
        for img in page.images
    ]

    if config.ollama_parallel > 1:
        with ThreadPoolExecutor(max_workers=config.ollama_parallel) as pool:
            futures = {}
            for page, img in page_image_pairs:
                futures[pool.submit(_describe_image, img.image_bytes, config)] = (page, img)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Describing images", unit="img"):
                page, img = futures[future]
                try:
                    desc = future.result()
                    page.text = page.text.rstrip() + f"\n\nImage description: {desc}"
                except Exception:
                    log.warning("Failed to describe image on page %d, skipping", page.page_number, exc_info=True)
    else:
        for page, img in tqdm(page_image_pairs, desc="Describing images", unit="img"):
            try:
                desc = _describe_image(img.image_bytes, config)
                page.text = page.text.rstrip() + f"\n\nImage description: {desc}"
            except Exception:
                log.warning("Failed to describe image on page %d, skipping", page.page_number, exc_info=True)

    return pages

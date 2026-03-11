"""Image description via Ollama vision model."""

import base64
import logging

from .config import Config
from .extract import PageChunk

log = logging.getLogger(__name__)

IMAGE_PROMPT = (
    "You are describing an image from an academic or professional PDF document "
    "for a listener who cannot see it. "
    "If it is a graph or chart: describe the type of chart, axes, key data points, and trends. "
    "If it is a photograph: describe the subject, setting, and significance. "
    "If it is a diagram or schematic: describe the components and relationships. "
    "Write 2-4 clear, specific sentences. Be factual and detailed."
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
    img_num = 0

    for page in pages:
        if not page.images:
            continue

        descriptions: list[str] = []
        for img in page.images:
            img_num += 1
            log.info("Describing image %d/%d (page %d)", img_num, total_images, page.page_number)
            try:
                desc = _describe_image(img.image_bytes, config)
                descriptions.append(f"Image description: {desc}")
            except Exception:
                log.warning("Failed to describe image on page %d, skipping", page.page_number, exc_info=True)

        if descriptions:
            page.text = page.text.rstrip() + "\n\n" + "\n\n".join(descriptions)

    return pages

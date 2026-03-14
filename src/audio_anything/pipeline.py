"""Top-level pipeline orchestrator."""

import logging
import time
from pathlib import Path

from .audio import export_m4b, export_mp3, synthesize_audio
from .clean import clean_transcript
from .config import Config
from .describe import describe_images
from .extract import extract_pages
from .tts import get_tts_backend

log = logging.getLogger(__name__)


def _validate_environment(config: Config) -> None:
    """Check that required services are available."""
    if not config.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {config.pdf_path}")

    # Check Ollama is running and model is available
    try:
        models = config.get_ollama_client().list()
        model_names = [m.model for m in models.models]
        if not any(config.ollama_model in name for name in model_names):
            log.warning(
                "Model %s not found in Ollama. Available: %s. "
                "Run: ollama pull %s",
                config.ollama_model, model_names, config.ollama_model,
            )
        if config.vision_model and not any(config.vision_model in name for name in model_names):
            log.warning(
                "Vision model %s not found in Ollama. Available: %s. "
                "Run: ollama pull %s",
                config.vision_model, model_names, config.vision_model,
            )
    except Exception:
        log.warning("Could not connect to Ollama. Is it running?", exc_info=True)


def run(config: Config, transcript_path: str | None = None) -> None:
    """Run the full PDF-to-audiobook pipeline."""
    start = time.time()
    stem = config.pdf_path.stem
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if transcript_path:
        # Skip extraction/cleaning, use existing transcript
        tp = Path(transcript_path)
        if not tp.exists():
            raise FileNotFoundError(f"Transcript not found: {tp}")
        transcript = tp.read_text()
        log.info("Using existing transcript: %s (%d chars)", tp, len(transcript))
    else:
        log.info(
            "Profile: %s | model: %s | vision: %s | preprocessing: %s | ctx: %d | chunks: %d%s",
            config.profile, config.ollama_model, config.vision_model or "disabled",
            config.preprocessing, config.llm_num_ctx, config.max_chunk_chars,
            f" | host: {config.ollama_host}" if config.ollama_host else "",
        )

        _validate_environment(config)

        # Phase 1: Extract
        t0 = time.time()
        pages = extract_pages(config.pdf_path, extract_images=bool(config.vision_model))
        log.info("Extraction: %.1fs (%d pages)", time.time() - t0, len(pages))

        # Phase 1.5: Describe images
        if config.vision_model:
            t0 = time.time()
            pages = describe_images(pages, config)
            log.info("Image description: %.1fs", time.time() - t0)

        # Phase 2: Clean
        t0 = time.time()
        transcript = clean_transcript(pages, config)
        log.info("Cleaning: %.1fs", time.time() - t0)

        # Save transcript
        saved_path = config.output_dir / f"{stem}_transcript.md"
        saved_path.write_text(transcript)
        log.info("Saved transcript to %s", saved_path)

        if config.dry_run:
            log.info("Dry run complete. Transcript saved, skipping TTS.")
            return

    # Phase 3: Synthesize
    t0 = time.time()
    backend = get_tts_backend(config)
    audio, chapters = synthesize_audio(transcript, backend, config)
    log.info("Synthesis: %.1fs (%d chapters detected)", time.time() - t0, len(chapters))

    if len(audio) == 0:
        log.error("No audio produced, skipping export")
        return

    # Phase 4: Export
    t0 = time.time()
    output_path = config.output_dir / stem
    if config.output_format == "m4b":
        out_file = export_m4b(audio, output_path, config, chapters)
    else:
        out_file = export_mp3(audio, output_path, config)
    log.info("Export: %.1fs", time.time() - t0)

    total = time.time() - start
    log.info("Pipeline complete in %.1fs → %s", total, out_file)

"""CLI entry point for audio-anything."""

import argparse
import logging
import sys

from .config import PROFILES, Config
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audio-anything",
        description="Convert a PDF into an audiobook.",
    )
    parser.add_argument("pdf_paths", nargs="*", help="Path(s) to input PDF file(s)")
    parser.add_argument("-o", "--output-dir", default="./output", help="Output directory (default: ./output)")
    parser.add_argument(
        "-t", "--tts-backend",
        choices=["kokoro", "orpheus", "elevenlabs", "openai"],
        default="kokoro",
        help="TTS backend (default: kokoro)",
    )
    parser.add_argument(
        "-f", "--output-format",
        choices=["m4b", "mp3"],
        default="m4b",
        help="Output audio format (default: m4b)",
    )
    parser.add_argument("-v", "--voice", default=None, help="Voice name/ID for chosen backend")
    parser.add_argument(
        "-p", "--profile",
        choices=list(PROFILES),
        default="home",
        help="Compute profile (default: home)",
    )
    parser.add_argument("-m", "--model", default=None, help="Ollama model (overrides profile default)")
    parser.add_argument("--vision-model", default=None, help="Ollama vision model (overrides profile default, use 'none' to disable)")
    parser.add_argument(
        "--preprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable mechanical preprocessing (overrides profile default)",
    )
    parser.add_argument("--ollama-host", default=None, help="Ollama server URL (default: localhost)")
    parser.add_argument("--dry-run", action="store_true", help="Extract + clean only, skip TTS")
    parser.add_argument("--estimate", action="store_true", help="Show estimated duration/segments and exit (no LLM or TTS)")
    parser.add_argument("--transcript", default=None, help="Skip extraction/cleaning, use existing transcript file for TTS")
    parser.add_argument("--input-dir", default=None, help="Process all PDFs in this directory")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Pass "none" as string — Config.__post_init__ handles the conversion
    vision_model = args.vision_model
    if vision_model is not None and vision_model.lower() == "none":
        vision_model = "none"

    from pathlib import Path
    from tqdm import tqdm

    # Collect all PDF paths
    pdf_paths: list[Path] = [Path(p) for p in args.pdf_paths]
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            logging.error("Input directory not found: %s", input_dir)
            sys.exit(1)
        pdf_paths.extend(sorted(input_dir.glob("*.pdf")))

    if not pdf_paths:
        parser.error("No PDF files specified. Provide paths or use --input-dir.")

    # Remove duplicates, preserve order
    seen = set()
    unique_paths = []
    for p in pdf_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)
    pdf_paths = unique_paths

    failed: list[tuple[Path, str]] = []

    pdf_iter = pdf_paths
    if len(pdf_paths) > 1:
        pdf_iter = tqdm(pdf_paths, desc="Batch", unit="pdf")

    for pdf_path in pdf_iter:
        config = Config(
            pdf_path=pdf_path,
            output_dir=args.output_dir,
            output_format=args.output_format,
            tts_backend=args.tts_backend,
            voice=args.voice,
            profile=args.profile,
            ollama_model=args.model,
            vision_model=vision_model,
            preprocessing=args.preprocessing,
            ollama_host=args.ollama_host,
            estimate=args.estimate,
            dry_run=args.dry_run,
            log_level=args.log_level,
        )

        try:
            run(config, transcript_path=args.transcript)
        except FileNotFoundError as e:
            logging.error(str(e))
            if len(pdf_paths) == 1:
                sys.exit(1)
            failed.append((pdf_path, str(e)))
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
            sys.exit(130)
        except Exception as e:
            logging.exception("Pipeline failed for %s", pdf_path)
            if len(pdf_paths) == 1:
                sys.exit(1)
            failed.append((pdf_path, str(e)))

    if failed:
        logging.error("Failed PDFs (%d/%d):", len(failed), len(pdf_paths))
        for path, err in failed:
            logging.error("  %s: %s", path, err)
        sys.exit(1)

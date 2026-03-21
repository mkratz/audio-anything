# Audio Anything

A local-first PDF-to-audiobook pipeline. Drop in a PDF, get back an M4B audiobook with chapters — no cloud services required.

Audio Anything extracts text from PDFs, cleans it into a narration-ready transcript using an LLM (via Ollama), synthesizes speech with a local TTS engine, and exports a chaptered audiobook file. It runs entirely on your machine, optimized for Apple Silicon.

## How It Works

The pipeline runs in four phases:

1. **Extract** — Pulls text and images from the PDF using PyMuPDF4LLM, preserving document structure. Detects and skips table-of-contents pages, filters out low-information images.

2. **Describe** *(optional)* — Sends extracted images to an Ollama vision model, generating concise descriptions that get woven into the transcript (e.g. "Image description: bar chart showing growth in R&D spending").

3. **Clean** — Two-stage cleaning:
   - *Mechanical preprocessing*: strips markdown artifacts, footers, page numbers, URLs, citations, bibliography entries, and sidebars via regex
   - *LLM semantic cleaning*: sends chunks to Ollama to fix hyphenation, remove remaining non-narrative content, and classify structural elements as chapters/sections
   - Merges paragraphs broken across page boundaries

4. **Synthesize & Export** — Splits the transcript into sentence-aware segments, synthesizes audio via the configured TTS backend, injects silence at chapter/section breaks, and exports to M4B (with chapter markers) or MP3.

## Features

- **Local-first**: Runs entirely on your machine using Ollama + Kokoro TTS. No API keys needed for the default setup.
- **Batch processing**: Pass multiple PDF paths or use `--input-dir` to process a whole folder.
- **Checkpoint/resume**: TTS progress is saved segment-by-segment. If a run is interrupted, restart and it picks up where it left off.
- **Cost estimation**: Use `--estimate` to preview projected duration, segment count, and chapter count before committing to a full run.
- **Dry-run mode**: `--dry-run` extracts and cleans the transcript without synthesizing audio — useful for reviewing transcript quality.
- **Chapter detection**: Automatically creates chapter markers from document structure, embedded in the M4B as bookmarks.
- **Image descriptions**: Optional vision model integration describes charts and figures for the narration.
- **Multiple TTS backends**: Kokoro (local, default), Orpheus (local GPU), ElevenLabs (cloud), OpenAI (cloud).
- **Configuration profiles**: `home` (CPU, smaller model) and `gpu` (GPU, larger model) with per-flag overrides.
- **Transcript reuse**: Re-synthesize a saved transcript with `--transcript` to try different voices or backends without re-extracting.

## Prerequisites

**Required:**

- Python >= 3.10
- [Ollama](https://ollama.com) running locally (or accessible remotely)
- ffmpeg

```bash
# macOS
brew install ffmpeg ollama

# Pull the default model
ollama pull qwen3.5:9b
```

## Installation

```bash
git clone https://github.com/your-username/audio_anything.git
cd audio_anything
pip install -e .
```

For cloud TTS backends:

```bash
pip install -e ".[elevenlabs]"   # ElevenLabs
pip install -e ".[openai]"       # OpenAI
```

## Quick Start

```bash
# Convert a PDF to an M4B audiobook
audio-anything document.pdf

# Preview estimated duration without running
audio-anything document.pdf --estimate

# Review the cleaned transcript first
audio-anything document.pdf --dry-run
# Then synthesize from the saved transcript
audio-anything --transcript output/document_transcript.md
```

## Usage

```
audio-anything [PDF_PATHS...] [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output-dir` | Output directory | `./output` |
| `-t, --tts-backend` | `kokoro`, `orpheus`, `elevenlabs`, or `openai` | `kokoro` |
| `-f, --output-format` | `m4b` or `mp3` | `m4b` |
| `-v, --voice` | Voice name/ID for the chosen backend | backend default |
| `-p, --profile` | `home` (CPU) or `gpu` | `home` |
| `-m, --model` | Override the Ollama model | profile default |
| `--vision-model` | Override the vision model (`none` to disable) | profile default |
| `--preprocessing` | `true` or `false` — toggle mechanical cleaning | profile default |
| `--ollama-host` | Ollama server URL | `localhost:11434` |
| `--input-dir` | Process all PDFs in a directory | — |
| `--transcript` | Skip extraction/cleaning, use existing transcript | — |
| `--dry-run` | Extract + clean only, skip TTS | — |
| `--estimate` | Show duration/segment estimates and exit | — |
| `--log-level` | `DEBUG`, `INFO`, or `WARNING` | `INFO` |

### Examples

```bash
# Batch process a folder
audio-anything --input-dir ./pdfs --output-dir ./audiobooks

# GPU profile with Orpheus TTS
audio-anything paper.pdf -p gpu -t orpheus

# Use a remote Ollama server
audio-anything paper.pdf --ollama-host http://192.168.1.100:11434

# Different voice
audio-anything book.pdf -v am_adam

# Multiple files
audio-anything ch1.pdf ch2.pdf ch3.pdf -o ./output
```

### Voices

| Backend | Available voices |
|---------|-----------------|
| Kokoro | `af_heart`, `am_adam`, `bf_bella`, `bm_george`, and others |
| Orpheus | `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe` |
| ElevenLabs | `Rachel`, `Domi`, `Bella`, `Antoni`, and others |
| OpenAI | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |

### Profiles

| Profile | Ollama Model | Context | Preprocessing | Best for |
|---------|-------------|---------|---------------|----------|
| `home` | `qwen3.5:9b` | 8K tokens | Enabled | MacBook Pro M2 (16GB) |
| `gpu` | `qwen3.5:27b` | 32K tokens | Enabled | GPU-accelerated systems |

## Output

Each run produces:

- **`{name}.m4b`** (or `.mp3`) — The audiobook file, with chapter bookmarks (M4B only)
- **`{name}_transcript.md`** — The cleaned transcript used for narration

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `ELEVENLABS_API_KEY` | `--tts-backend elevenlabs` |
| `OPENAI_API_KEY` | `--tts-backend openai` |

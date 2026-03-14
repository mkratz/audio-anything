"""Configuration dataclass with sensible defaults."""

from dataclasses import dataclass, field
from pathlib import Path

PROFILES = {
    "home": {
        "ollama_model": "qwen3.5:9b",
        "vision_model": "qwen3.5:9b",
        "preprocessing": True,
        "max_chunk_chars": 6_000,
        "llm_num_ctx": 8192,
    },
    "gpu": {
        "ollama_model": "qwen3.5:27b",
        "vision_model": "qwen3.5:27b",
        "preprocessing": True,
        "max_chunk_chars": 12_000,
        "llm_num_ctx": 32768,
    },
}


@dataclass
class Config:
    pdf_path: Path
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    tts_backend: str = "kokoro"
    voice: str | None = None
    ollama_model: str | None = None
    vision_model: str | None = None
    output_format: str = "m4b"
    dry_run: bool = False
    estimate: bool = False
    log_level: str = "INFO"

    # Profile
    profile: str = "home"
    preprocessing: bool | None = None
    ollama_host: str | None = None

    # Chunking
    max_chunk_chars: int | None = None

    # LLM
    llm_temperature: float = 0.3
    llm_num_ctx: int | None = None

    # Audio
    sample_rate: int = 24_000
    silence_duration: float = 0.75
    segment_max_chars: int = 500
    mp3_bitrate: str = "192k"

    # Internal
    _ollama_client: object = field(default=None, repr=False)

    def __post_init__(self):
        self.pdf_path = Path(self.pdf_path)
        self.output_dir = Path(self.output_dir)

        # Resolve profile defaults for None fields
        defaults = PROFILES[self.profile]
        if self.ollama_model is None:
            self.ollama_model = defaults["ollama_model"]
        if self.max_chunk_chars is None:
            self.max_chunk_chars = defaults["max_chunk_chars"]
        if self.llm_num_ctx is None:
            self.llm_num_ctx = defaults["llm_num_ctx"]
        if self.preprocessing is None:
            self.preprocessing = defaults["preprocessing"]

        # vision_model: "none" string → None, else fill from profile
        if self.vision_model is None:
            self.vision_model = defaults["vision_model"]
        elif self.vision_model.lower() == "none":
            self.vision_model = None

        if self.voice is None:
            self.voice = self._default_voice()

    def _default_voice(self) -> str:
        defaults = {
            "kokoro": "af_heart",
            "orpheus": "tara",
            "elevenlabs": "Rachel",
            "openai": "nova",
        }
        return defaults.get(self.tts_backend, "default")

    def get_ollama_client(self):
        """Return a (cached) ollama.Client, optionally pointed at a remote host."""
        if self._ollama_client is None:
            import ollama
            if self.ollama_host:
                self._ollama_client = ollama.Client(host=self.ollama_host)
            else:
                self._ollama_client = ollama.Client()
        return self._ollama_client

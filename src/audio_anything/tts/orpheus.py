"""Orpheus TTS backend via Ollama + SNAC codec.

Orpheus is a Llama-3B model fine-tuned to emit audio tokens that are
decoded through the SNAC (Multi-Scale Neural Audio Codec) into 24 kHz audio.
Requires a CUDA GPU for practical use — restricted to the 'gpu' profile.
"""

import logging
import re

import numpy as np
import torch

from ..config import Config
from .base import TTSBackend

log = logging.getLogger(__name__)

CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}

ORPHEUS_MODEL = "legraphista/Orpheus:3b-ft-q8"


class OrpheusBackend(TTSBackend):
    def __init__(self, config: Config):
        super().__init__(config)

        if config.profile != "gpu":
            raise RuntimeError(
                "Orpheus TTS requires the 'gpu' profile — it needs a CUDA GPU for SNAC decoding. "
                "Run with: -p gpu -t orpheus"
            )

        from snac import SNAC

        self._voice = config.voice if config.voice in VOICES else "tara"
        self._client = config.get_ollama_client()

        log.info("Loading SNAC 24kHz decoder...")
        self._snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._snac = self._snac.to(self._device)
        log.info("Initialized Orpheus TTS (voice=%s, model=%s, snac_device=%s)",
                 self._voice, ORPHEUS_MODEL, self._device)

    def _build_prompt(self, text: str) -> str:
        """Build the raw prompt Orpheus was trained on."""
        return (
            f"<custom_token_3>"          # start_of_human
            f"<|begin_of_text|>"
            f"{self._voice}: {text}"
            f"<|eot_id|>"
            f"<custom_token_4>"          # end_of_human
            f"<custom_token_5>"          # start_of_ai
            f"<custom_token_1>"          # start_of_speech
        )

    def _extract_token_ids(self, response_text: str) -> list[int]:
        """Parse <custom_token_N> strings into SNAC code values."""
        matches = CUSTOM_TOKEN_RE.findall(response_text)
        return [
            int(tok) - 10 - ((i % 7) * 4096)
            for i, tok in enumerate(matches)
        ]

    def _decode_to_audio(self, token_ids: list[int]) -> np.ndarray:
        """Decode a flat list of SNAC codes into a float32 audio array."""
        num_frames = len(token_ids) // 7
        if num_frames == 0:
            return np.array([], dtype=np.float32)

        token_ids = token_ids[:num_frames * 7]
        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(-1, 7)

        # Redistribute the interleaved 7-token frames into SNAC's 3 layers
        codes_0 = tokens[:, 0].unsqueeze(0).to(self._device)
        codes_1 = (
            torch.stack([tokens[:, 1], tokens[:, 4]], dim=1)
            .flatten().unsqueeze(0).to(self._device)
        )
        codes_2 = (
            torch.stack([tokens[:, 2], tokens[:, 3], tokens[:, 5], tokens[:, 6]], dim=1)
            .flatten().unsqueeze(0).to(self._device)
        )

        # Validate code ranges (each codebook has 4096 entries)
        for codes in (codes_0, codes_1, codes_2):
            if torch.any(codes < 0) or torch.any(codes > 4095):
                log.warning("SNAC codes out of range [0, 4095], skipping segment")
                return np.array([], dtype=np.float32)

        with torch.inference_mode():
            audio = self._snac.decode([codes_0, codes_1, codes_2])

        return audio.squeeze().cpu().numpy().astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        prompt = self._build_prompt(text)

        try:
            response = self._client.generate(
                model=ORPHEUS_MODEL,
                prompt=prompt,
                raw=True,
                stream=False,
                options={
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 8192,
                },
            )

            token_ids = self._extract_token_ids(response.response)
            if not token_ids:
                log.warning("No audio tokens generated for: %.80s...", text)
                return np.array([], dtype=np.float32)

            return self._decode_to_audio(token_ids)

        except Exception:
            log.warning("Orpheus synthesis failed, returning silence", exc_info=True)
            return np.array([], dtype=np.float32)

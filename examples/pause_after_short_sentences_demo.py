#!/usr/bin/env python3
"""Compare audio with and without pauses after very short sentences.

Usage:
    python examples/pause_after_short_sentences_demo.py

Output:
    pause_after_short_sentences_demo.wav
"""

from __future__ import annotations

import re

import numpy as np
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    ShortSentenceConfig,
    ShortSentenceInterval,
)

VOICE = "bf_lily" # some voices that tend to leave shorter pauses in general benefit more from this option
LANG = "en-gb"
SPEED = 0.9
OUTPUT_FILE = "pause_after_short_sentences_demo.wav"

TEXT = (
    "One … step. In front. Of. The other."
)

SHORT_SENTENCE_PATTERN = re.compile(r"([^.!?\n]+[.!?])(\s+)(?!\.\.\.)")
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
SSMD_BREAK_PATTERN = re.compile(r"\.\.\.(?:[csp]|\d+(?:\.\d+)?(?:ms|s)?)\b")


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)



def add_pause_after_short_sentences(text: str) -> str:
    """Add a clause pause after one- to three-word sentences."""

    def replace(match: re.Match[str]) -> str:
        sentence = match.group(1)
        whitespace = match.group(2)
        if SSMD_BREAK_PATTERN.search(sentence):
            return match.group(0)
        words = WORD_PATTERN.findall(sentence)
        if not 1 <= len(words) <= 3:
            return match.group(0)
        return f"{sentence} ...c{whitespace}"

    return SHORT_SENTENCE_PATTERN.sub(replace, text)


def generate(pipe: KokoroPipeline, text: str) -> tuple[np.ndarray, int]:
    result = pipe.run(text)
    return result.audio, result.sample_rate


def main() -> None:
    print_separator("PAUSE AFTER SHORT SENTENCES DEMO")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print(f"Speed: {SPEED}")
    print("Short-sentence mode: phrase, selection=end")

    plain_text = TEXT
    paused_text = add_pause_after_short_sentences(TEXT)

    print_separator("WITHOUT OPTION")
    print(plain_text)

    print_separator("WITH OPTION")
    print(paused_text)

    pipe = KokoroPipeline(
        PipelineConfig(
            voice=VOICE,
            generation=GenerationConfig(lang=LANG, speed=SPEED),
        )
    )

    try:
        sample_rate = 24000
        pause = np.zeros(int(sample_rate * 0.8), dtype=np.float32)
        short_pause = np.zeros(int(sample_rate * 0.35), dtype=np.float32)

        plain_heading, sample_rate = generate(
            pipe,
            "Without pause after short sentences.",
        )
        plain_audio, sample_rate = generate(pipe, plain_text)

        paused_heading, sample_rate = generate(
            pipe,
            "With pause after short sentences.",
        )
        paused_audio, sample_rate = generate(pipe, paused_text)

        combined = np.concatenate(
            [
                plain_heading,
                short_pause,
                plain_audio,
                pause,
                paused_heading,
                short_pause,
                paused_audio,
            ]
        )
        sf.write(OUTPUT_FILE, combined, sample_rate)
    finally:
        pipe.close()

    print_separator("OUTPUT")
    print(f"Created {OUTPUT_FILE}")
    print(f"Total duration: {len(combined) / sample_rate:.2f}s")
    print("Listen for the tighter first pass, then the paced second pass.")


if __name__ == "__main__":
    main()

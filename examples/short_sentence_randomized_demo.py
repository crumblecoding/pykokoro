#!/usr/bin/env python3
"""Generate audio comparing single and randomized short-sentence phrases."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    RandomizedPhraseResolveMode,
    ShortSentenceConfig,
    ShortSentenceInterval,
)

VOICE = "bf_lily"
LANG = "en-us"
SPEED = 0.83
TEST_SENTENCES = [
    "Yes!",
    "Hi!",
    "Why?",
    "Hush.",
    "Not yet.",
    "Mr. Vale.",
    "'Tis.",
    "Chapter IV.",
    "Hermione.",
]
OUTPUT_FILE = "short_sentence_randomized_demo.wav"


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def single_phrase_config(template: str) -> ShortSentenceConfig:
    """Force one phrase template for every input in a comparison block."""
    return ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase=template,
                end_phrase=template,
                cutter="energy-valley",
            )
        },
        intervals=[ShortSentenceInterval("single phrase", 40, "phrase")],
    )


def randomized_phrase_config(templates: list[str]) -> ShortSentenceConfig:
    """Force one randomized phrase family for every input in a comparison block."""
    return ShortSentenceConfig(
        resolve_modes={
            "randomized-phrase": RandomizedPhraseResolveMode(
                neutral_phrases=templates,
                end_phrases=templates,
                cutter="energy-valley",
            )
        },
        intervals=[
            ShortSentenceInterval("randomized phrases", 40, "randomized-phrase")
        ],
    )


def generate(text: str, config: ShortSentenceConfig | None) -> tuple[np.ndarray, int]:
    pipe = KokoroPipeline(
        PipelineConfig(
            voice=VOICE,
            generation=GenerationConfig(lang=LANG, speed=SPEED),
            short_sentence_config=config,
        )
    )
    try:
        result = pipe.run(text)
        return result.audio, result.sample_rate
    finally:
        pipe.close()


def main() -> None:
    single = PhraseResolveMode()
    randomized = RandomizedPhraseResolveMode()
    groups = [
        (
            "Generating using neutral phrases:",
            [
                ("Using short sentences with a single phrase.", single_phrase_config(single.neutral_phrase)),
                (
                    "Using short sentences with a randomized phrase.",
                    randomized_phrase_config(randomized.neutral_phrases),
                ),
            ],
        ),
        (
            "Generating using end phrases:",
            [
                ("Using short sentences with a single phrase.", single_phrase_config(single.end_phrase)),
                ("Using short sentences with a randomized phrase.", randomized_phrase_config(randomized.end_phrases)),
            ],
        ),
    ]

    sample_rate = 24000
    pause = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
    group_pause = np.zeros(int(sample_rate * 0.8), dtype=np.float32)
    all_samples: list[np.ndarray] = []

    print_separator("SHORT SENTENCE RANDOMIZED DEMO")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print(f"Speed: {SPEED}")

    for group_name, modes in groups:
        print_separator(group_name.upper())
        heading_audio, sample_rate = generate(group_name, ShortSentenceConfig(enabled=False))
        all_samples.extend([heading_audio, group_pause])

        for mode_name, config in modes:
            print(f"\n{mode_name}:")
            heading_audio, sample_rate = generate(
                mode_name,
                ShortSentenceConfig(enabled=False),
            )
            all_samples.extend([heading_audio, pause])
            for text in TEST_SENTENCES:
                audio, sample_rate = generate(text, config)
                print(f"  {text!r} -> {len(audio) / sample_rate:.3f}s")
                all_samples.extend([audio, pause])
            all_samples.append(group_pause)

    combined = np.concatenate(all_samples) if all_samples else np.array([], dtype=np.float32)
    sf.write(OUTPUT_FILE, combined, sample_rate)

    print_separator("OUTPUT")
    print(f"Created {OUTPUT_FILE}")
    print(f"Total duration: {len(combined) / sample_rate:.2f}s")
    print("Listen in the same order printed above.")


if __name__ == "__main__":
    main()

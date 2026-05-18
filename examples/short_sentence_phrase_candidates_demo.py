#!/usr/bin/env python3
"""Generate audio for comparing short-sentence carrier phrase candidates."""

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

VOICE = "af_heart"
LANG = "en-us"
SPEED = 0.83
TEST_TEXTS = ["Yes!", "One … step. In front."]
OUTPUT_FILE = "short_sentence_phrase_candidates_demo.wav"


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def phrase_config(template: str) -> ShortSentenceConfig:
    """Force one phrase template for both neutral and sentence-end inputs."""
    return ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase=template,
                end_phrase=template,
            )
        },
        intervals=[ShortSentenceInterval("candidate phrase", 40, "phrase")],
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
    candidates = RandomizedPhraseResolveMode()
    groups = [
        ("neutral phrases", candidates.neutral_phrases),
        ("end phrases", candidates.end_phrases),
    ]

    sample_rate = 24000
    pause = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
    group_pause = np.zeros(int(sample_rate * 0.8), dtype=np.float32)
    all_samples: list[np.ndarray] = []

    print_separator("SHORT SENTENCE PHRASE CANDIDATE DEMO")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print(f"Speed: {SPEED}")

    for text in TEST_TEXTS:
        print_separator(f"Input: {text}")

        for group_name, templates in groups:
            print(f"\n{group_name}:")
            heading_audio, sample_rate = generate(
                group_name,
                ShortSentenceConfig(enabled=False),
            )
            all_samples.extend([heading_audio, pause])
            for index, template in enumerate(templates, start=1):
                audio, sample_rate = generate(text, phrase_config(template))
                print(
                    f"  {index}. {template!r} -> "
                    f"{len(audio) / sample_rate:.3f}s"
                )
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

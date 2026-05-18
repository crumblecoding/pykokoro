#!/usr/bin/env python3
"""Rank voices by how cleanly they isolate inserted short text."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np

from pykokoro.audio_generator import _join_timestamps
from pykokoro.constants import SAMPLE_RATE
from pykokoro.onnx_backend import Kokoro
from pykokoro.short_sentence_handler import phonemize_short_sentence_phrase
from pykokoro.trim import energy_based_vad
from pykokoro.types import PhonemeSegment

LANG = "en-us"
SPEED = 1
FRAME_DURATION_MS = 5
ENERGY_THRESHOLD = 0.05
MIN_SILENCE_SECONDS = 0.02
MAX_BOUNDARY_GAP_SECONDS = 1.00
VOICE_FILTER_PREFIX = ("af_", "am_", "bf_", "bm_")
PHRASE_CANDIDATES = [
    "He paused for a long time. … {segment} … Then he continued.",
]
DEMO_SEGMENTS = [
    "Hi!",
    "Why?",
    "No.",
    "No!",
    "Yes!",
    "Help!",
    "Oh!",
    "Stop!",
    "What?",
    "Don't!",
    "One … step. In front.",
    "Of.",
    "The other.",
]


@dataclass
class VoiceScore:
    voice: str
    successes: int
    attempts: int
    mean_gap_seconds: float
    mean_pause_seconds: float

    @property
    def success_ratio(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


def main() -> None:
    kokoro = Kokoro()
    try:
        kokoro._init_kokoro()
        assert kokoro._session is not None
        assert kokoro._audio_generator is not None
        output_names = [output.name for output in kokoro._session.get_outputs()]
        if len(output_names) < 2:
            raise RuntimeError(
                "Loaded model has no timestamp output. "
                "Use the timestamped ONNX model before running this script."
            )

        scores = []
        for voice in english_voices(kokoro.get_voices()):
            score = score_voice(
                voice,
                phrases=PHRASE_CANDIDATES,
                audio_generator=kokoro._audio_generator,
                voice_style=kokoro.resolve_voice_style(voice),
            )
            scores.append(score)
            print(
                f"Processed {score.voice} | "
                f"success={score.successes}/{score.attempts} "
                f"({score.success_ratio:.0%}) | "
                f"mean_boundary_gap={score.mean_gap_seconds:.3f}s | "
                f"mean_pause={score.mean_pause_seconds:.3f}s"
            )
    finally:
        kokoro.close()

    ranked = sorted(
        scores,
        key=lambda score: (
            score.success_ratio,
            score.successes,
            -score.mean_gap_seconds,
            score.mean_pause_seconds,
        ),
        reverse=True,
    )
    ranked_by_successes = sorted(
        scores,
        key=lambda score: score.successes,
        reverse=True,
    )

    print_scores("Top 10 voices", ranked)
    print_scores("Top 10 voices by successes only", ranked_by_successes)


def print_scores(title: str, scores: list[VoiceScore]) -> None:
    print(title)
    for index, score in enumerate(scores[:10], start=1):
        print(
            f"{index}. {score.voice} | "
            f"success={score.successes}/{score.attempts} "
            f"({score.success_ratio:.0%}) | "
            f"mean_boundary_gap={score.mean_gap_seconds:.3f}s | "
            f"mean_pause={score.mean_pause_seconds:.3f}s"
        )


def english_voices(voices: list[str]) -> list[str]:
    return [
        voice
        for voice in voices
        if voice == "af" or voice.startswith(VOICE_FILTER_PREFIX)
    ]


def score_voice(
    voice: str,
    *,
    phrases: list[str],
    audio_generator: object,
    voice_style: np.ndarray,
) -> VoiceScore:
    gaps: list[float] = []
    pause_lengths: list[float] = []
    successes = 0
    attempts = 0

    for phrase in phrases:
        for index, text in enumerate(DEMO_SEGMENTS):
            attempts += 1
            pause_lengths.extend([0.0, 0.0])
            segment = PhonemeSegment(
                id=f"{voice}_{index}",
                segment_id=f"{voice}_{index}",
                phoneme_id=0,
                text=text,
                phonemes="",
                tokens=[],
                lang=LANG,
            )
            phonemes, _, timing_tokens = phonemize_short_sentence_phrase(
                segment,
                phrase,
            )
            audio, pred_dur = audio_generator._run_onnx(phonemes, voice_style, SPEED)
            if pred_dur is None:
                continue

            timestamps = _join_timestamps(timing_tokens, pred_dur)
            target_tokens = [
                token
                for token in timestamps
                if token.get("is_target")
                and isinstance(token.get("start_ts"), (int, float))
                and isinstance(token.get("end_ts"), (int, float))
            ]
            if not target_tokens:
                continue

            target_start = int(
                min(float(token["start_ts"]) for token in target_tokens) * SAMPLE_RATE
            )
            target_end = int(
                max(float(token["end_ts"]) for token in target_tokens) * SAMPLE_RATE
            )
            runs = quiet_runs(audio)
            before = nearest_run_before(runs, target_start)
            after = nearest_run_after(runs, target_end)
            if before is None or after is None:
                continue

            before_gap = max(0, target_start - before[1]) / SAMPLE_RATE
            after_gap = max(0, after[0] - target_end) / SAMPLE_RATE
            if (
                before_gap > MAX_BOUNDARY_GAP_SECONDS
                or after_gap > MAX_BOUNDARY_GAP_SECONDS
            ):
                continue

            successes += 1
            gaps.extend([before_gap, after_gap])
            pause_lengths[-2:] = [
                (before[1] - before[0]) / SAMPLE_RATE,
                (after[1] - after[0]) / SAMPLE_RATE,
            ]

    return VoiceScore(
        voice=voice,
        successes=successes,
        attempts=attempts,
        mean_gap_seconds=mean(gaps) if gaps else float("inf"),
        mean_pause_seconds=mean(pause_lengths) if pause_lengths else 0.0,
    )


def quiet_runs(audio: np.ndarray) -> list[tuple[int, int]]:
    speech_frames = energy_based_vad(
        audio,
        SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        energy_threshold=ENERGY_THRESHOLD,
    )
    quiet_frames = ~speech_frames
    samples_per_frame = max(1, int(SAMPLE_RATE * FRAME_DURATION_MS / 1000))
    min_frames = max(1, int(MIN_SILENCE_SECONDS * 1000 / FRAME_DURATION_MS))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_quiet in enumerate(quiet_frames):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            if index - start >= min_frames:
                runs.append((start * samples_per_frame, index * samples_per_frame))
            start = None
    if start is not None and len(quiet_frames) - start >= min_frames:
        runs.append((start * samples_per_frame, len(audio)))
    return runs


def nearest_run_before(
    runs: list[tuple[int, int]],
    anchor: int,
) -> tuple[int, int] | None:
    candidates = [run for run in runs if run[1] <= anchor]
    return max(candidates, key=lambda run: run[1], default=None)


def nearest_run_after(
    runs: list[tuple[int, int]],
    anchor: int,
) -> tuple[int, int] | None:
    candidates = [run for run in runs if run[0] >= anchor]
    return min(candidates, key=lambda run: run[0], default=None)


if __name__ == "__main__":
    main()

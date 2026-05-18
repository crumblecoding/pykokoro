#!/usr/bin/env python3
"""Rank short-sentence carrier phrases across a fixed set of voices."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).parent))

import find_short_sentence_end_phrase_candidates as end_phrase_candidates
import find_short_sentence_phrase_candidates as neutral_phrase_candidates
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
TOP_VOICE_COUNT_FOR_PHRASE_RANKING = 15
CACHE_FLUSH_EVERY_N_VOICES = 5
RESULTS_PATH = Path(__file__).with_name(f"{Path(__file__).stem}_results.json")

# Keep this list explicit so a run compares the same voice set every time.
VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
]

DEMO_SEGMENTS = neutral_phrase_candidates.DEMO_SEGMENTS
NEUTRAL_PHRASES = neutral_phrase_candidates.PHRASE_CANDIDATES
END_PHRASES = end_phrase_candidates.PHRASE_CANDIDATES


@dataclass(frozen=True)
class PhraseSpec:
    kind: str
    text: str
    require_after_boundary: bool


@dataclass
class PhraseScore:
    kind: str
    phrase: str
    successes: int
    attempts: int
    mean_gap_seconds: float
    mean_pause_seconds: float

    @property
    def success_ratio(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


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


@dataclass
class AttemptResult:
    success: bool
    gaps: list[float]
    pause_lengths: list[float]


ResultCache = dict[str, Any]
CACHE_SETTINGS = {
    "lang": LANG,
    "speed": SPEED,
    "frame_duration_ms": FRAME_DURATION_MS,
    "energy_threshold": ENERGY_THRESHOLD,
    "min_silence_seconds": MIN_SILENCE_SECONDS,
    "max_boundary_gap_seconds": MAX_BOUNDARY_GAP_SECONDS,
}


def main() -> None:
    phrase_specs = [
        *(PhraseSpec("neutral", phrase, True) for phrase in NEUTRAL_PHRASES),
        *(PhraseSpec("end", phrase, False) for phrase in END_PHRASES),
    ]

    kokoro = Kokoro(provider="cuda")
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

        voice_styles = {
            voice: kokoro.resolve_voice_style(voice)
            for voice in VOICES
        }
        result_cache = load_result_cache()
        print(f"Loaded {count_cached_attempts(result_cache)} cached attempts from {RESULTS_PATH.name}")
        phrase_results_by_spec: dict[PhraseSpec, dict[str, list[AttemptResult]]] = {}
        voice_attempts = {voice: [] for voice in VOICES}

        for phrase_index, spec in enumerate(phrase_specs, start=1):
            print(
                f"Scoring phrase {phrase_index}/{len(phrase_specs)} "
                f"[{spec.kind}] {spec.text!r}"
            )
            phrase_results_by_voice = {voice: [] for voice in VOICES}
            phrase_has_new_results = False
            for voice_index, (voice, voice_style) in enumerate(voice_styles.items(), start=1):
                cached_count = 0
                computed_count = 0
                for index, text in enumerate(DEMO_SEGMENTS):
                    result = get_cached_attempt(result_cache, spec, voice, text)
                    if result is None:
                        result = score_attempt(
                            spec,
                            text=text,
                            index=index,
                            voice=voice,
                            audio_generator=kokoro._audio_generator,
                            voice_style=voice_style,
                        )
                        set_cached_attempt(result_cache, spec, voice, text, result)
                        computed_count += 1
                        phrase_has_new_results = True
                    else:
                        cached_count += 1
                    phrase_results_by_voice[voice].append(result)
                    voice_attempts[voice].append(result)
                if computed_count and voice_index % CACHE_FLUSH_EVERY_N_VOICES == 0:
                    save_result_cache(result_cache)
                print(
                    f"  Finished voice {voice_index}/{len(voice_styles)} "
                    f"for current phrase: {voice} "
                    f"(computed={computed_count}, cached={cached_count})"
                )

            if phrase_has_new_results:
                save_result_cache(result_cache)
            phrase_results_by_spec[spec] = phrase_results_by_voice
            score = summarize_phrase(
                spec,
                [
                    attempt
                    for attempts in phrase_results_by_voice.values()
                    for attempt in attempts
                ],
            )
            print(
                f"Processed {score.kind} phrase {score.phrase!r} | "
                f"success={score.successes}/{score.attempts} "
                f"({score.success_ratio:.0%}) | "
                f"mean_boundary_gap={score.mean_gap_seconds:.3f}s | "
                f"mean_pause={score.mean_pause_seconds:.3f}s"
            )
    finally:
        kokoro.close()

    voice_scores = [
        summarize_voice(voice, attempts)
        for voice, attempts in voice_attempts.items()
    ]
    ranked_voices = sorted_scores(voice_scores)
    print_voice_scores("Voices ranked across all phrases", ranked_voices)

    top_voices = {
        score.voice
        for score in ranked_voices[:TOP_VOICE_COUNT_FOR_PHRASE_RANKING]
    }
    phrase_scores = [
        summarize_phrase(
            spec,
            [
                attempt
                for voice, attempts in phrase_results_by_voice.items()
                if voice in top_voices
                for attempt in attempts
            ],
        )
        for spec, phrase_results_by_voice in phrase_results_by_spec.items()
    ]
    ranked_neutral_phrases = sorted_scores(
        [score for score in phrase_scores if score.kind == "neutral"]
    )
    ranked_end_phrases = sorted_scores(
        [score for score in phrase_scores if score.kind == "end"]
    )
    print_phrase_scores(
        f"Neutral phrases ranked across top {TOP_VOICE_COUNT_FOR_PHRASE_RANKING} voices",
        ranked_neutral_phrases,
    )
    print_phrase_scores(
        f"End phrases ranked across top {TOP_VOICE_COUNT_FOR_PHRASE_RANKING} voices",
        ranked_end_phrases,
    )


def score_attempt(
    spec: PhraseSpec,
    *,
    text: str,
    index: int,
    voice: str,
    audio_generator: object,
    voice_style: np.ndarray,
) -> AttemptResult:
    expected_pause_count = 2 if spec.require_after_boundary else 1
    failure = AttemptResult(False, [], [0.0] * expected_pause_count)
    segment = PhonemeSegment(
        id=f"{voice}_{index}",
        segment_id=f"{voice}_{index}",
        phoneme_id=0,
        text=text,
        phonemes="",
        tokens=[],
        lang=LANG,
    )
    phonemes, _, timing_tokens = phonemize_short_sentence_phrase(segment, spec.text)
    audio, pred_dur = audio_generator._run_onnx(phonemes, voice_style, SPEED)
    if pred_dur is None:
        return failure

    timestamps = _join_timestamps(timing_tokens, pred_dur)
    target_tokens = [
        token
        for token in timestamps
        if token.get("is_target")
        and isinstance(token.get("start_ts"), (int, float))
        and isinstance(token.get("end_ts"), (int, float))
    ]
    if not target_tokens:
        return failure

    target_start = int(min(float(token["start_ts"]) for token in target_tokens) * SAMPLE_RATE)
    target_end = int(max(float(token["end_ts"]) for token in target_tokens) * SAMPLE_RATE)
    runs = quiet_runs(audio)
    before = nearest_run_before(runs, target_start)
    after = nearest_run_after(runs, target_end) if spec.require_after_boundary else None
    if before is None or (spec.require_after_boundary and after is None):
        return failure

    before_gap = max(0, target_start - before[1]) / SAMPLE_RATE
    gaps = [before_gap]
    pause_lengths = [(before[1] - before[0]) / SAMPLE_RATE]
    if after is not None:
        after_gap = max(0, after[0] - target_end) / SAMPLE_RATE
        gaps.append(after_gap)
        pause_lengths.append((after[1] - after[0]) / SAMPLE_RATE)

    if any(gap > MAX_BOUNDARY_GAP_SECONDS for gap in gaps):
        return failure

    return AttemptResult(True, gaps, pause_lengths)


def summarize_phrase(spec: PhraseSpec, attempts: list[AttemptResult]) -> PhraseScore:
    return PhraseScore(
        kind=spec.kind,
        phrase=spec.text,
        successes=sum(attempt.success for attempt in attempts),
        attempts=len(attempts),
        mean_gap_seconds=mean_gap(attempts),
        mean_pause_seconds=mean_pause(attempts),
    )


def summarize_voice(voice: str, attempts: list[AttemptResult]) -> VoiceScore:
    return VoiceScore(
        voice=voice,
        successes=sum(attempt.success for attempt in attempts),
        attempts=len(attempts),
        mean_gap_seconds=mean_gap(attempts),
        mean_pause_seconds=mean_pause(attempts),
    )


def mean_gap(attempts: list[AttemptResult]) -> float:
    gaps = [gap for attempt in attempts if attempt.success for gap in attempt.gaps]
    return mean(gaps) if gaps else float("inf")


def mean_pause(attempts: list[AttemptResult]) -> float:
    # Failed attempts already carry zero pause values so they affect the average.
    pause_lengths = [pause for attempt in attempts for pause in attempt.pause_lengths]
    return mean(pause_lengths) if pause_lengths else 0.0


def load_result_cache() -> ResultCache:
    if not RESULTS_PATH.exists():
        return {"version": 1, "settings": CACHE_SETTINGS, "phrases": {}}

    with RESULTS_PATH.open(encoding="utf-8") as results_file:
        payload = json.load(results_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid result cache in {RESULTS_PATH}: expected object")
    if payload.get("version") != 1 or not isinstance(payload.get("phrases"), dict):
        raise ValueError(f"Invalid result cache in {RESULTS_PATH}: unsupported schema")
    if payload.get("settings") != CACHE_SETTINGS:
        raise ValueError(
            f"Result cache in {RESULTS_PATH} was created with different scoring settings. "
            "Remove it before rerunning with the new settings."
        )
    return payload


def save_result_cache(result_cache: ResultCache) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(result_cache, results_file, ensure_ascii=False, indent=2, sort_keys=True)
        results_file.write("\n")


def count_cached_attempts(result_cache: ResultCache) -> int:
    return sum(
        len(voice_entry)
        for phrase_entry in result_cache["phrases"].values()
        if isinstance(phrase_entry, dict)
        for voice_entry in phrase_entry.get("voices", {}).values()
        if isinstance(voice_entry, dict)
    )


def phrase_cache_key(spec: PhraseSpec) -> str:
    return json.dumps(
        {
            "kind": spec.kind,
            "phrase": spec.text,
            "require_after_boundary": spec.require_after_boundary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def get_cached_attempt(
    result_cache: ResultCache,
    spec: PhraseSpec,
    voice: str,
    text: str,
) -> AttemptResult | None:
    phrase_entry = result_cache["phrases"].get(phrase_cache_key(spec))
    if not isinstance(phrase_entry, dict):
        return None
    voice_entry = phrase_entry.get("voices", {}).get(voice)
    if not isinstance(voice_entry, dict):
        return None
    attempt_entry = voice_entry.get(text)
    if not isinstance(attempt_entry, dict):
        return None
    if not isinstance(attempt_entry.get("success"), bool):
        return None
    gaps = attempt_entry.get("gaps")
    pause_lengths = attempt_entry.get("pause_lengths")
    if not isinstance(gaps, list) or not isinstance(pause_lengths, list):
        return None
    if not all(isinstance(value, (int, float)) for value in gaps):
        return None
    if not all(isinstance(value, (int, float)) for value in pause_lengths):
        return None
    return AttemptResult(
        success=attempt_entry["success"],
        gaps=[float(value) for value in gaps],
        pause_lengths=[float(value) for value in pause_lengths],
    )


def set_cached_attempt(
    result_cache: ResultCache,
    spec: PhraseSpec,
    voice: str,
    text: str,
    result: AttemptResult,
) -> None:
    phrase_entry = result_cache["phrases"].setdefault(
        phrase_cache_key(spec),
        {
            "kind": spec.kind,
            "phrase": spec.text,
            "require_after_boundary": spec.require_after_boundary,
            "voices": {},
        },
    )
    voice_entry = phrase_entry["voices"].setdefault(voice, {})
    voice_entry[text] = {
        "success": result.success,
        "gaps": result.gaps,
        "pause_lengths": result.pause_lengths,
    }


def sorted_scores(scores: list[PhraseScore] | list[VoiceScore]) -> list[PhraseScore] | list[VoiceScore]:
    return sorted(
        scores,
        key=lambda score: (
            score.success_ratio,
            score.successes,
            -score.mean_gap_seconds,
            score.mean_pause_seconds,
        ),
        reverse=True,
    )


def print_phrase_scores(title: str, scores: list[PhraseScore]) -> None:
    print(title)
    for index, score in enumerate(scores, start=1):
        print(
            f"{index}. [{score.kind}] {score.phrase!r} | "
            f"success={score.successes}/{score.attempts} "
            f"({score.success_ratio:.0%}) | "
            f"mean_boundary_gap={score.mean_gap_seconds:.3f}s | "
            f"mean_pause={score.mean_pause_seconds:.3f}s"
        )


def print_voice_scores(title: str, scores: list[VoiceScore]) -> None:
    print(title)
    for index, score in enumerate(scores, start=1):
        print(
            f"{index}. {score.voice} | "
            f"success={score.successes}/{score.attempts} "
            f"({score.success_ratio:.0%}) | "
            f"mean_boundary_gap={score.mean_gap_seconds:.3f}s | "
            f"mean_pause={score.mean_pause_seconds:.3f}s"
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

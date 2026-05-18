"""Short sentence handling for pykokoro using single-word context approach.

This module provides functionality to improve audio quality for short, single-word
sentences by applying a "context-prepending" technique during phoneme creation.

Only activates for short (<5 phonemes) AND single-word sentences (no spaces)

This approach produces better prosody and intonation compared to generating
very short sentences directly, as neural TTS models typically need more context
to produce natural-sounding speech.

Multi-word or sentences with internal breaks will NOT use this handler, as they
already have sufficient context for natural prosody.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .constants import MAX_PHONEME_LENGTH, SAMPLE_RATE, SUPPORTED_LANGUAGES

if TYPE_CHECKING:
    from .types import PhonemeSegment

logger = logging.getLogger(__name__)

SHORT_SENTENCE_META_KEY = "__short_sentence"
ResolveModeName = str | Literal[False]


@dataclass
class WrapResolveMode:
    """Configuration for phoneme pretext wrapping."""

    kind: Literal["wrap"] = "wrap"
    phoneme_pretext: str = "—"


@dataclass
class PhraseResolveMode:
    """Configuration for phrase generation and cutting."""

    kind: Literal["phrase"] = "phrase"
    neutral_phrase: str = "He paused for a long time. … {segment} … Then he continued."
    end_phrase: str = "The word is hello. The word is —'{segment}'"
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02


@dataclass
class RandomizedPhraseResolveMode:
    """Configuration for randomized phrase generation and cutting."""

    kind: Literal["randomized-phrase"] = "randomized-phrase"
    neutral_phrases: list[str] = field(
        default_factory=lambda: [
            "He paused for a long time. … {segment} … Then he continued.",
            "He paused. … {segment} Then he continued.",
           # "The transcript paused…: {segment}; the next entry followed.",
            "She looked up…: {segment}. The conversation resumed.",
            #"The line ended: {segment}; the next line began.",
            "The line ended…: {segment}; the next line began.",
            "The letter paused, {segment}, before the final line."
            "The conversation stopped, {segment}, before someone answered."
        ]
    )
    end_phrases: list[str] = field(
        default_factory=lambda: [
            "The judge asked for a final answer. {segment}",
            "The recording trails off after the words, … {segment}",
            "The final word is hello. The final word is '{segment}'",
            "The final word is hello. The final word is … '{segment}'",
            "The caller left one final message…: {segment}"
        ]
    )
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02


ShortSentenceResolveMode = (
    WrapResolveMode | PhraseResolveMode | RandomizedPhraseResolveMode
)


@dataclass
class ShortSentenceInterval:
    """Token interval that selects a short sentence resolve mode."""

    name: str
    max_token_length: int
    resolve_mode: ResolveModeName


@dataclass
class ShortSentenceApplication:
    """Result of applying a short sentence resolve mode."""

    phonemes: str
    tokens: list[int]
    metadata: dict[str, object] | None = None


@dataclass
class ShortSentenceTimingToken:
    """Serializable token metadata used to map timestamped model durations."""

    text: str
    phonemes: str
    whitespace: str
    is_target: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "phonemes": self.phonemes,
            "whitespace": self.whitespace,
            "is_target": self.is_target,
        }


@dataclass
class ShortSentenceConfig:
    """Configuration for short sentence handling using single-word context.

    Short, single-word sentences (< 10 phonemes, no spaces) often sound robotic
    when generated alone. This module improves quality by:
    1. Checking sentence is both short AND single-word (no spaces)
    2. Adding phoneme around word

    Multi-word sentences or sentences with breaks will NOT use this handler.

    Attributes:
        min_phoneme_length: Threshold below which sentences are considered "short"
            based on token count and will use context extraction. Default: 10.
        phoneme_pretext: Phoneme(s) to add before and after the target word
            when generating combined audio for context. Default: "—".
        enabled: Whether short sentence handling is enabled. Default: True.

    """

    min_phoneme_length: int = 5
    phoneme_pretext: str = "—"
    enabled: bool = True
    resolve_modes: dict[str, ShortSentenceResolveMode] = field(
        default_factory=lambda: {
            "wrap": WrapResolveMode(),
            "phrase": PhraseResolveMode(),
            "randomized-phrase": RandomizedPhraseResolveMode(),
        }
    )
    intervals: list[ShortSentenceInterval] = field(
        default_factory=lambda: [
            ShortSentenceInterval("single syllable", 5, "wrap"),
            ShortSentenceInterval("short", 10, False),
            ShortSentenceInterval("medium", 20, False),
        ]
    )

    def should_use_pause_surrounding(self, phoneme_length: int, text: str) -> bool:
        """Check if segment should use pause surrounding.

        Args:
            phoneme_length: Token count for the segment
            text: The text content to check for single-word status

        Returns:
            True if pause-surrounding should be applied
            (sentence is short AND single-word)
        """
        return self.get_resolve_mode_name(phoneme_length) is not False

    def get_resolve_mode_name(self, phoneme_length: int) -> ResolveModeName:
        """Return the configured resolve mode name for a token length."""
        if not self.enabled:
            return False

        for interval in self.intervals:
            if phoneme_length < interval.max_token_length:
                return interval.resolve_mode

        if phoneme_length < self.min_phoneme_length:
            return "wrap"
        return False

    def get_resolve_mode(self, phoneme_length: int) -> ShortSentenceResolveMode | None:
        """Return the configured resolve mode for a token length."""
        mode_name = self.get_resolve_mode_name(phoneme_length)
        if mode_name is False:
            return None
        return self.resolve_modes.get(mode_name)

    def contains_only_punctuation(self, phoneme: str) -> bool:
        """Check if segment contains only pounctions.

        Args:
            phoneme_length: Number of phonemes in the segment
            text: The text content to check for single-word status

        Returns:
            True if segment skipping should be applied
            (sentence is short AND single-word)
        """
        contains_only = ';:,.!?—…"()“” '

        return (
            self.enabled
            and len(phoneme) < self.min_phoneme_length
            and all(char in contains_only for char in phoneme)
        )


def is_segment_empty(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment contains only .

    Checks if segment is BOTH short (<10 phonemes) AND contains only pounctions.

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns:
        True if segment should be skipped
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False
    return config.contains_only_punctuation(segment.phonemes)


def is_segment_short(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment should use context-prepending.

    Checks if segment is BOTH short (<10 phonemes) AND single-word (no spaces).

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns
        True if segment should use pause-surrounding (short AND single-word)
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False

    token_length = len(segment.tokens) if segment.tokens else len(segment.phonemes)
    return config.should_use_pause_surrounding(token_length, segment.text)


def phonemize_short_sentence_phrase(
    segment: PhonemeSegment, phrase_template: str
) -> tuple[str, list[int], list[dict[str, object]]]:
    """Phonemize a phrase containing the short segment text."""
    import kokorog2p

    phrase_text = phrase_template.replace("{segment}", segment.text)
    segment_start = phrase_template.find("{segment}")
    segment_end = segment_start + len(segment.text) if segment_start >= 0 else -1
    lang = SUPPORTED_LANGUAGES.get(segment.lang, segment.lang)
    result = kokorog2p.phonemize(
        phrase_text,
        language=lang,
        return_phonemes=True,
        return_ids=True,
    )
    phonemes = getattr(result, "phonemes", None) or getattr(result, "phoneme", "")
    tokens = getattr(result, "ids", None) or getattr(result, "token_ids", [])
    timing_tokens = _build_timing_tokens(
        getattr(result, "tokens", []),
        segment_start=segment_start,
        segment_end=segment_end,
    )
    return str(phonemes), list(tokens), timing_tokens


def apply_short_sentence_mode(
    segment: PhonemeSegment,
    phonemes: str,
    tokens: list[int],
    config: ShortSentenceConfig,
    tokenize: Callable[[str], list[int]],
) -> ShortSentenceApplication:
    """Apply the configured short sentence resolve mode to a segment."""
    mode_name = config.get_resolve_mode_name(len(tokens))
    if mode_name is False:
        return ShortSentenceApplication(phonemes, tokens)

    mode = config.resolve_modes.get(mode_name)
    if mode is None:
        logger.warning("Unknown short sentence resolve mode '%s'", mode_name)
        return ShortSentenceApplication(phonemes, tokens)

    if mode.kind == "wrap":
        pretext = mode.phoneme_pretext
        if pretext == "—":
            pretext = config.phoneme_pretext
        wrapped = f"{pretext}{phonemes}{pretext}"
        return ShortSentenceApplication(wrapped, tokenize(wrapped))

    phrase_template = _select_phrase_template(segment.text, mode)
    try:
        phrase_result = phonemize_short_sentence_phrase(segment, phrase_template)
    except Exception as exc:
        logger.warning(
            "Failed to phonemize short sentence phrase for '%s': %s",
            segment.text[:50],
            exc,
        )
        return ShortSentenceApplication(phonemes, tokens)

    phrase_phonemes, phrase_tokens, timing_tokens = _coerce_phrase_result(phrase_result)
    if len(phrase_tokens) > MAX_PHONEME_LENGTH:
        logger.warning(
            "Short sentence phrase for '%s' exceeded max token length; "
            "using original segment",
            segment.text[:50],
        )
        return ShortSentenceApplication(phonemes, tokens)

    metadata = _build_short_sentence_metadata(
        mode_name=mode_name,
        mode=mode,
        original_token_count=len(tokens),
        generated_token_count=len(phrase_tokens),
        phrase_template=phrase_template,
        timing_tokens=timing_tokens,
    )
    return ShortSentenceApplication(phrase_phonemes, phrase_tokens, metadata)


def cut_short_sentence_phrase_audio(
    audio: np.ndarray, metadata: dict[str, object]
) -> np.ndarray:
    """Cut phrase-generated short sentence audio using timestamps when available."""
    kind = metadata.get("kind")
    if kind not in {"phrase", "randomized-phrase"}:
        return audio
    if audio.size == 0:
        return audio

    target_start_ts = metadata.get("target_start_ts")
    target_end_ts = metadata.get("target_end_ts")
    if isinstance(target_start_ts, (int, float)) and isinstance(
        target_end_ts, (int, float)
    ):
        target_start = max(0, int(float(target_start_ts) * SAMPLE_RATE))
        target_end = min(len(audio), int(float(target_end_ts) * SAMPLE_RATE))
        threshold = float(metadata.get("silence_threshold", 1e-4))
        start = _nearest_directional_quiet_sample(
            audio,
            target_start,
            direction="before",
            threshold=threshold,
        )
        end = _nearest_directional_quiet_sample(
            audio,
            target_end,
            direction="after",
            threshold=threshold,
        )
        if end > start:
            return audio[start:end]

    if kind == "randomized-phrase":
        logger.warning(
            "Randomized short sentence phrase has no usable target timestamps; "
            "leaving phrase audio uncut."
        )
        return audio

    expected_cut_ratio = float(metadata.get("expected_cut_ratio", 1.0))
    target = int(len(audio) * max(0.01, min(0.99, expected_cut_ratio)))
    silence_threshold = float(metadata.get("silence_threshold", 1e-4))
    min_silence_seconds = float(metadata.get("min_silence_seconds", 0.02))
    min_silence_samples = max(1, int(SAMPLE_RATE * min_silence_seconds))

    quiet = np.abs(audio) <= silence_threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for idx, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = idx
        elif not is_quiet and start is not None:
            if idx - start >= min_silence_samples:
                runs.append((start, idx))
            start = None

    if start is not None and len(audio) - start >= min_silence_samples:
        runs.append((start, len(audio)))

    if not runs:
        return audio[:target]

    cut_start, _ = min(runs, key=lambda run: abs(run[0] - target))
    return audio[: max(1, cut_start)]


def _nearest_directional_quiet_sample(
    audio: np.ndarray,
    anchor: int,
    *,
    direction: Literal["before", "after"],
    threshold: float,
) -> int:
    """Find the nearest near-zero sample on the semantically correct side."""
    if direction == "before":
        candidates = np.flatnonzero(np.abs(audio[: anchor + 1]) <= threshold)
        return int(candidates[-1]) if candidates.size else anchor

    candidates = np.flatnonzero(np.abs(audio[anchor:]) <= threshold)
    return anchor + int(candidates[0]) if candidates.size else anchor


def _select_phrase_template(segment_text: str, mode: ShortSentenceResolveMode) -> str:
    use_end_phrase = segment_text.rstrip().endswith(".")
    if isinstance(mode, RandomizedPhraseResolveMode):
        if use_end_phrase:
            choices = mode.end_phrases or ["The word is hello. The word is '{segment}'"]
        else:
            choices = mode.neutral_phrases or ["The word, {segment}, appears here."]
        return random.choice(choices)
    if isinstance(mode, PhraseResolveMode):
        return mode.end_phrase if use_end_phrase else mode.neutral_phrase
    return ""


def _build_short_sentence_metadata(
    *,
    mode_name: str,
    mode: ShortSentenceResolveMode,
    original_token_count: int,
    generated_token_count: int,
    phrase_template: str | None = None,
    timing_tokens: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    expected_cut_ratio = 1.0
    if generated_token_count > 0:
        expected_cut_ratio = original_token_count / generated_token_count
    metadata: dict[str, object] = {
        "mode": mode_name,
        "kind": mode.kind,
        "phrase_template": phrase_template,
        "original_token_count": original_token_count,
        "generated_token_count": generated_token_count,
        "expected_cut_ratio": max(0.01, min(0.99, expected_cut_ratio)),
        "silence_threshold": mode.silence_threshold,
        "min_silence_seconds": mode.min_silence_seconds,
    }
    if timing_tokens:
        metadata["timing_tokens"] = timing_tokens
    return metadata


def _coerce_phrase_result(
    phrase_result: tuple[str, list[int]] | tuple[str, list[int], list[dict[str, object]]],
) -> tuple[str, list[int], list[dict[str, object]]]:
    """Accept legacy two-item monkeypatched test tuples and new timing tuples."""
    if len(phrase_result) == 2:
        phrase_phonemes, phrase_tokens = phrase_result
        return phrase_phonemes, phrase_tokens, []
    phrase_phonemes, phrase_tokens, timing_tokens = phrase_result
    return phrase_phonemes, phrase_tokens, timing_tokens


def _build_timing_tokens(
    tokens: object,
    *,
    segment_start: int,
    segment_end: int,
) -> list[dict[str, object]]:
    timing_tokens: list[dict[str, object]] = []
    for token in tokens or []:
        phonemes = _token_attr(token, "phonemes") or _token_attr(token, "phoneme") or ""
        text = str(_token_attr(token, "text") or "")
        whitespace = str(_token_attr(token, "whitespace") or "")
        char_start = _token_attr(token, "char_start")
        char_end = _token_attr(token, "char_end")
        is_target = _token_overlaps_segment(
            char_start,
            char_end,
            segment_start=segment_start,
            segment_end=segment_end,
        )
        timing_tokens.append(
            ShortSentenceTimingToken(
                text=text,
                phonemes=str(phonemes),
                whitespace=whitespace,
                is_target=is_target,
            ).to_dict()
        )
    return timing_tokens


def _token_attr(token: object, name: str) -> object:
    value = getattr(token, name, None)
    if value is not None:
        return value
    meta = getattr(token, "meta", None)
    if isinstance(meta, dict) and name in meta:
        return meta[name]
    get = getattr(token, "get", None)
    if callable(get):
        return get(name)
    return None


def _token_overlaps_segment(
    char_start: object,
    char_end: object,
    *,
    segment_start: int,
    segment_end: int,
) -> bool:
    if segment_start < 0 or segment_end < 0:
        return False
    if not isinstance(char_start, int) or not isinstance(char_end, int):
        return False
    return char_start < segment_end and char_end > segment_start

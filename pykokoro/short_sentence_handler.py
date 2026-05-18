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

from .constants import MAX_PHONEME_LENGTH, SUPPORTED_LANGUAGES
from .short_sentence_cutters import cut_phrase_audio

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
    neutral_phrase: str = "She looked up…: {segment}. The conversation resumed."
    end_phrase: str = "The teacher waited for a response. {segment}"
    frame_duration_ms: int = 5
    energy_threshold: float = 0.05
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02
    cutter: Literal["vad", "energy-valley"] = "vad"


@dataclass
class RandomizedPhraseResolveMode:
    """Configuration for randomized phrase generation and cutting."""

    kind: Literal["randomized-phrase"] = "randomized-phrase"
    neutral_phrases: list[str] = field(
        default_factory=lambda: [
            "She looked up…: {segment}. The conversation resumed.",
            "The transcript paused…: {segment}; the next entry followed.",
            "The hallway went quiet; {segment}; then footsteps resumed.",
            "The clerk paused, {segment}, before the next name was called.",
        ]
    )
    end_phrases: list[str] = field(
        default_factory=lambda: [
            "The recording trails off after the words, … {segment}",
            "Is that … {segment}?",
            "The transcript closes with the words, {segment}",
            "The lesson ended when the teacher asked, {segment}",
            "The letter closed with this unfinished thought — {segment}",
            "The report concludes with this note: {segment}",
            "The teacher waited for a response. {segment}",
            "The announcement ended like this: {segment}",
            "There was a pause before the answer came: {segment}",
            "The host asked again, more quietly this time: {segment}",
            "The conversation stopped after one last reply: {segment}",
        ]
    )
    frame_duration_ms: int = 5
    energy_threshold: float = 0.05
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02
    cutter: Literal["vad", "energy-valley"] = "vad"


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

    fallback_phonemes = _wrap_phonemes(phonemes, config)
    metadata = _build_short_sentence_metadata(
        mode_name=mode_name,
        mode=mode,
        original_token_count=len(tokens),
        generated_token_count=len(phrase_tokens),
        phrase_template=phrase_template,
        timing_tokens=timing_tokens,
        fallback_phonemes=fallback_phonemes,
        fallback_tokens=tokenize(fallback_phonemes),
    )
    return ShortSentenceApplication(phrase_phonemes, phrase_tokens, metadata)


def cut_short_sentence_phrase_audio(
    audio: np.ndarray, metadata: dict[str, object]
) -> np.ndarray | None:
    """Cut phrase-generated short sentence audio using the configured cutter."""
    kind = metadata.get("kind")
    if kind not in {"phrase", "randomized-phrase"}:
        return audio
    if audio.size == 0:
        return audio
    return cut_phrase_audio(audio, metadata)


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
    fallback_phonemes: str | None = None,
    fallback_tokens: list[int] | None = None,
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
        "frame_duration_ms": mode.frame_duration_ms,
        "energy_threshold": mode.energy_threshold,
        "silence_threshold": mode.silence_threshold,
        "min_silence_seconds": mode.min_silence_seconds,
        "cutter": mode.cutter,
    }
    if timing_tokens:
        metadata["timing_tokens"] = timing_tokens
    if fallback_phonemes is not None:
        metadata["fallback_phonemes"] = fallback_phonemes
    if fallback_tokens is not None:
        metadata["fallback_tokens"] = fallback_tokens
    return metadata


def _wrap_phonemes(phonemes: str, config: ShortSentenceConfig) -> str:
    wrap_mode = config.resolve_modes.get("wrap")
    pretext = wrap_mode.phoneme_pretext if isinstance(wrap_mode, WrapResolveMode) else "â€”"
    if pretext == "â€”":
        pretext = config.phoneme_pretext
    return f"{pretext}{phonemes}{pretext}"


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

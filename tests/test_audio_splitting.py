from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

import pykokoro.short_sentence_handler as short_sentence_handler
from pykokoro.audio_generator import AudioGenerator
from pykokoro.constants import MAX_PHONEME_LENGTH
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    RandomizedPhraseResolveMode,
    SHORT_SENTENCE_META_KEY,
    ShortSentenceConfig,
    ShortSentenceInterval,
)
from pykokoro.types import PhonemeSegment


class DummyTokenizer:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def tokenize(self, text: str):
        return list(range(len(text) * self.factor))

    def detokenize(self, tokens):
        if not tokens:
            return ""
        return "a" * max(1, len(tokens) // self.factor)


class DummySession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids")]

    def get_outputs(self):
        return [SimpleNamespace(name="waveform")]


class TimestampSession(DummySession):
    def get_outputs(self):
        return [SimpleNamespace(name="waveform"), SimpleNamespace(name="pred_dur")]

    def run(self, output_names, inputs):
        _ = output_names, inputs
        audio = np.zeros(8, dtype=np.float32)
        pred_dur = np.array([3, 1, 1, 1, 2, 2, 0], dtype=np.float32)
        return [audio, pred_dur]


def test_split_phonemes_uses_token_count():
    tokenizer = DummyTokenizer(factor=300)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )

    batches = generator.split_phonemes("hi")

    assert len(batches) > 1


def test_preprocess_pipeline_splits_segments():
    tokenizer = DummyTokenizer(factor=MAX_PHONEME_LENGTH)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="hello",
        phonemes="aa",
        tokens=[],
        pause_before=0.3,
        pause_after=0.7,
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert len(processed) == 2
    assert processed[0].id == "seg_1_ph0"
    assert processed[1].id == "seg_1_ph1"
    assert processed[0].pause_before == 0.3
    assert processed[0].pause_after == 0.0
    assert processed[1].pause_before == 0.0
    assert processed[1].pause_after == 0.7
    assert len(processed[0].tokens) == MAX_PHONEME_LENGTH
    assert len(processed[1].tokens) == MAX_PHONEME_LENGTH


def test_preprocess_wraps_short_segments_when_enabled():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
        pause_before=0.3,
        pause_after=0.7,
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert len(processed) == 1
    pretext = ShortSentenceConfig().phoneme_pretext
    assert processed[0].phonemes == f"{pretext}abc{pretext}"
    assert len(processed[0].tokens) == len(f"{pretext}abc{pretext}")
    assert processed[0].pause_before == 0.3
    assert processed[0].pause_after == 0.7


def test_preprocess_does_not_wrap_short_segments_when_disabled():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert processed[0].phonemes == "abc"
    assert len(processed[0].tokens) == len("abc")


def test_preprocess_uses_custom_short_sentence_pretext():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=ShortSentenceConfig(phoneme_pretext="..."),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "...abc..."


def test_preprocess_does_not_wrap_long_segments():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Hello",
        phonemes="abcde",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "abcde"


def test_preprocess_punctuation_only_segments_follow_short_sentence_override():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="!",
        phonemes="?!",
        tokens=[],
    )

    enabled = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )
    disabled = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert enabled[0].phonemes == ""
    assert enabled[0].tokens == []
    assert disabled[0].phonemes == "?!"
    assert len(disabled[0].tokens) == len("?!")


def test_preprocess_phrase_mode_uses_phrase_phonemes_and_metadata(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase="The word, {segment}, appears here.",
                end_phrase="The word is hello. The word is '{segment}'",
            )
        },
        intervals=[ShortSentenceInterval("single syllable", 5, "phrase")],
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == "The word, {segment}, appears here."
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "abc def."
    assert processed[0].tokens == list(range(8))
    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["mode"] == "phrase"
    assert metadata["kind"] == "phrase"
    assert metadata["phrase_template"] == "The word, {segment}, appears here."
    assert metadata["original_token_count"] == 3
    assert metadata["generated_token_count"] == 8


def test_preprocess_randomized_phrase_mode_uses_configured_phrases(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "randomized-phrase": RandomizedPhraseResolveMode(
                neutral_phrases=["The word, {segment}, appears here."],
                end_phrases=["The word is hello. The word is '{segment}'"],
            )
        },
        intervals=[ShortSentenceInterval("single syllable", 5, "randomized-phrase")],
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert phrase_template == "The word, {segment}, appears here."
        return "abc option.", list(range(10))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "abc option."
    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["mode"] == "randomized-phrase"
    assert metadata["phrase_template"] == "The word, {segment}, appears here."


def test_preprocess_phrase_mode_uses_end_phrase_for_period(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase="The word, {segment}, appears here.",
                end_phrase="The word is hello. The word is '{segment}'",
            )
        },
        intervals=[ShortSentenceInterval("single syllable", 5, "phrase")],
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go."
        assert phrase_template == "The word is hello. The word is '{segment}'"
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go.",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_template"] == "The word is hello. The word is '{segment}'"


def test_postprocess_phrase_mode_cuts_at_quiet_run_near_expected_ratio():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(10, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            np.ones(30, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "expected_cut_ratio": 10 / 45,
                "silence_threshold": 1e-4,
                "min_silence_seconds": 0.0,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert len(processed[0].processed_audio) == 10


def test_phrase_modes_fall_back_to_wrap_once_without_timestamp_output(
    monkeypatch, capsys
):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={"phrase": PhraseResolveMode()},
        intervals=[ShortSentenceInterval("single syllable", 5, "phrase")],
    )

    def unexpected_phrase(*args, **kwargs):
        raise AssertionError("phrase mode should not run without timestamp output")

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        unexpected_phrase,
    )
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segments = [
        PhonemeSegment(
            id=f"seg_{idx}",
            segment_id=f"seg_{idx}",
            phoneme_id=0,
            text="Go",
            phonemes="abc",
            tokens=[],
        )
        for idx in range(2)
    ]

    processed = generator._preprocess_segments(
        segments,
        enable_short_sentence_override=True,
    )

    pretext = ShortSentenceConfig().phoneme_pretext
    assert [segment.phonemes for segment in processed] == [
        f"{pretext}abc{pretext}",
        f"{pretext}abc{pretext}",
    ]
    assert all(segment.ssmd_metadata is None for segment in processed)
    captured = capsys.readouterr()
    assert captured.out.count("Falling back to wrap mode for this run.") == 1


def test_postprocess_randomized_phrase_mode_cuts_from_target_timestamps():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.arange(100, dtype=np.float32)
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "randomized-phrase",
                "target_start_ts": 10 / 24000,
                "target_end_ts": 30 / 24000,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert np.array_equal(processed[0].processed_audio, audio[10:30])


def test_generate_logs_randomized_phrase_target_timestamps(capsys):
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="thgo",
        tokens=[1, 2, 3, 4],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "randomized-phrase",
                "timing_tokens": [
                    {
                        "text": "The",
                        "phonemes": "th",
                        "whitespace": " ",
                        "is_target": False,
                    },
                    {
                        "text": "Go",
                        "phonemes": "go",
                        "whitespace": "",
                        "is_target": True,
                    },
                ],
            }
        },
    )

    generator._generate_raw_audio_segments(
        [segment],
        voice_style=np.zeros((16, 256), dtype=np.float32),
        speed=1.0,
        voice_resolver=None,
    )

    captured = capsys.readouterr()
    assert "segment='Go'" in captured.out
    assert "token='Go'" in captured.out
    assert "start=" in captured.out
    assert "end=" in captured.out
    metadata = segment.ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["target_start_ts"] == 0.0625
    assert metadata["target_end_ts"] == 0.175

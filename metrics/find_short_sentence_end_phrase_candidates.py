#!/usr/bin/env python3
"""Rank carrier phrases that place inserted short text at the end."""

from __future__ import annotations

import find_short_sentence_phrase_candidates as phrase_candidates

PHRASE_CANDIDATES = [
    "When the room finally settled, someone said, {segment}",
    "The note ended with one short line: {segment}",
    "The message was brief — just {segment}",
    "The final entry in the log says: {segment}",
    "The transcript closes with the words, {segment}",
    "The final line of the letter was, {segment}",
    "The announcement ended like this: {segment}",
    "At last, the guide called out, {segment}",
    "The page ended with a single word: {segment}",
    "The message on the screen changed to {segment}",
    "The room went still before she answered, {segment}",
    "The teacher waited for a response. {segment}",
    "The report concludes with this note: {segment}",
    "The host asked again, more quietly this time: {segment}",
    "The final caption on the screen read, {segment}",
    "The narrator slowed down, then said, {segment}",
    "There was a pause before the answer came: {segment}",
    "The last thing she heard was, {segment}",
    "The student stared at the page and finally said, {segment}",
    "The line in the script ends with, {segment}",
    "The announcer let the moment hang, then said, {segment}",
    "The note on the desk simply said, {segment}",
    "The interview closed on one short answer: {segment}",
    "The final sentence in the memo was: {segment}",
    "The voice on the radio faded into a final word: {segment}",
    "The last audible phrase is … {segment}",
    "The final prompt on the form reads: {segment}",
    "The lesson ended when the teacher asked, {segment}",
    "The conversation stopped after one last reply: {segment}",
    "The letter closed with this unfinished thought — {segment}",
    "The judge asked for a final answer. {segment}",
    "The recording trails off after the words, … {segment}",
    "The final word is hello. The final word is '{segment}'",
    "The final word is hello. The final word is … '{segment}'",
    "The caller left one final message…: {segment}"
]


def main() -> None:
    phrase_candidates.PHRASE_CANDIDATES = PHRASE_CANDIDATES
    phrase_candidates.REQUIRE_AFTER_BOUNDARY = False
    phrase_candidates.main()


if __name__ == "__main__":
    main()

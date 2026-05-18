#!/usr/bin/env python3
"""Rank carrier phrases that place inserted short text at the end."""

from __future__ import annotations

import find_short_sentence_phrase_candidates as phrase_candidates

PHRASE_CANDIDATES = [
    "He paused, then said: {segment}",
    "She looked up and answered, {segment}",
    "After a long silence, he whispered, {segment}",
    "The last words on the recording were {segment}",
    "When the room finally settled, someone said, {segment}",
    "The note ended with one short line: {segment}",
    "At the bottom of the page, it read, {segment}",
    "The operator waited, then replied: {segment}",
    "The judge asked for a final answer. {segment}",
    "The child took a breath before saying, {segment}",
    "The speaker stopped, thought for a moment, and added, {segment}",
    "The message was brief — just {segment}",
    "The final entry in the log says: {segment}",
    "No one spoke until the caller said, {segment}",
    "The sign above the door read, {segment}",
    "The transcript closes with the words, {segment}",
    "The actor crossed the stage and murmured, {segment}",
    "After the music faded, the singer said, {segment}",
    "The answer came after a pause: {segment}",
    "The question was simple. The reply was {segment}",
    "The final line of the letter was, {segment}",
    "The announcement ended like this: {segment}",
    "The witness hesitated … then answered, {segment}",
    "At last, the guide called out, {segment}",
    "The page ended with a single word: {segment}",
    "The message on the screen changed to {segment}",
    "He lowered his voice and said, {segment}",
    "The caller left one final message: {segment}",
    "The room went still before she answered, {segment}",
    "The teacher waited for a response. {segment}",
    "The report concludes with this note: {segment}",
    "The recording trails off after the words, {segment}",
    "The host asked again, more quietly this time: {segment}",
    "The clerk checked the list and called, {segment}",
    "The only reply from the hallway was {segment}",
    "The final caption on the screen read, {segment}",
    "The narrator slowed down, then said, {segment}",
    "There was a pause before the answer came: {segment}",
    "The last thing she heard was, {segment}",
    "The student stared at the page and finally said, {segment}",
    "The line in the script ends with, {segment}",
    "The announcer let the moment hang, then said, {segment}",
    "The note on the desk simply said, {segment}",
    "The interview closed on one short answer: {segment}",
    "The captain gave the order — {segment}",
    "The final sentence in the memo was: {segment}",
    "The doctor asked once more, and the patient answered, {segment}",
    "The voice on the radio faded into a final word: {segment}",
    "The crowd quieted when the speaker said, {segment}",
    "The chapter ends with the question, {segment}",
    "The last audible phrase is … {segment}",
    "Before the call disconnected, he said, {segment}",
    "The final prompt on the form reads: {segment}",
    "The lesson ended when the teacher asked, {segment}",
    "The conversation stopped after one last reply: {segment}",
    "The letter closed with this unfinished thought — {segment}",
]


def main() -> None:
    phrase_candidates.PHRASE_CANDIDATES = PHRASE_CANDIDATES
    phrase_candidates.main()


if __name__ == "__main__":
    main()

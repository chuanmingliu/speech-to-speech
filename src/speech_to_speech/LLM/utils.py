import base64
import io
import re
from typing import Optional

import requests  # type: ignore[import-untyped]
from nltk import sent_tokenize
from PIL import Image

SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

SPEECHABLE_PATTERN = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–\n\r\t。！？；、，：]",
    flags=re.UNICODE,
)


def remove_unspeechable(text: str) -> str:
    """Keep only speechable characters: letters, digits, punctuation, whitespace.
    support unicode characters (english, arabic, chinese, japanese, korean, etc.)
    """
    text = text.translate(SMART_PUNCT_TRANSLATION)
    return SPEECHABLE_PATTERN.sub("", text)


# nltk.sent_tokenize (punkt) does not treat CJK full stops as sentence
# boundaries, so a Chinese reply would otherwise sit in the LLM buffer until
# the stream ended. Flush as soon as a spoken terminator lands.
_SENTENCE_END_CHARS = frozenset(".!?。！？；…")
_CJK_LOOKBEHIND_SPLIT = re.compile(r"(?<=[。！？；…])")


def split_spoken_sentences(text: str) -> list[str]:
    """Split spoken text into sentences, including a possibly incomplete tail.

    CJK spans are split on ``。！？；…``. Latin spans still go through
    ``nltk.sent_tokenize`` so abbreviations and decimals stay intact.
    """
    if not text:
        return []

    sentences: list[str] = []
    for part in _CJK_LOOKBEHIND_SPLIT.split(text):
        if not part:
            continue
        latin = sent_tokenize(part)
        sentences.extend(latin if latin else [part])
    return sentences


# Clause punctuation is not a sentence boundary, but it *is* a safe place to
# hand a prefix to the TTS engine. Waiting for a full sentence before the first
# synthesis request costs the whole "…, " prefix in tokens, which is dead time
# on the end-to-end path: the listener hears nothing while the model is still
# writing a clause the engine could already be speaking.
_CLAUSE_BREAK_CHARS = frozenset(",;:，、；：")
_SPOKEN_BREAK_CHARS = _CLAUSE_BREAK_CHARS | _SENTENCE_END_CHARS
# A one-character opening unit ("，") carries no speech; never flush one.
_MIN_FIRST_UNIT_CHARS = 2


def _is_spoken_break(text: str, index: int) -> bool:
    """Whether ``text[index]`` ends a clause that is safe to speak on its own."""
    char = text[index]
    if char not in _SPOKEN_BREAK_CHARS:
        return False
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    # "1,000", "3.14", "12:30" — punctuation inside a number is not a pause. A
    # digit directly before an unterminated buffer tail is treated the same way,
    # because the next token may well continue the number.
    if previous.isdigit() and (following.isdigit() or not following):
        return False
    # ASCII punctuation only separates clauses when whitespace follows it, so
    # "e.g." and "gpt-5.4-mini" stay intact. A break at the very end of the
    # buffer is genuine: the model emitted the punctuation and nothing after it.
    if char.isascii() and following and not following.isspace():
        return False
    return True


def split_first_spoken_unit(text: str, lookahead_chars: int) -> tuple[str, str]:
    """Return ``(first_unit, remainder)`` for the opening clause of a reply.

    Unlike :func:`split_spoken_units` this breaks on clause punctuation too, so
    the first TTS request goes out before the sentence is finished. The split
    only happens once ``lookahead_chars`` characters are buffered, which bounds
    -- but does not eliminate -- the risk that playback runs dry: a short
    opening clause ("Sure,") can still finish speaking before the rest of the
    sentence has been generated and synthesised. That only bites when the model
    streams slowly; ``scripts/latency_ab_benchmark.py`` measures it, and below
    roughly 30 tok/s the gap becomes audible for short English clauses.

    Note this bounds the *buffer*, not the returned prefix -- gating on prefix
    length instead would skip the early comma that makes this worth doing.

    ``("", text)`` means nothing is ready yet and the caller should keep
    buffering.
    """
    if lookahead_chars <= 0 or len(text.strip()) < lookahead_chars:
        return "", text
    for index in range(len(text)):
        if not _is_spoken_break(text, index):
            continue
        head = text[: index + 1]
        if len(head.strip()) >= _MIN_FIRST_UNIT_CHARS:
            return head, text[index + 1 :]
    return "", text


def split_spoken_units(text: str) -> tuple[list[str], str]:
    """Return ``(complete_sentences, remainder)``.

    A sentence is complete when it ends with a spoken terminator. The remainder
    is the unfinished tail that must stay buffered for the next token.
    """
    sentences = split_spoken_sentences(text)
    if not sentences:
        return [], ""
    last = sentences[-1]
    stripped = last.rstrip()
    if stripped and stripped[-1] in _SENTENCE_END_CHARS:
        return sentences, ""
    return sentences[:-1], last


WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}


def resolve_auto_language(language_code: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Strip the ``-auto`` suffix and resolve the human-readable language name.

    Returns ``(clean_code, language_name)``.  ``language_name`` is non-None
    when the code (with or without ``-auto``) maps to a known language.
    """
    if not language_code:
        return language_code, None
    if language_code.endswith("-auto"):
        language_code = language_code[:-5]
    if language_code not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
        return language_code, None
    return language_code, WHISPER_LANGUAGE_TO_LLM_LANGUAGE.get(language_code)


def image_url_to_pil(image_url: str) -> Image.Image:
    """Convert an image URL or base64 data URI to a PIL Image.

    Accepts:
    - 'data:image/...;base64,<b64>' data URIs
    - 'https://...`` or ``http://...' URLs (fetched with a 10s timeout)
    """
    if image_url.startswith("data:"):
        _, b64_data = image_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64_data)))
    resp = requests.get(image_url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

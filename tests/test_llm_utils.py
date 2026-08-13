import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")

from speech_to_speech.LLM.utils import remove_unspeechable, split_spoken_sentences, split_spoken_units


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_remove_unspeechable_keeps_cjk_sentence_punctuation() -> None:
    assert remove_unspeechable("你好。今天天气不错！") == "你好。今天天气不错！"


def test_split_spoken_sentences_splits_chinese_terminators() -> None:
    assert split_spoken_sentences("你好。今天天气不错。") == ["你好。", "今天天气不错。"]


def test_split_spoken_units_flushes_complete_chinese_sentence() -> None:
    assert split_spoken_units("你好。") == (["你好。"], "")
    assert split_spoken_units("你好。我是") == (["你好。"], "我是")


def test_split_spoken_units_keeps_latin_decimals_together() -> None:
    complete, remainder = split_spoken_units("The value is 3.14 now")
    assert complete == []
    assert remainder == "The value is 3.14 now"


def test_split_spoken_units_splits_english_sentences() -> None:
    complete, remainder = split_spoken_units("Hello. How are you?")
    assert complete == ["Hello.", "How are you?"]
    assert remainder == ""


def test_split_spoken_units_mixed_chinese_and_english() -> None:
    complete, remainder = split_spoken_units("你好。Hello there. Still going")
    assert complete == ["你好。", "Hello there."]
    assert remainder == "Still going"

from dataclasses import dataclass, field


@dataclass
class LanguageModelBaseArguments:
    model_name: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "The pretrained language model to use."},
    )
    user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in the chat context. Default is 'user'."},
    )
    init_chat_role: str = field(
        default="system",
        metadata={"help": "Initial role for setting up the chat context. Default is 'system'."},
    )
    init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={"help": "The initial chat prompt to establish context for the language model."},
    )
    chat_size: int = field(
        default=30,
        metadata={"help": "Number of interactions assistant-user to keep for the chat."},
    )
    stream_batch_sentences: int = field(
        default=3,
        metadata={
            "help": "Number of sentences to accumulate before yielding a batch during streaming. "
            "Set to 1 for sentence-by-sentence streaming. Default is 3."
        },
    )
    stream_first_chunk_lookahead_chars: int = field(
        default=8,
        metadata={
            "help": "Buffer this many characters past the reply's opening clause, then flush that clause to TTS on a "
            "clause boundary (',' ';' ':' and their CJK forms) instead of waiting for the sentence to terminate. "
            "Synthesis then starts while the model is still writing the first sentence, which is the largest single "
            "component of speech-stop-to-first-audio. Set to 0 to keep the sentence-only behaviour. Default is 8."
        },
    )
    request_hedge_after_ms: float = field(
        default=0.0,
        metadata={
            "help": "When greater than zero, issue a second identical completion if the first has produced no token "
            "within this many milliseconds, and stream whichever answers first. Trades a duplicate request on the "
            "slow tail of turns for a much shorter p95/p99 time-to-first-audio. Set to 0 to disable. Default is 0."
        },
    )
    enable_lang_prompt: bool = field(
        default=False,
        metadata={
            "help": "When True, append a user message instructing the model to reply in the detected/selected "
            "language (e.g. 'Please reply to my message in French.'). Default is False."
        },
    )
    compact_history: bool = field(
        default=True,
        metadata={
            "help": "When True, summarize older turns in the background once the chat exceeds chat_size, "
            "instead of synchronously evicting them. Adds an extra LLM call per compaction. Default is True."
        },
    )

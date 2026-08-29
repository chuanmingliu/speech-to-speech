from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniMaxTTSHandlerArguments:
    minimax_tts_prime_texts: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pipe-separated opening clauses to pre-synthesise into the TTS cache at startup, e.g. "
            "'好的，|没问题，|Sure,'. With the clause-early first flush these are exactly the chunks a reply "
            "opens with, so a hit removes MiniMax's first-byte time (~200ms) from the front of the turn. Each "
            "entry costs one billable synthesis at startup and must match the flushed chunk exactly, punctuation "
            "included. When unset, MINIMAX_TTS_PRIME_TEXTS is used; empty disables priming."
        },
    )
    minimax_tts_speed: Optional[float] = field(
        default=None,
        metadata={
            "help": "MiniMax speech speed multiplier in the provider-supported range 0.5 to 2.0. "
            "When unset, MINIMAX_TTS_SPEED or 1.0 is used."
        },
    )

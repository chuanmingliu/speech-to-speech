from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniMaxTTSHandlerArguments:
    minimax_tts_speed: Optional[float] = field(
        default=None,
        metadata={
            "help": "MiniMax speech speed multiplier in the provider-supported range 0.5 to 2.0. "
            "When unset, MINIMAX_TTS_SPEED or 1.0 is used."
        },
    )

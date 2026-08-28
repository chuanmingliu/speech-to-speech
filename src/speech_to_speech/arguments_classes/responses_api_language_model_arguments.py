from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ResponsesApiLanguageModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="gpt-5.4-mini",
        metadata={"help": "The model to use with the OpenAI-compatible API. Default is 'gpt-5.4-mini'."},
    )
    responses_api_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key used to authenticate access to the OpenAI-compatible API. Default is None."},
    )
    responses_api_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the OpenAI-compatible API endpoint. Default is None (uses OpenAI)."},
    )
    responses_api_stream: bool = field(
        default=True,
        metadata={
            "help": "The stream parameter typically indicates whether data should be transmitted in a continuous flow rather"
            " than in a single, complete response, often used for handling large or real-time data.Default is True"
        },
    )
    responses_api_disable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Disable provider-side thinking/reasoning when supported by the OpenAI-compatible backend. "
            "Uses thinking.type=disabled for DeepSeek and chat_template_kwargs.enable_thinking=false for Qwen."
        },
    )
    responses_api_connection_keepalive_s: float = field(
        default=300.0,
        metadata={
            "help": "How long idle hosted-LLM HTTP connections remain reusable. The OpenAI SDK default is only five "
            "seconds, which often makes the first spoken turn repeat DNS/TCP/TLS setup. Default is 300 seconds."
        },
    )
    responses_api_prewarm_wait_s: float = field(
        default=0.5,
        metadata={
            "help": "Maximum time a model request waits for an in-progress connection refresh. This prevents an "
            "immediate telephony greeting from racing the refresh and opening another cold connection. Default is 0.5."
        },
    )

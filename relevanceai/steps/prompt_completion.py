from pydantic import Field

from typing import Literal, List, Any
from relevanceai.steps.base import Step


class PromptCompletion(Step):
    transformation: str = "prompt_completion"
    prompt: str = Field(...)
    model: Literal[
        "openai-gpt35",
        "openai-gpt4",
        "anthropic-claude-instant-v1",
        "anthropic-claude-v1",
        "anthropic-claude-instant-v1-100k",
        "anthropic-claude-v1-100k",
        "palm-chat-bison",
        "palm-text-bison",
        "cohere-command-light",
        "cohere-command",
    ] = Field("openai-gpt35")
    history: List[Any] = Field(None)
    system_prompt: str = Field(None)
    strip_linebreaks: bool = Field(None)
    temperature: float = Field(None)
    validators: List[Any] = Field(None)

    @property
    def output_spec(self):
        return ["answer", "prompt", "user_key_used", "validation_history"]

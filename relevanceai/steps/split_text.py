from pydantic import Field

from typing import Literal
from relevanceai.steps.base import Step


class SplitText(Step):
    transformation: str = "split_text"
    text: str = Field(...)
    method: Literal["tokens", "separator"] = Field(...)
    num_tokens: int = Field(None)
    num_tokens_to_slide_window: int = Field(None)
    sep: str = Field(None)

    @property
    def output_spec(self):
        return ["chunks"]

from pydantic import Field

from relevanceai.steps.base import Step


class ToJson(Step):
    transformation: str = "to_json"
    text: str = Field(...)

    @property
    def output_spec(self):
        return ["output"]

from pydantic import Field

from relevanceai.steps.base import Step


class ExecuteJavascriptCode(Step):
    transformation: str = "js_code_transformation"
    code: str = Field(...)

    @property
    def output_spec(self):
        return ["transformed", "duration", "stdout", "stderr"]

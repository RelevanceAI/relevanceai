from pydantic import Field

from relevanceai.types import JSONDict
from relevanceai.steps.base import Step


class RunChain(Step):
    transformation: str = "run_chain"
    chain_id: str = Field(...)
    params: JSONDict = Field(...)

    @property
    def output_spec(self):
        return [
            "output",
            "state",
            "status",
            "errors",
            "cost",
            "credits_used",
            "executionTime",
        ]

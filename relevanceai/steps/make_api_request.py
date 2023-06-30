from pydantic import Field

from typing import Literal
from relevanceai.steps.base import Step
from relevanceai.types import JSONDict, JSONObject


class MakeAPIRequest(Step):
    transformation: str = "api_call"
    url: str = Field(...)
    method: Literal[
        "GET", "POST", "HEAD", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"
    ] = Field(...)
    headers: JSONDict = Field(None)
    body: JSONObject = Field(None)
    response_type: str = Field(None)

    @property
    def output_spec(self):
        return ["response_body", "status"]

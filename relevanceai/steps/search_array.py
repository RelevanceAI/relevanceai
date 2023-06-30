from pydantic import Field

from relevanceai.steps.base import Step


class SearchArray(Step):
    transformation: str = "search_array"
    array: str = Field(...)
    query: str = Field(...)
    page_size: int = Field(5)
    field: str = Field(None)

    @property
    def output_spec(self):
        return ["results"]

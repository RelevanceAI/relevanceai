from pydantic import Field
from typing import Any, List, Dict
from relevanceai.steps.base import Step


class BrowserlessScrape(Step):
    transformation: str = "browserless_scrape"
    website_url: str = Field(...)
    element_selector: List[Any] = Field(None)
    extra_headers: Dict[str, Any] = Field(None)

    @property
    def output_spec(self):
        return ["output"]

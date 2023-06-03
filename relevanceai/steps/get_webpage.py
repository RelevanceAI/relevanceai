from relevanceai.steps._base import StepBase


class GetWebpage(StepBase):
    def __init__(
        self, website_url: str, step_name: str = "get_webpage", *args, **kwargs
    ):
        self.website_url = website_url
        self.step_name = step_name
        self._outputs = ["contents"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        return [
            {
                "transformation": "get_webpage",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": {"website_url": self.website_url},
            }
        ]

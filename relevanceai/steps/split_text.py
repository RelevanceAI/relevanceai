from relevanceai.steps._base import StepBase


class SplitText(StepBase):
    def __init__(
        self,
        text: str,
        sep: str,
        method: str = "tokens",
        num_tokens: int = 500,
        num_tokens_to_slide_window: int = 1,
        step_name: str = "split_text",
        *args,
        **kwargs,
    ) -> None:
        self.text = text
        self.sep = sep
        self.method = method
        self.num_tokens = num_tokens
        self.num_tokens_to_slide_window = num_tokens_to_slide_window
        self.step_name = step_name
        self._outputs = ["chunks"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        step_params = {
            "text": self.text,
            "method": self.method,
            "num_tokens": self.num_tokens,
            "num_tokens_to_slide_window": self.num_tokens_to_slide_window,
        }
        if self.sep is not None:
            step_params["sep"] = self.sep
        return [
            {
                "transformation": "split_text",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": step_params,
            }
        ]

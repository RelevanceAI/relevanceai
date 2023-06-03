from relevanceai.steps._base import StepBase


class PDFToText(StepBase):
    def __init__(
        self,
        pdf_url: str,
        use_ocr: bool = False,
        step_name: str = "pdf_to_text",
        *args,
        **kwargs,
    ) -> None:
        self.pdf_url = pdf_url
        self.use_ocr = use_ocr
        self.step_name = step_name
        self._outputs = ["text", "number_of_pages"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        step_params = {
            "pdf_url": self.pdf_url,
            "use_ocr": self.use_ocr,
        }
        return [
            {
                "transformation": "pdf_to_text",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": step_params,
            }
        ]

from relevanceai.steps._base import StepBase

class ExecuteJavascriptCode(StepBase):
    """Execute Javascript code
    Run JavaScript code to transform any inputs
    Args:
        code (str): The JS code to execute, returning the value to be used as the output of this transformation.
    Returns:
        transformed (any): Return value of provided code
        duration (int): Duration of provided code in milliseconds
    """

    def __init__(self, code: str, step_name: str = "js_code_transformation", *args, **kwargs):
        self.code = code
        self.step_name = step_name
        self._outputs = ["transformed", "duration"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        return [
            {
                "transformation": "js_code_transformation",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": {"code": self.code},
            }
        ]
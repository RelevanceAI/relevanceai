from relevanceai.steps._base import StepBase


class RunChain(StepBase):
    def __init__(
        self, chain_id: str, params: dict, step_name: str = "run_chain", *args, **kwargs
    ):
        self.chain_id = chain_id
        self.params = params
        self.step_name = step_name
        self._outputs = [
            "output",
            "state",
            "status",
            "errors",
            "cost",
            "credits_used",
            "executionTime",
        ]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        return [
            {
                "transformation": "run_chain",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": {"chain_id": self.chain_id, "params": self.params},
            }
        ]

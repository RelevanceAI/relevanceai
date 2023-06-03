import requests
from relevanceai._request import handle_response
from relevanceai.auth import config, Auth
from relevanceai.steps._base import StepBase


def list_all_steps(auth: Auth = None):
    if auth is None:
        auth = config.auth
    response = requests.get(
        f"https://api-{auth.region}.stack.tryrelevance.com/latest/studios/transformations/list",
    )
    res = handle_response(response)
    results_list = []
    for s in res["transformations"]:
        results_list.append(
            {
                "id": s["transformation_id"],
                "name": s["name"],
                "description": s["description"],
            }
        )
    return results_list


class RunStep(StepBase):
    def __init__(self, step_id: str, step_name: str = None, *args, **kwargs):
        self.list_of_steps = list_all_steps()
        self.step_id = step_id
        for step in self.list_of_steps:
            if step["transformation_id"] == self.step_id:
                self.step_definition = step
        self.step_name = (
            self.step_definition["name"] if step_name is None else step_name
        )
        self._inputs = [
            t for t in self.step_definition["input_schema"]["properties"].keys()
        ]
        self._required = (
            self.step_definition["input_schema"]["required"]
            if "required" in self.step_definition["input_schema"]
            else []
        )
        self._outputs = [
            t for t in self.step_definition["output_schema"]["properties"].keys()
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
                "params": {},
            }
        ]

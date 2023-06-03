import requests
from relevanceai.steps._base import StepBase


class MakeAPIRequest(StepBase):
    """Make API request
    Make an API request based on provided input and return the response.

    Args:
        url (str): The URL to make the request to.
        method (str): The HTTP method to use.
        headers ((Optional) dict): The headers to send with the request.
        body ((Optional) str or dict): The body to send with the request.
        response_type ((Optional) str): The format of the response.

    Returns:
        response_body (dict): {}
        status (int): {'type': 'number'}
    """

    def __init__(
        self,
        url: str,
        method: str,
        headers: dict = None,
        body: str or dict = None,
        response_type: str = None,
        step_name: str = "api_call",
        *args,
        **kwargs,
    ) -> None:
        self.url = url
        self.method = method
        self.headers = headers
        self.body = body
        self.response_type = response_type
        self.step_name = step_name
        self._outputs = ["response_body", "status"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        step_params = {
            "url": self.url,
            "method": self.method,
        }
        if self.headers is not None:
            step_params["headers"] = self.headers
        if self.body is not None:
            step_params["body"] = self.body
        if self.response_type is not None:
            step_params["response_type"] = self.response_type
        return [
            {
                "transformation": "api_call",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": step_params,
            }
        ]
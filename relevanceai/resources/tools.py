
from ..types.params import *
from .._resource import SyncAPIResource
from functools import cached_property


class Tools(SyncAPIResource):

    def _format_params(self, params: list):
        formatted_params = []

        for param in params:
            param_type = param.get("type")

            if param_type == "string":
                formatted_params.append(
                    StringParam(
                        name=param.get("name"),
                        long=param.get("long", False),
                        title=param.get("title", "Text Input"),
                        description=param.get("description", "")
                    )
                )
            elif param_type == "number":
                formatted_params.append(
                    NumberParam(
                        name=param.get("name"),
                        min=param.get("min"),
                        max=param.get("max"),
                        title=param.get("title", "Number Input"),
                        description=param.get("description", "")
                    )
                )
            elif param_type == "options":
                formatted_params.append(
                    OptionsParam(
                        name=param.get("name"),
                        options=param.get("options"),
                        title=param.get("title", "Options"),
                        description=param.get("description", "")
                    )
                )
            # Add more param types as needed
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        # Create a Parameters object and convert it to JSON
        parameters_obj = Parameters(formatted_params)
        return parameters_obj.to_json()

    def trigger(
        self, 
        tool_id: str, 
        params: dict = None, 
        full_response: bool = False
    ): 

        # Format the parameters using the _format_params method
        formatted_params = self._format_params(params or [])

        # Define the request path and body
        path = f"studios/{tool_id}/trigger" 
        body = {"params": formatted_params}

        # Make the API call
        response = self._post(path=path, body=body)
        
        if full_response:
            return response.json()
        else:
            return response.json().get("output")
        


from relevanceai.steps._base import StepBase

class VectorizeAndSearchArray(StepBase):
    """Vectorize and search array
    Vectorise an array of strings and rank items by relevance to your search query.
    Args:
        array (list): The array of data to search. If it is an object, it will be stringified when searching.
        query (str): The query to search for.
        page_size (int): The number of results to return.
        field (str): The field to search in if array includes objects.
    Returns:
        results (list): {'type': 'array', 'items': {}}
    """
    def __init__(
        self,
        array: list,
        query: str,
        page_size: int = 5,
        field: str = None,
        step_name: str = "vectorize_and_search_array",
        *args,
        **kwargs
    ) -> None:
        self.array = array
        self.query = query
        self.page_size = page_size
        self.field = field
        self.step_name = step_name
        self._outputs = ["results"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        step_params = {
            "array": self.array,
            "query": self.query,
            "page_size": self.page_size,
        }
        if self.field is not None:
            step_params["field"] = self.field
        return [
            {
                "transformation": "search_array",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": step_params,
            }
        ]
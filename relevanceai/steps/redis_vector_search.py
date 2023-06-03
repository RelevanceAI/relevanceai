from relevanceai.steps._base import StepBase

class RedisSearch(StepBase):
    """Vector search on Redis
    Retrieve data from Redis based on semantic similarity.
    Args:
        index (str): The name of the index to search.
        query (str): The search query.
        vector_field (str): The name of the field that contains the vector.
        model (str): The model name to use.
        page_size (int): The number of results to return.
    Returns:
        results (list): The search results.
    """
    def __init__(
        self,
        index: str,
        query: str,
        vector_field: str,
        model: str,
        page_size: int = 5,
        step_name: str = "redis_search",
        *args,
        **kwargs
    ) -> None:
        self.index = index
        self.query = query
        self.vector_field = vector_field
        self.model = model
        self.page_size = page_size
        self.step_name = step_name
        self._outputs = ["results"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        return [
            {
                "transformation": "redis_search",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": {
                    "index": self.index,
                    "query": self.query,
                    "vector_field": self.vector_field,
                    "model": self.model,
                    "page_size": self.page_size
                }
            }
        ]
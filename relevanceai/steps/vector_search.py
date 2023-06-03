from relevanceai.steps._base import StepBase


class VectorSimilaritySearch(StepBase):
    """Vector similarity search on Relevance dataset
    Search your dataset based on semantic similarity.
    Args:
        dataset_id (str): The ID of the dataset to search.
        query (str): The query to search for.
        vector_field (str): The name of the field that contains the vector.
        model (str): The model name to use.
        content_field ((Optional) str):
        page_size ((Optional) int): The number of results to return.
    Returns:
        results (list): {'type': 'array', 'items': {'type': ['string', 'object']}}
    """

    def __init__(
        self,
        dataset_id: str,
        query: str,
        vector_field: str,
        model: str,
        content_field: str = None,
        page_size: int = 5,
        step_name: str = "vector_similarity_search",
        *args,
        **kwargs,
    ) -> None:
        self.dataset_id = dataset_id
        self.query = query
        self.vector_field = vector_field
        self.model = model
        self.content_field = content_field
        self.page_size = page_size
        self.step_name = step_name
        self._outputs = ["results"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        step_params = {
            "dataset_id": self.dataset_id,
            "query": self.query,
            "vector_field": self.vector_field,
            "model": self.model,
        }
        if self.content_field is not None:
            step_params["content_field"] = self.content_field
        if self.page_size is not None:
            step_params["page_size"] = self.page_size
        return [
            {
                "transformation": "search",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": step_params,
            }
        ]

from relevanceai.steps._base import StepBase

class GenerateVectorEmbedding(StepBase):
    """Generate vector embedding
    Generate a vector embedding from a given input with a choice of models.
    Args:
        input (str): The input to generate a vector embedding with.
        model (str): The model name to use.
    Returns:
        vector (list): The vector embedding.
    """
    def __init__(
        self,
        input: str,
        model: str,
        step_name: str = "generate_vector_embedding",
        *args,
        **kwargs
    ) -> None:
        self.input = input
        self.model = model
        self.step_name = step_name
        self._outputs = ["vector"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        return [
            {
                "transformation": "generate_vector_embedding",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": {
                    "input": self.input,
                    "model": self.model
                }
            }
        ]
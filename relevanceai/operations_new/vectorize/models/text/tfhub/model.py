from typing import List, Any

from relevanceai.operations_new.vectorize.models.base import ModelBase
from relevanceai.utils.decorators.vectors import catch_errors


class TFHubText2Vec(ModelBase):
    def __init__(self, url, vector_length):
        import tensorflow_hub as hub

        self.model: Any = hub.load(url)
        self.vector_length: int = vector_length

    @catch_errors
    def encode(self, text: str) -> List[float]:
        """The function takes a string as input and returns a list of floats

        Parameters
        ----------
        text : str
            The text to be encoded.

        Returns
        -------
            A list of floats.

        """
        return self.model.encode([text])[0].tolist()

    @catch_errors
    def bulk_encode(self, texts: List[str]) -> List[List[float]]:
        """The function takes a list of strings and returns a list of lists of floats

        Parameters
        ----------
        texts : List[str]
            List[str]

        Returns
        -------
            A list of lists of floats.

        """
        return self.model.encode(texts).tolist()

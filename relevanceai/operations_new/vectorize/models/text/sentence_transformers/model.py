from typing import List

from relevanceai.operations_new.vectorize.models.base import ModelBase
from relevanceai.utils.decorators.vectors import catch_errors


class SentenceTransformer2Vec(ModelBase):
    def __init__(self, model, vector_length, model_name):
        from sentence_transformers import SentenceTransformer

        self.model: SentenceTransformer = model
        self.vector_length: int = vector_length
        self.model_name = model_name

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

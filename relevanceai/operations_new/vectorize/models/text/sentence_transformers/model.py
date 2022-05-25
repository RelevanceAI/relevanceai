from typing import Any, List, Union

from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.utils.decorators.vectors import catch_errors


class SentenceTransformer2Vec(VectorizeModelBase):
    def __init__(
        self,
        model: Any,
        vector_length: Union[None, int],
        model_name: str,
    ):
        from sentence_transformers import SentenceTransformer

        self.model: SentenceTransformer = model
        if vector_length is None:
            self.vector_length = self._get_vector_length()
        else:
            self.vector_length = vector_length
        self.model_name = model_name

    def _get_vector_length(self) -> int:
        """It takes a string, encodes it into a vector, and returns the length of that vector

        Returns
        -------
            The length of the vector.

        """
        vector = self.model.encode(["test"]).tolist()[0]
        length = len(vector)
        return length

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

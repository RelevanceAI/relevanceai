"""
Faiss Base
"""
from typing import Any, List, Dict, Optional, Tuple

import numpy as np

from relevanceai.operations_new.cluster.models.base import ClusterModelBase

from faiss import Kmeans


class FaissModel(ClusterModelBase):
    """Faiss model base"""

    model: Kmeans

    def __init__(
        self,
        model: Any,
        model_kwargs: Optional[Dict] = None,
    ):

        if isinstance(model, str):
            model = ClusterModelBase.import_from_string(f"faiss.Kmeans")
            if model_kwargs is None:
                model_kwargs = {}

            self.model = model(**model_kwargs)
        else:
            self.model = model

    @staticmethod
    def check_vectors(input: Any) -> np.ndarray:
        """It takes a list of vectors, converts them to a numpy array, and then converts them to a float32

        Parameters
        ----------
        input : Any
            Any

        Returns
        -------
            the input as a numpy array.

        """
        if not isinstance(input, np.ndarray):
            try:
                input = np.array(input)
            except Exception:
                raise ValueError("Could not convert vectors to np.ndarray")

        try:
            output = input.astype(np.float32)
        except Exception:
            raise ValueError("Could not convert vectors to np.float32")

        return output

    def fit_predict(self, vectors: Any) -> List[int]:
        """> The function takes in a list of vectors and returns a list of labels

        Parameters
        ----------
        vectors : Any
            Any - the vectors to be clustered

        Returns
        -------
            The labels of the vectors.

        """
        vectors = FaissModel.check_vectors(vectors)

        self.fit(vectors)
        res: Tuple[np.ndarray, np.ndarray] = self.model.assign(vectors)

        labels: np.ndarray = res[1]

        return labels.tolist()

    def fit(self, vectors: Any) -> np.float64:
        """It trains the model.

        Parameters
        ----------
        vectors : Any
            Any

        Returns
        -------
            The model is being trained on the vectors.

        """
        vectors = FaissModel.check_vectors(vectors)

        return self.model.train(vectors)

    def predict(self, vectors) -> List[int]:
        """This function takes in a list of vectors and returns a list of labels

        Parameters
        ----------
        vectors
            The vectors to be predicted.

        Returns
        -------
            The labels of the vectors.

        """
        vectors = FaissModel.check_vectors(vectors)

        res: Tuple[np.ndarray, np.ndarray] = self.model.assign(vectors)

        labels: np.ndarray = res[1]

        return labels.tolist()

    @property
    def cluster_centers_(self):
        """It returns the centroids of the clusters

        Returns
        -------
            The cluster centers

        """
        return self.model.centroids

    @property
    def alias(self):
        """The function takes a string and returns a string

        Returns
        -------
            The alias of the model.

        """
        return f"fiass{self.name}-{str(self.model.k)}"

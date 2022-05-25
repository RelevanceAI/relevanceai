from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from relevanceai.utils import DocUtils


class _ModelUtils(DocUtils):
    _centroids = None

    def predict_documents(self, vector_fields, documents):
        if len(vector_fields) == 1:
            vectors = self.get_field_across_documents(vector_fields[0], documents)
            return self.predict(vectors)
        raise NotImplementedError(
            "support for multiple vector fields not available right now."
        )

    def fit_predict_documents(
        self,
        vector_fields: List[str],
        documents: List[Dict[str, Any]],
        warm_start: bool = False,
    ):
        if len(vector_fields) == 1:
            vectors = self.get_field_across_documents(vector_fields[0], documents)
            cluster_labels = self.fit_predict(vectors, warm_start)
            return self.format_cluster_labels(cluster_labels)
        raise NotImplementedError(
            "support for multiple vector fields not available right now."
        )

    def __str__(self):
        if hasattr(self.model):
            return str(self.model)
        else:
            return "generic_cluster_model"

    def format_cluster_label(self, label):
        """> If the label is an integer, return a string that says "cluster_" and the integer. If the label is
        a string, return the string. If the label is neither, raise an error

        Parameters
        ----------
        label
            the label of the cluster. This can be a string or an integer.

        Returns
        -------
            A list of lists.

        """
        if isinstance(label, str):
            return label
        return "cluster_" + str(label)

    def format_cluster_labels(self, labels):
        return [self.format_cluster_label(label) for label in labels]


class ModelBase(ABC, _ModelUtils):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def calculate_centroids(
        self, labels: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        centroids = []
        for label in sorted(np.unique(labels).tolist()):
            centroid = vectors[labels == label].mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        return centroids

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[Dict[str, Any]]:
        # Returns output cluster labels
        # assumes model has been trained
        # You need to run `fit` or `fit_predict` first
        # self.model.predict()
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self, *args, **kwargs) -> List[int]:
        # Trains on vectors and returns output cluster labels
        # Sklearn gives some optimization between fit and predict step (idk what though)
        raise NotImplementedError

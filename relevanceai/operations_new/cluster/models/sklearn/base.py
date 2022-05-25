"""
Sklearn Base
"""
from typing import Any, List, Dict, Optional, Union

import numpy as np

from sklearn.base import ClusterMixin

from relevanceai.operations_new.cluster.models.base import ModelBase
from relevanceai.operations_new.cluster.models.sklearn import sklearn_models


class SklearnModelBase(ModelBase):
    """Sklearn model base"""

    def __init__(
        self,
        model: Union[ClusterMixin, str],
        model_kwargs: dict,
    ):
        if isinstance(model, str):
            assert model in sklearn_models
            self.model_ref = SklearnModelBase.import_from_string(
                f"sklearn.cluster.{sklearn_models[model]}"
            )
            if model_kwargs is None:
                model_kwargs = {}

            self.model = self.model_ref(**model_kwargs)
        else:
            assert type(model).__name__ in list(sklearn_models.values())
            self.model = model

        self.model_kwargs = model_kwargs
        for key, value in self.model_kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def import_from_string(name):
        components = name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def warm_start(self):
        # not really
        model_ref = SklearnModelBase.import_from_string("sklearn.cluster.KMeans")
        kwargs = self.model_kwargs
        kwargs["init"] = self.model.init
        self.model = model_ref(kwargs)

    def partial_fit(self, *args, **kwargs) -> Any:
        if hasattr(self.model, "partial_fit"):
            return self.model.partial_fit(*args, **kwargs)
        raise ValueError("Model class does not have a `partial_fit` method")

    def fit_predict(self, vectors, *args, **kwargs) -> List[int]:
        warm_start = kwargs.pop("warm_state", False)
        if warm_start and self.name == "kmeans":
            self.warm_start()

        labels: np.ndarray = self.model.fit_predict(vectors, *args, **kwargs)
        self._centroids = self.calculate_centroids(labels, vectors)
        return labels.tolist()

    def fit(self, *args, **kwargs) -> Any:
        warm_start = kwargs.pop("warm_state", False)
        if warm_start and self.name == "kmeans":
            self.warm_start()

        return self.model.fit(*args, **kwargs)

    def predict(self, vectors, *args, **kwargs) -> List[int]:
        labels: np.ndarray = self.model.predict(vectors, *args, **kwargs)
        self._centroids = self.calculate_centroids(labels, vectors)
        return labels.tolist()

    @property
    def name(self):
        return type(self.model).__name__.lower()

    @property
    def alias(self):
        return self.name + str(self.model.n_clusters)


from sklearn.cluster import KMeans

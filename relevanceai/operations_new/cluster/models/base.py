from abc import ABC, abstractmethod
from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class ModelBase(ABC, DocUtils):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

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

    def predict_documents(self, vector_fields, documents):
        if len(vector_fields) == 1:
            vectors = self.get_field_across_documents(vector_fields[0], documents)
            return self.predict(vectors)
        raise NotImplementedError(
            "support for multiple vector fields not available right now."
        )

    def fit_predict_documents(self, vector_fields, documents):
        if len(vector_fields) == 1:
            vectors = self.get_field_across_documents(vector_fields[0], documents)
            return self.fit_predict(vectors)
        raise NotImplementedError(
            "support for multiple vector fields not available right now."
        )

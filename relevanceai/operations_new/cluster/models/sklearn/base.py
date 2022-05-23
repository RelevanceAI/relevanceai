"""
Sklearn Base
"""
from relevanceai.operations_new.cluster.models.base import ModelBase
from relevanceai.utils.doc_utils.doc_utils import DocumentList
from typing import List, Union
from sklearn.base import ClusterMixin


class SklearnModelBase(ClusterMixin, ModelBase):
    """Sklearn model base"""

    model: ClusterMixin

    def fit_predict(self, *args, **kwargs) -> List:
        return self.model.fit_predict(*args, **kwargs)

    def fit(self, *args, **kwargs) -> List:
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs) -> List:
        raise self.model.predict(*args, **kwargs)

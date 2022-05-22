"""
Sklearn Base
"""
from relevanceai.operations_new.cluster.models.base import ModelBase
from relevanceai.utils.doc_utils.doc_utils import DocumentList
from typing import List, Union
from sklearn.base import ClusterMixin


class SklearnModelBase(ClusterMixin, ModelBase):
    """Sklearn model base"""

    def fit_predict(self, *args, **kwargs) -> List:
        return super().fit_predict(*args, **kwargs)

    def predict_documents(self, documents: Union[List, DocumentList]) -> DocumentList:
        pass

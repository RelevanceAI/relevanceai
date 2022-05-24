"""
OperationBase
"""
from typing import Any, Optional, List
from relevanceai.operations_new.cluster.batch.models.base import BatchClusterModelBase
from relevanceai.operations_new.base import OperationBase


class BatchClusterBase(OperationBase):
    def __init__(
        self,
        vector_fields: list,
        model: BatchClusterModelBase = None,
        model_kwargs: Optional[dict] = None,
        *args,
        **kwargs
    ):
        self.vector_fields = vector_fields
        self._check_vector_fields()
        self.model = self._get_model(model)
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs: dict = model_kwargs

    def partial_fit(self, documents: List):
        """Run partial fitting on a list of documents"""
        return self.model.partial_fit(
            self.get_field_across_documents(self.vector_fields, documents)
        )

    def transform(self, documents):
        if hasattr(self.model, "transform"):
            cluster_labels = self.model.predict(documents)
            self.set_field_across_documents(cluster_labels, documents)
        return documents

    def _get_model(self, model):
        if isinstance(model, str):
            self.model_name = model

            model = OperationBase.normalize_string(model)
            if model == "minibatchkmeans":
                from sklearn.cluster import MiniBatchKMeans

                self.model = MiniBatchKMeans(**self.model_kwargs)
            else:
                raise ValueError("Only supports minibatchkmeans for now.")
        return model

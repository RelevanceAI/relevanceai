"""
OperationBase
"""
from typing import Any, Optional, List
from relevanceai.operations_new.cluster.alias import ClusterAlias
from relevanceai.operations_new.cluster.batch.models.base import BatchClusterModelBase
from relevanceai.operations_new.base import OperationBase
from relevanceai.operations_new.cluster.base import ClusterBase


class BatchClusterBase(ClusterBase, ClusterAlias):
    def __init__(
        self,
        vector_fields: list,
        model: BatchClusterModelBase = None,
        model_kwargs: Optional[dict] = None,
        alias: str = None,
        *args,
        **kwargs,
    ):
        self.vector_fields = vector_fields
        self._check_vector_fields()
        self.model = self._get_model(model)
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs: dict = model_kwargs
        self.alias = self._get_alias(alias)

    def partial_fit(self, documents: List):
        """Run partial fitting on a list of documents"""
        if isinstance(self.model, str):
            self.model = self._get_model(self.model)
        return self.model.partial_fit(
            self.get_field_across_documents(self.vector_fields, documents)
        )

    @property
    def full_cluster_field(self):
        if not hasattr(self, "_cluster_field"):
            self._cluster_field = self._get_cluster_field_name()
            print("Saving cluster labels to: ")
            print(self._cluster_field)
        return self._cluster_field

    def transform(self, documents):
        if hasattr(self.model, "predict"):
            cluster_labels = self.model.predict(
                self.get_field_across_documents(self.vector_fields[0], documents)
            )
            cluster_labels = self.format_cluster_labels(cluster_labels)
            self.set_field_across_documents(
                self.full_cluster_field, cluster_labels, documents
            )
        elif hasattr(self.model, "transform"):
            documents = self.model.transform(documents)
        else:
            raise AttributeError("Model missing a predict.")
        return [
            {
                "_id": d["_id"],
                self.full_cluster_field: self.get_field(self.full_cluster_field, d),
            }
            for d in documents
        ]

    def _get_model(self, model):
        if isinstance(model, str):
            self.model_name = model
            model = OperationBase.normalize_string(model)
            if model == "minibatchkmeans":
                from sklearn.cluster import MiniBatchKMeans

                self.model = MiniBatchKMeans(**self.model_kwargs)
                return self.model
            else:
                raise ValueError("Only supports minibatchkmeans for now.")
        return model

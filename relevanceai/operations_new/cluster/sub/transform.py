"""
Subclustering operation
"""
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List, Union
from relevanceai.operations_new.cluster.transform import ClusterTransform


class SubClusterTransform(ClusterTransform):
    def __init__(
        self,
        model,
        alias: str,
        vector_fields,
        parent_field,
        filters: Optional[list] = None,
        cluster_ids: Optional[list] = None,
        min_parent_cluster_size: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        cluster_field: str = "_cluster_",
        outlier_value: Union[int, str] = -1,
        outlier_label: str = "outlier",
        **kw
    ):
        self.model = model
        self.alias = alias
        if len(vector_fields) > 1:
            raise NotImplementedError
        self.vector_fields = vector_fields
        self.parent_field = parent_field
        self.filters = filters
        self.cluster_ids = cluster_ids
        self.min_parent_cluster_size = min_parent_cluster_size
        self.cluster_field = cluster_field
        self.model = self._get_model(model=model, model_kwargs=model_kwargs)
        self.outlier_value = outlier_value
        self.outlier_label = outlier_label
        for k, v in kw.items():
            setattr(self, k, v)

    def transform(self, documents):
        if not self.is_field_across_documents(self.vector_fields[0], documents):
            raise ValueError(
                "You have missing vectors in your document. You will want to apply a filter for vector fields. See here for a page of filter options: https://relevanceai.readthedocs.io/en/development/core/filters/exists.html#exists."
            )
        parent_values = self.get_field_across_documents(self.parent_field, documents)
        vectors = self.get_field_across_documents(self.vector_fields[0], documents)
        labels = self.model.fit_predict(vectors)
        sub_labels = self._format_sub_labels(parent_values, labels)
        # Get the cluster field name
        cluster_field_name = self._get_cluster_field_name()

        documents_to_upsert = [{"_id": d["_id"]} for d in documents]

        self.set_field_across_documents(
            cluster_field_name, sub_labels, documents_to_upsert
        )
        return documents_to_upsert

    def _format_sub_labels(self, parent_values: list, labels: np.ndarray) -> List[str]:
        if len(parent_values) != len(labels):
            raise ValueError("Improper logic for parent values")

        if isinstance(labels, np.ndarray):
            labels = labels.flatten().tolist()

        cluster_labels = [
            label + "-" + str(labels[i])
            if label != self.outlier_value
            else self.outlier_label
            for i, label in enumerate(parent_values)
        ]
        return cluster_labels

    def store_subcluster_metadata(self, parent_field: str, cluster_field: str):
        """
        Store subcluster metadata
        """
        return self.append_metadata_list(
            field="_subcluster_",
            value_to_append={
                "parent_field": parent_field,
                "cluster_field": cluster_field,
            },
            only_unique=True,
        )

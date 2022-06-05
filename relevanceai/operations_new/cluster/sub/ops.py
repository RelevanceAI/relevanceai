from relevanceai.operations_new.cluster.sub.base import SubClusterBase
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.cluster.ops import ClusterOps
from typing import Optional, Union


class SubClusterOps(SubClusterBase, ClusterOps):
    def __init__(
        self,
        model,
        alias: str,
        vector_fields,
        parent_field,
        dataset_id,
        filters: Optional[list] = None,
        cluster_ids: Optional[list] = None,
        min_parent_cluster_size: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        cluster_field: str = "_cluster_",
        outlier_value: Union[int, str] = -1,
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
        self.dataset_id = dataset_id
        for k, v in kw.items():
            setattr(self, k, v)

    def run(self, *args, **kwargs):
        # Loop through unique cluster values first
        # then run through it
        super().run(*args, **kwargs)
        self.store_subcluster_metadata(
            parent_field=self.parent_field, cluster_field=self.cluster_field
        )

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

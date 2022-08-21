import warnings
from relevanceai.operations_new.cluster.sub.transform import SubClusterTransform
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.cluster.ops import ClusterOps
from typing import Optional, Union
from copy import deepcopy


class SubClusterOps(SubClusterTransform, ClusterOps):
    def __init__(
        self,
        model,
        alias: str,
        vector_fields,
        parent_field,
        dataset_id,
        filters: Optional[list] = None,
        cluster_ids: Optional[list] = None,
        min_parent_cluster_size: int = 0,
        model_kwargs: Optional[dict] = None,
        cluster_field: str = "_cluster_",
        outlier_value: Union[int, str] = -1,
        **kw,
    ):
        self.model = model
        self.alias = alias
        if len(vector_fields) > 1:
            raise NotImplementedError
        self.vector_fields = vector_fields
        self.parent_field = parent_field
        self.filters = filters if filters is not None else []
        self.cluster_ids = cluster_ids
        self.min_parent_cluster_size = min_parent_cluster_size
        self.cluster_field = cluster_field
        self.model = self._get_model(model=model, model_kwargs=model_kwargs)
        self.outlier_value = outlier_value
        self.dataset_id = dataset_id
        self.include_cluster_report = False
        for k, v in kw.items():
            setattr(self, k, v)

    def _list_unique_values(self, field):
        # TODO: Switch to a more complete subclustering approach
        facets = self.datasets.facets(
            dataset_id=self.dataset_id, fields=[field], page_size=9999
        )
        return [
            x["value"]
            for x in facets["results"][self.parent_field]
            if x["frequency"] > self.min_parent_cluster_size
        ]

    def run(self, *args, **kwargs):
        # Loop through unique cluster values first
        # then run through it
        if kwargs.get("filters") is not None:
            self.filters = kwargs.pop("filters")
        cluster_ids = (
            self._list_unique_values(field=self.parent_field)
            if self.cluster_ids is None
            else self.cluster_ids
        )
        for cluster_id in cluster_ids:
            self.parent_label = cluster_id
            print(f"Subclustering on {cluster_id}...")
            # Create filters to loop through
            new_filters = deepcopy(self.filters) + [
                {
                    "field": self.parent_field,
                    "filter_type": "exact_match",
                    "condition": "==",
                    "condition_value": cluster_id,
                }
            ]
            super().run(filters=new_filters, *args, **kwargs)

        subcluster_field_name = self._get_cluster_field_name()
        # Store the relevant metadata
        self.store_subcluster_metadata(
            parent_field=self.parent_field, cluster_field=subcluster_field_name
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

    def get_centroid_documents(self):
        centroid_vectors = {}
        if hasattr(self.model, "_centroids") and self.model._centroids is not None:
            centroid_vectors = self.model._centroids
            # get the cluster label function
            labels = range(len(centroid_vectors))
            cluster_ids = self.format_subcluster_labels(
                labels, [self.parent_label for _ in range(len(labels))]
            )
            if len(self.vector_fields) > 1:
                warnings.warn(
                    "Currently do not support inserting centroids with multiple vector fields"
                )
            centroids = [
                {"_id": k, self.vector_fields[0]: v}
                for k, v in zip(cluster_ids, centroid_vectors)
            ]
        else:
            centroids = self.create_subcluster_centroids()
        return centroids

    def create_subcluster_centroids(self):
        if len(self.vector_fields) > 1:
            raise NotImplementedError(
                "Do not currently support multiple vector fields for centroid creation."
            )

        # calculate the centroids
        centroid_vectors = self.calculate_centroids()
        return centroid_vectors

    def format_subcluster_labels(self, labels: list, parent_labels: str):
        return [
            self.format_subcluster_label(label, parent_labels[i])
            for i, label in enumerate(labels)
        ]

    def format_subcluster_label(self, label: str, parent_label: str):
        if isinstance(label, str):
            return label
        return parent_label + "-" + str(label)

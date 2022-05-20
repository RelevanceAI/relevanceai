from relevanceai._api.api_client import (
    APIClient,
)  # this needs to be replace by below in dr PR

# from relevanceai.operations_new.apibase import OperationsAPIClient


from relevanceai.constants.errors import MissingClusterError
from typing import Callable, Dict, Any, Set, List
from relevanceai.operations_new.cluster.base import ClusterBase


class ClusterOps(ClusterBase, APIClient):
    """
    Cluster-related functionalities
    """

    # These need to be instantiated on __init__
    def __init__(
        self,
        dataset_id: str,
        vector_fields: list,
        alias: str,
        cluster_field: str,
        *args,
        **kwargs,
    ):
        """
        ClusterOps objects
        """
        self.dataset_id = dataset_id
        self.vector_fields = vector_fields
        self.alias = alias
        self.cluster_field = cluster_field

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_cluster_field_name(self, alias: str = None):
        if alias is None:
            alias = self.alias
        if isinstance(self.vector_fields, list):
            if hasattr(self, "cluster_field"):
                set_cluster_field = (
                    f"{self.cluster_field}.{'.'.join(self.vector_fields)}.{alias}"
                )
            else:
                set_cluster_field = f"_cluster_.{'.'.join(self.vector_fields)}.{alias}"
        elif isinstance(self.vector_fields, str):
            set_cluster_field = f"{self.cluster_field}.{self.vector_fields}.{alias}"
        return set_cluster_field

    def _operate(self, cluster_id: str, field: str, output: dict, func: Callable):
        """
        Internal function for operations

        It takes a cluster_id, a field, an output dictionary, and a function, and then it gets all the
        documents in the cluster, gets the field across all the documents, and then applies the function
        to the field

        Parameters
        ----------
        cluster_id : str
            str, field: str, output: dict, func: Callable
        field : str
            the field you want to get the value for
        output : dict
            dict
        func : Callable
            Callable

        """
        cluster_field = self._get_cluster_field_name()
        # TODO; change this to fetch all documents
        documents = self.datasets.documents.get_where(
            self.dataset_id,
            filters=[
                {
                    "field": cluster_field,
                    "filter_type": "exact_match",
                    "condition": "==",
                    "condition_value": cluster_id,
                },
                {
                    "field": field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                },
            ],
            select_fields=[field, cluster_field],
            page_size=9999,
        )
        # get the field across each
        arr = self.get_field_across_documents(field, documents["documents"])
        output[cluster_id] = func(arr)

    def _operate_across_clusters(self, field: str, func: Callable):
        output: Dict[str, Any] = dict()
        for cluster_id in self.list_cluster_ids():
            self._operate(cluster_id=cluster_id, field=field, output=output, func=func)
        return output

    def list_cluster_ids(
        self,
        alias: str = None,
        minimum_cluster_size: int = 3,
        dataset_id: str = None,
        num_clusters: int = 1000,
    ):
        """
        List unique cluster IDS

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            cluster_ops = client.ClusterOps(
                alias="kmeans_8", vector_fields=["sample_vector_]
            )
            cluster_ops.list_cluster_ids()

        Parameters
        -------------
        alias: str
            The alias to use for clustering
        minimum_cluster_size: int
            The minimum size of the clusters
        dataset_id: str
            The dataset ID
        num_clusters: int
            The number of clusters

        """
        # Mainly to be used for subclustering
        # Get the cluster alias
        cluster_field = self._get_cluster_field_name(alias=self.alias)

        # currently the logic for facets is that when it runs out of pages
        # it just loops - therefore we need to store it in a simple hash
        # and then add them to a list
        all_cluster_ids: Set = set()

        while len(all_cluster_ids) < num_clusters:
            facet_results = self.datasets.facets(
                dataset_id=self.dataset_id,
                fields=[cluster_field],
                page_size=int(self.config["data.max_clusters"]),
                page=1,
                asc=True,
            )
            if "results" in facet_results:
                facet_results = facet_results["results"]
            if cluster_field not in facet_results:
                raise MissingClusterError(alias=alias)
            for facet in facet_results[cluster_field]:
                if facet["frequency"] > minimum_cluster_size:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[cluster_field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)

        return list(all_cluster_ids)

    def _insert_centroids(
        self,
        dataset_id: str,
        vector_fields: List[str],
        centroid_documents: List[Dict[str, Any]],
    ) -> None:
        self.datasets.cluster.centroids.insert(
            dataset_id=dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=vector_fields,
            alias=self.alias,
        )

    def create_centroids(self):
        """
        Calculate centroids from your vectors

        Example
        --------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            cluster_ops = ds.ClusterOps(
                alias="kmeans-25",
                vector_fields=['sample_vector_']
            )
            centroids = cluster_ops.create_centroids()

        """
        import numpy as np

        # Get an array of the different vectors
        if len(self.vector_fields) > 1:
            raise NotImplementedError(
                "Do not currently support multiple vector fields for centroid creation."
            )
        centroid_vectors = {}

        def calculate_centroid(vectors):
            X = np.array(vectors)
            return X.mean(axis=0)

        centroid_vectors = self._operate_across_clusters(
            field=self.vector_fields[0], func=calculate_centroid
        )

        # Does this insert properly?
        if isinstance(centroid_vectors, dict):
            centroid_vectors = [
                {"_id": k, self.vector_fields[0]: v}
                for k, v in centroid_vectors.items()
            ]
        self._insert_centroids(
            dataset_id=self.dataset_id,
            vector_fields=[self.vector_fields[0]],
            centroid_documents=centroid_vectors,
        )
        return centroid_vectors

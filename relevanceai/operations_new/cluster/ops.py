from relevanceai._api.api_client import (
    APIClient,
)  # this needs to be replace by below in dr PR

from relevanceai.constants.errors import MissingClusterError
from relevanceai.utils.decorators.analytics import track
from relevanceai.operations_new.cluster.base import ClusterBase
from typing import Callable, Dict, Any, Set, List, Optional


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
        centroid_documents,
    ) -> None:
        # Centroid documents are in the format {"cluster-0": [1, 1, 1]}
        self.datasets.cluster.centroids.insert(
            dataset_id=self.dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=self.vector_fields,
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
            centroid_documents=centroid_vectors,
        )
        return centroid_vectors

    def list_closest(
        self,
        cluster_ids: Optional[list] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[list] = None,
        facets: Optional[list] = None,
        include_vector: bool = False,
        cluster_properties_filters: Optional[Dict] = None,
        include_count: bool = False,
        include_facets: bool = False,
        verbose: bool = False,
    ):
        """
        List of documents closest from the center.
        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_fields: list
            The vector fields where a clustering task runs
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
        centroid_vector_fields: list
            Vector fields stored
        select_fields: list
            Fields to include in the search results, empty array/list means all fields
        approx: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
        sum_fields: bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        facets: list
            Fields to include in the facets, if [] then all
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        include_facets: bool
            Include facets in the search results
        cluster_properties_filter: dict
            Filter if clusters with certain characteristics should be hidden in results
        """
        return self.datasets.cluster.centroids.list_closest_to_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=cluster_ids,
            select_fields=select_fields,
            approx=approx,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            filters=filters,
            facets=facets,
            include_vector=include_vector,
            include_count=include_count,
            include_facets=include_facets,
            cluster_properties_filter=cluster_properties_filters,
            verbose=verbose,
        )

    @track
    def list_furthest(
        self,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 3,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        # facets: List = [],
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
        cluster_properties_filter: Optional[Dict] = {},
    ):
        """
        List documents furthest from the center.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_fields: list
            The vector field where a clustering task was run.
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
        select_fields: list
            Fields to include in the search results, empty array/list means all fields
        approx: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
        sum_fields: bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        facets: list
            Fields to include in the facets, if [] then all
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        include_facets: bool
            Include facets in the search results
        """
        return self.datasets.cluster.centroids.list_furthest_from_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=cluster_ids,
            centroid_vector_fields=centroid_vector_fields,
            select_fields=select_fields,
            approx=approx,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            filters=filters,
            min_score=min_score,
            include_vector=include_vector,
            include_count=include_count,
            cluster_properties_filter=cluster_properties_filter,
        )

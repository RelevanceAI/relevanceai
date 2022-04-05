from typing import Dict, List, Optional
from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials


class Centroids(APIClient):
    def __init__(self, credentials: Credentials, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(credentials)

    def __call__(
        self, vector_fields: list, alias: str, cluster_field: str = "_cluster_"
    ):
        """
        Instaniates Centroids Class which stores centroid information to be called

        Parameters
        ----------
        vector_fields: list
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster
        cluster_field: string
            Name of clusters in documents

        Example
        --------
        .. code-block
            from relevanceai import Client

            client = Client()

            df = client.Dataset("sample_dataset_id")

            df.get(["sample_id"], include_vector=False)

        """

        self.vector_fields = vector_fields
        self.alias = alias
        self.cluster_field = cluster_field
        self.cluster_doc_field = (
            f"{self.cluster_field}.{self.vector_fields[0]}.{self.alias}"
        )
        return self

    def closest(
        self,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
    ):
        """
        List of documents closest from the centre.

        Parameters
        ----------
        cluster_ids: list
            Any of the cluster ids
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
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results

        Example
        -----------------
        .. code-block::

            from relevanceai import Client
            from relevanceai.ops.clusterer import ClusterOps
            from relevanceai.ops.clusterer.kmeans_clusterer import KMeansModel

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            vector_field = "vector_field_"
            n_clusters = 10

            model = KMeansModel(k=n_clusters)

            df.cluster(model=model, alias=f"kmeans-{n_clusters}", vector_fields=[vector_field])

        """
        cluster_ids = [] if cluster_ids is None else cluster_ids
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters

        return self.datasets.cluster.centroids.list_closest_to_center(
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
        )

    def furthest(
        self,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
    ):
        """
        List of documents furthest from the centre.

        Parameters
        ----------
        cluster_ids: list
            Any of the cluster ids
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
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results

        """
        cluster_ids = [] if cluster_ids is None else cluster_ids
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters

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
        )

    def update(
        self,
        dataset_id: str,
        vector_fields: List[str],
        centroid_vector_fields: List[str],
        alias: str,
        cluster_centers: List[Dict[str, List[float]]],
    ):
        """
        API reference link: https://api.us-east-1.relevance.ai/latest/core/documentation#operation/UpdateClusterCentroids

        Update the centroids contained within your dataset

        Parameters
        ----------

        dataset_id: str
            The name of the dataset

        vector_fields: List[str]
            A lit of the vectors fields in your dataset that have cluster centroids you wish to update

        alias: str
            The alias that was used to cluster

        cluster_centers: List[Dict[str: List[float]]]
            A List containing dictionaries of cluster id's to be updated, with their keys being the new centroids
        """

        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/cluster/centroids/update",
            method="POST",
            parameters={
                "vector_fields": vector_fields,
                "centroid_vector_fields": centroid_vector_fields,
                "alias": alias,
                "cluster_centers": cluster_centers,
            },
        )

    def delete_centroid_by_id(
        self, centroid_id: str, dataset_id: str, vector_field: str, alias: str
    ):
        """
        OLD API reference link: https://api.us-east-1.relevance.ai/latest/documentation#operation/delete_centroids_api_services_cluster_centroids__centroid_id__delete_post

        Delete a centroid by ID

        Parameters
        ----------

        centroid_id: str
            The id of the centroid

        dataset_id: str
            The name of the dataset

        vector_field: str
            The vector_field that contains the cluster id

        alias: str
            The alias that was used to cluster
        """

        return self.make_http_request(
            endpoint=f"services/cluster/centroids/{centroid_id}/delete",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "vector_field": vector_field,
                "alias": alias,
            },
        )

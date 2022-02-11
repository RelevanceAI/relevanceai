from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.groupby import Groupby, Agg
from typing import List


class Centroids(BatchAPIClient):
    def __init__(self, project: str, api_key: str, dataset_id: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid
        self.dataset_id = dataset_id
        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)

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

        # Check if cluster is in schema
        schema = self.datasets.schema(self.dataset_id)
        self._are_fields_in_schema([self.cluster_doc_field], self.dataset_id, schema)
        self.cluster_field_type = schema[self.cluster_doc_field]

        self.cluster_groupby = [
            {
                "name": "cluster",
                "field": self.cluster_doc_field,
                "agg": self.cluster_field_type,
            }
        ]
        self.groupby = Groupby(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            _pre_groupby=self.cluster_groupby,
        )
        self.agg = Agg(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            groupby_call=self.cluster_groupby,
        )
        return self

    def closest(
        self,
        cluster_ids: List = [],
        centroid_vector_fields: List = [],
        select_fields: List = [],
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: List = [],
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
            from relevanceai.clusterer import ClusterOps
            from relevanceai.clusterer.kmeans_clusterer import KMeansModel

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            vector_field = "vector_field_"
            n_clusters = 10

            model = KMeansModel(k=n_clusters)

            df.cluster(model=model, alias=f"kmeans-{n_clusters}", vector_fields=[vector_field])


        """

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
        cluster_ids: List = [],
        centroid_vector_fields: List = [],
        select_fields: List = [],
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: List = [],
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

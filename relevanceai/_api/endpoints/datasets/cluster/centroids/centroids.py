from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from typing import List, Optional, Dict


class CentroidsClient(_Base):
    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    def list_closest_to_center(
        self,
        dataset_id: str,
        vector_fields: List,
        alias: str,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        facets: Optional[List] = None,
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
        include_facets: bool = False,
        cluster_properties_filter: Optional[Dict] = {},
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
        cluster_ids = [] if cluster_ids is None else cluster_ids
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters
        facets = [] if facets is None else facets

        if not centroid_vector_fields:
            centroid_vector_fields = vector_fields
        parameters = {
            "vector_fields": vector_fields,
            "centroid_vector_fields": centroid_vector_fields,
            "alias": alias,
            "dataset_id": dataset_id,
            "cluster_ids": cluster_ids,
            "select_fields": select_fields,
            "approx": approx,
            "sum_fields": sum_fields,
            "page_size": page_size,
            "page": page,
            "similarity_metric": similarity_metric,
            "filters": filters,
            "facets": facets,
            "min_score": min_score,
            "include_vector": include_vector,
            "include_count": include_count,
            "include_facets": include_facets,
            "cluster_properties_filter": cluster_properties_filter,
        }
        endpoint = f"/datasets/{dataset_id}/cluster/centroids/list_closest_to_center"
        method = "POST"
        # self._log_to_dashboard(
        #     method=method,
        #     parameters=parameters,
        #     endpoint=endpoint,
        #     dashboard_type="cluster_centroids_closest",
        #     verbose=verbose,
        # )
        return self.make_http_request(endpoint, method=method, parameters=parameters)

    documents_closest_to_center = list_closest_to_center

    def list_furthest_from_center(
        self,
        dataset_id: str,
        vector_fields: List,
        alias: str,
        centroid_vector_fields: Optional[List] = None,
        cluster_ids: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        facets: Optional[List] = None,
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
        include_facets: bool = False,
        cluster_properties_filter: Optional[Dict] = {},
    ):
        """
        List of documents furthest from the center.
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
        cluster_properties_filter: dict
            Filter if clusters with certain characteristics should be hidden in results
        """
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        cluster_ids = [] if cluster_ids is None else cluster_ids
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters
        facets = [] if facets is None else facets

        if not centroid_vector_fields:
            centroid_vector_fields = vector_fields
        endpoint = f"/datasets/{dataset_id}/cluster/centroids/list_furthest_from_center"
        method = "POST"
        parameters = {
            "dataset_id": dataset_id,
            "vector_fields": vector_fields,
            "alias": alias,
            "cluster_ids": cluster_ids,
            "select_fields": select_fields,
            "centroid_vector_fields": centroid_vector_fields,
            "approx": approx,
            "sum_fields": sum_fields,
            "page_size": page_size,
            "page": page,
            "similarity_metric": similarity_metric,
            "filters": filters,
            "facets": facets,
            "min_score": min_score,
            "include_vector": include_vector,
            "include_count": include_count,
            "include_facets": include_facets,
            "cluster_properties_filter": cluster_properties_filter,
        }
        # self._log_to_dashboard(
        #     method=method,
        #     parameters=parameters,
        #     endpoint=endpoint,
        #     dashboard_type="cluster_centroids_furthest",
        # )
        response = self.make_http_request(
            endpoint, method=method, parameters=parameters
        )
        return response

    documents_furthest_from_center = list_furthest_from_center

    def update(
        self,
        dataset_id: str,
        vector_fields: List[str],
        alias: str,
        cluster_centers: List[Dict[str, List[float]]],
        centroid_vector_fields: List[str] = None,
    ):
        """
        API reference link: https://api.us-east-1.relevance.ai/latest/core/documentation#operation/UpdateClusterCentroids

        Update the centroids contained within your dataset

        Parameters
        ----------

        dataset_id: str
            The name of the dataset

        vector_fields: List[str]
            A list of the vectors fields in your dataset that have cluster centroids you wish to update

        alias: str
            The alias that was used to cluster

        cluster_centers: List[Dict[str: List[float]]]
            A List containing dictionaries of cluster id's to be updated, with their keys being the new centroids
        """
        if centroid_vector_fields is None:
            centroid_vector_fields = []
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/cluster/centroids/update",
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

    def insert(
        self,
        dataset_id: str,
        cluster_centers: List,
        vector_fields: List,
        alias: str = "default",
    ):
        """
        Insert your own cluster centroids for it to be used in approximate search settings and cluster aggregations.
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        cluster_centers : list
            Cluster centers with the key being the index number
        vector_field: string
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster
        """
        return self.make_http_request(
            f"/datasets/{dataset_id}/cluster/centroids/insert",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "cluster_centers": cluster_centers,
                "vector_fields": vector_fields,
                "alias": alias,
            },
        )

    def documents(
        self,
        dataset_id: str,
        vector_fields: List,
        cluster_ids: Optional[List] = None,
        alias: str = "default",
        page_size: int = 5,
        page: int = 1,
        include_vector: bool = False,
    ):
        """
        Retrieve the cluster centroids by IDs

        Parameters
        -------------

        cluster_ids : list
            List of cluster IDs
        vector_fields: list
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster
        page_size: int
            Size of each page of results
        cursor: string
            Cursor to paginate the document retrieval
        page: int
            Page of the results
        include_vector: bool
            Include vectors in the search results
        """
        cluster_ids = [] if cluster_ids is None else cluster_ids

        return self.make_http_request(
            f"/datasets/{dataset_id}/cluster/centroids/documents",
            method="POST",
            parameters={
                "cluster_ids": cluster_ids,
                "vector_fields": vector_fields,
                "alias": alias,
                "page_size": page_size,
                "page": page,
                "include_vector": include_vector,
            },
        )

    list = documents

    def delete(
        self,
        dataset_id: str,
        centroid_id: str,
        alias: str,
        vector_fields: List[str],
        centroid_vector_fields: Optional[List[str]] = None,
        centroid_dataset_id: Optional[str] = None,
    ):
        """
        Delete centroids by dataset ID, vector field and alias

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector_field: string
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster

        """
        parameters = {
            "dataset_id": dataset_id,
            "vector_fields": vector_fields,
            "alias": alias,
        }

        if centroid_vector_fields is not None:
            parameters.update({"centroid_vector_fields": centroid_vector_fields})

        if centroid_dataset_id is not None:
            parameters.update({"centroid_dataset_id": centroid_dataset_id})

        return self.make_http_request(
            f"/datasets/{dataset_id}/cluster/centroids/{centroid_id}/delete",
            method="POST",
            parameters=parameters,
        )

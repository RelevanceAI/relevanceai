from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from typing import List, Optional


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
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
    ):
        """
        List of documents closest from the centre.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_field: string
            The vector field where a clustering task was run.
        cluster_ids: lsit
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
        """
        cluster_ids = [] if cluster_ids is None else cluster_ids
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters

        if not centroid_vector_fields:
            centroid_vector_fields = vector_fields
        parameters = {
            "dataset_id": dataset_id,
            "vector_fields": vector_fields,
            "alias": alias,
            "cluster_ids": cluster_ids,
            "centroid_vector_fields": centroid_vector_fields,
            "select_fields": select_fields,
            "approx": approx,
            "sum_fields": sum_fields,
            "page_size": page_size,
            "page": page,
            "similarity_metric": similarity_metric,
            "filters": filters,
            "min_score": min_score,
            "include_vector": include_vector,
            "include_count": include_count,
        }
        endpoint = f"/datasets/{dataset_id}/cluster/centroids/list_closest_to_center"
        method = "POST"
        self._log_to_dashboard(
            method=method,
            parameters=parameters,
            endpoint=endpoint,
            dashboard_type="cluster_centroids_closest",
        )
        return self.make_http_request(endpoint, method=method, parameters=parameters)

    def list_furthest_from_center(
        self,
        dataset_id: str,
        vector_fields: List[str],
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
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
    ):
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        cluster_ids = [] if cluster_ids is None else cluster_ids
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters
        """
        List of documents furthest from the centre.

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
            "min_score": min_score,
            "include_vector": include_vector,
            "include_count": include_count,
        }
        self._log_to_dashboard(
            method=method,
            parameters=parameters,
            endpoint=endpoint,
            dashboard_type="cluster_centroids_furthest",
        )
        response = self.make_http_request(
            endpoint, method=method, parameters=parameters
        )
        return response

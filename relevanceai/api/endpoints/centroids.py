from relevanceai.base import _Base
from typing import Optional, Dict, Any, List


class CentroidsClient(_Base):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    def list(
        self,
        dataset_id: str,
        vector_field: str,
        alias: str = "default",
        page_size: int = 5,
        cursor: str = None,
        include_vector: bool = False,
        base_url="https://gateway-api-aueast.relevance.ai/latest",
    ):
        """
        Retrieve the cluster centroid

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector_field: string
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster
        page_size: int
            Size of each page of results
        cursor: string
            Cursor to paginate the document retrieval
        include_vector: bool
            Include vectors in the search results
        """
        return self.make_http_request(
            "/services/cluster/centroids/list",
            method="GET",
            parameters={
                "dataset_id": dataset_id,
                "vector_field": vector_field,
                "alias": alias,
                "page_size": page_size,
                "cursor": cursor,
                "include_vector": include_vector,
            },
            base_url=base_url,
        )

    def get(
        self,
        dataset_id: str,
        cluster_ids: list,
        vector_field: str,
        alias: str = "default",
        page_size: int = 5,
        cursor: str = None,
    ):
        """
        Retrieve the cluster centroids by IDs

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        cluster_ids : list
            List of cluster IDs
        vector_field: string
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster
        page_size: int
            Size of each page of results
        cursor: string
            Cursor to paginate the document retrieval
        """
        return self.make_http_request(
            "/services/cluster/centroids/get",
            method="GET",
            parameters={
                "dataset_id": dataset_id,
                "cluster_ids": cluster_ids,
                "vector_field": vector_field,
                "alias": alias,
                "page_size": page_size,
                "cursor": cursor,
            },
        )

    def insert(
        self,
        dataset_id: str,
        cluster_centers: list,
        vector_field: str,
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
            "/services/cluster/centroids/insert",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "cluster_centers": cluster_centers,
                "vector_field": vector_field,
                "alias": alias,
            },
        )

    def documents(
        self,
        dataset_id: str,
        cluster_ids: list,
        vector_field: str,
        alias: str = "default",
        page_size: int = 5,
        cursor: str = None,
        page: int = 1,
        include_vector: bool = False,
        similarity_metric: str = "cosine",
    ):
        """
        Retrieve the cluster centroids by IDs

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        cluster_ids : list
            List of cluster IDs
        vector_field: string
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
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']

        """
        return self.make_http_request(
            "/services/cluster/centroids/documents",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "cluster_ids": cluster_ids,
                "vector_field": vector_field,
                "alias": alias,
                "page_size": page_size,
                "cursor": cursor,
                "page": page,
                "include_vector": include_vector,
                "similarity_metric": similarity_metric,
            },
        )

    def metadata(
        self,
        dataset_id: str,
        vector_field: str,
        alias: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        If metadata is none, retrieves metadata about a dataset. notably description, data source, etc
        Otherwise, you can store the metadata about your cluster here.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_field: string
            The vector field where a clustering task was run.
        alias: string
            Alias is used to name a cluster
        metadata: Optional[dict]
           If None, it will retrieve the metadata, otherwise
           it will overwrite the metadata of the cluster

        """
        if metadata is None:
            return self.make_http_request(
                "/services/cluster/centroids/metadata",
                method="GET",
                parameters={
                    "dataset_id": dataset_id,
                    "vector_field": vector_field,
                    "alias": alias,
                },
            )
        else:
            return self.make_http_request(
                "/services/cluster/centroids/metadata",
                method="POST",
                parameters={
                    "dataset_id": dataset_id,
                    "vector_field": vector_field,
                    "alias": alias,
                    "metadata": metadata,
                },
            )

    def list_closest_to_center(
        self,
        dataset_id: str,
        vector_field: str,
        cluster_ids: List = [],
        alias: str = "default",
        select_fields: list = [],
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: list = [],
        facets: list = [],
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
        include_facets: bool = False,
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

        parameters = {
            "dataset_id": dataset_id,
            "vector_field": vector_field,
            "alias": alias,
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
        }
        endpoint = "/services/cluster/centroids/list_closest_to_center"
        method = "POST"
        self._log_to_dashboard(
            method=method,
            parameters=parameters,
            endpoint=endpoint,
            dashboard_type="cluster_centroids_closest",
        )
        return self.make_http_request(endpoint, method=method, parameters=parameters)

    docs_closest_to_center = list_closest_to_center

    def list_furthest_from_center(
        self,
        dataset_id: str,
        vector_field: str,
        cluster_ids: List = [],
        alias: str = "default",
        select_fields: List = [],
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: List = [],
        facets: List = [],
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
        include_facets: bool = False,
    ):
        """
        List of documents furthest from the centre.

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

        endpoint = "/services/cluster/centroids/list_furthest_from_center"
        method = "POST"
        parameters = {
            "dataset_id": dataset_id,
            "vector_field": vector_field,
            "alias": alias,
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

    docs_furthest_from_center = list_furthest_from_center

    def delete(
        self,
        dataset_id: str,
        vector_field: str,
        alias: str = "default",
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
        return self.make_http_request(
            "/services/cluster/centroids/delete",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "vector_field": vector_field,
                "alias": alias,
            },
        )

    def update(
        self,
        dataset_id: str,
        vector_field: str,
        id: str,
        update: dict = {},
        alias: str = "default",
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
        id: string
            The centroid ID
        update: dict
            The update to be applied to the document
        """
        return self.make_http_request(
            "/services/cluster/centroids/update",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "vector_field": vector_field,
                "alias": alias,
                "id": id,
                "update": update,
            },
        )

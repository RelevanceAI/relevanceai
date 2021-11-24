from relevanceai.base import Base


class Centroids(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def list(
        self,
        dataset_id: str,
        vector_field: str,
        alias: str = "default",
        page_size: int = 5,
        cursor: str = None,
        include_vector: bool = False,
        output_format: str = "json",
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
            output_format=output_format,
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
        output_format: str = "json",
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
            output_format=output_format,
        )

    def insert(
        self,
        dataset_id: str,
        cluster_centers: dict,
        vector_field: str,
        alias: str = "default",
        output_format: str = "json",
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
            output_format=output_format,
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
        output_format: str = "json",
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
            output_format=output_format,
        )
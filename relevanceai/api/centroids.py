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
        base_url="https://gateway-api-aueast.relevance.ai/latest/",
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
            "services/cluster/centroids/list",
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
            "services/cluster/centroids/get",
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

from relevanceai.base import Base
from relevanceai.api.endpoints.centroids import Centroids


class Cluster(Base):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        self.centroids = Centroids(project=project, api_key=api_key)
        super().__init__(project, api_key)

    def aggregate(
        self,
        dataset_id: str,
        vector_field: str,
        metrics: list = [],
        groupby: list = [],
        filters: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default"
    ):
        """ 
        Takes an aggregation query and gets the aggregate of each cluster in a collection. This helps you interpret each cluster and what is in them.
        It can only can be used after a vector field has been clustered. \n

        For more information about aggregations check out services.aggregate.aggregate. 

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector_field : string
            The vector field that was clustered on
        metrics: list
            Fields and metrics you want to calculate
        groupby: list 
            Fields you want to split the data into
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten
        alias: string
            Alias used to name a vector field. Belongs in field_{alias}vector
        """
        return self.make_http_request(
            endpoint="/services/cluster/aggregate",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "aggregation_query": {"groupby": groupby, "metrics": metrics},
                "filters": filters,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "flatten": flatten,
                "vector_field": vector_field,
                "alias": alias,
            }
        )

    def facets(
        self,
        dataset_id: str,
        facets_fields: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        date_interval: str = "monthly"
    ):
        """ 
        Takes a high level aggregation of every field and every cluster in a collection. This helps you interpret each cluster and what is in them. \n
        Only can be used after a vector field has been clustered.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        facets_fields : list
            Fields to include in the facets, if [] then all
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        date_interval: string
            Interval for date facets
        """
        return self.make_http_request(
            endpoint="/services/cluster/facets",
            method="GET",
            parameters={
                "dataset_id": dataset_id,
                "facets_fields": facets_fields,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "date_interval": date_interval,
            }
        )
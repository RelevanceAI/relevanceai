from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from relevanceai._api.endpoints.datasets.cluster.centroids import CentroidsClient
from typing import Optional, List


class ClusterClient(_Base):
    def __init__(self, credentials: Credentials):
        self.centroids = CentroidsClient(credentials)
        super().__init__(credentials)

    def aggregate(
        self,
        dataset_id: str,
        vector_fields: list,
        metrics: Optional[list] = None,
        groupby: Optional[list] = None,
        sort: Optional[list] = None,
        filters: Optional[list] = None,
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default",
    ):
        """
        Takes an aggregation query and gets the aggregate of each cluster in a collection. This helps you interpret each cluster and what is in them.
        It can only can be used after a vector field has been clustered. \n
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector_fields : list
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
        metrics = [] if metrics is None else metrics
        groupby = [] if groupby is None else groupby
        sort = [] if sort is None else sort
        filters = [] if filters is None else filters

        endpoint = f"/datasets/{dataset_id}/cluster/aggregate"
        method = "POST"
        parameters = {
            "dataset_id": dataset_id,
            "aggregation_query": {"groupby": groupby, "metrics": metrics, "sort": sort},
            "filters": filters,
            "page_size": page_size,
            "page": page,
            "asc": asc,
            "flatten": flatten,
            "vector_fields": vector_fields,
            "alias": alias,
        }
        # self._log_to_dashboard(
        #     method=method,
        #     parameters=parameters,
        #     endpoint=endpoint,
        #     dashboard_type="cluster_aggregation",
        # )
        return self.make_http_request(
            endpoint=endpoint, method=method, parameters=parameters
        )

    def merge(
        self,
        dataset_id: str,
        vector_fields: List[str],
        alias: str,
        cluster_ids: list,
        centroid_vector_fields: List[str] = None,
    ):
        """
        Merge clusters together
        """
        endpoint = f"/datasets/{dataset_id}/cluster/merge"
        method = "POST"
        if centroid_vector_fields is None:
            centroid_vector_fields = []
        parameters = {
            "dataset_id": dataset_id,
            "vector_fields": vector_fields,
            "centroid_vector_fields": centroid_vector_fields,
            "alias": alias,
            "cluster_ids": cluster_ids,
        }
        return self.make_http_request(
            endpoint=endpoint, method=method, parameters=parameters
        )

from typing import Optional, Union, Dict, List, Mapping
from relevanceai.dataset.read.statistics import Statistics


class ClusterRead(Statistics):
    def list_closest_to_center(
        self,
        vector_fields: list,
        alias: str,
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
        return self.datasets.cluster.centroids.list_closest_to_center(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            alias=alias,
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

from ..base import Base


class Aggregate(Base):
    """Aggregate service"""

    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def aggregate(
        self,
        dataset_id: str,
        metrics: list = [],
        groupby: list = [],
        filters: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default",
    ):
        """Aggregate means that you are"""
        return self.make_http_request(
            "services/aggregate/aggregate",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "aggregation_query": {"groupby": groupby, "metrics": metrics},
                "filters": filters,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "flatten": flatten,
                "alias": alias,
            },
        )

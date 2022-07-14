from typing import Dict, List, Optional, Union
from relevanceai.apps.explorer_app.base import ExplorerBase


class ExplorerApp(ExplorerBase):
    def preview_fields(self, fields: List = None, append=True):
        fields = [] if fields is None else fields
        self._append_or_replace(
            "default-preview-centroids-columns", fields, default_value=[]
        )

    # Search and filter section
    def facets(self, fields: List = None, append=True):
        fields = [] if fields is None else fields
        self._append_or_replace("filtering-facets", fields, default_value=[])

    def sort(self, metrics: List = None, append=True):
        metrics = [] if metrics is None else metrics
        if "key-metrics" not in self.config:
            self.config["key-metrics"]["previewQuery"] = {"groupby": [], "metrics": []}
        if append:
            self.config["key-metrics"]["previewQuery"][
                "metrics"
            ] += self.dataset._clean_metrics(metrics, name_prefix="")[0]
        else:
            self.config["key-metrics"]["previewQuery"][
                "metrics"
            ] = self.dataset._clean_metrics(metrics, name_prefix="")[0]

    def sort_default(self, metric_name: str):
        if "key-metrics" not in self.config or not self.config["key-metrics"]:
            raise TypeError("'sort' needs to be specified to use 'sort_default'")
        if isinstance(metric_name, str):
            self.config["sort-default"] = self.dataset._return_sort_in_metrics(
                metric_name, self.config["key-metrics"]["previewQuery"]["metrics"]
            )
        else:
            raise TypeError("'sort_default' needs to be of type string.")

    def search_fields(self, fields: List = None):
        fields = [] if fields is None else fields
        self.config["text-fields-to-search"] = fields

    def vector_search_fields(self, vector_fields: List = None):
        vector_fields = [] if vector_fields is None else vector_fields
        self.config["vector-fields-to-search"] = vector_fields

    def search_min_relevance(self, min_relevance: int):
        self.config["search-minimum-relevance"] = min_relevance

    # Chart section
    def chart(
        self,
        groupby: List[Dict] = None,
        metrics: List[Dict] = None,
        sort: List[Dict] = None,
        page_size: int = None,
        chart_name: str = None,
        chart_mode: str = "column",
        show_frequency: bool = False,
        **kwargs
    ):
        groupby = [] if groupby is None else groupby
        metrics = [] if metrics is None else metrics

        main_groupby = self.dataset._clean_groupby(groupby)[0]
        main_metrics = self.dataset._clean_metrics(metrics)[0]
        chart_config = self._agg_to_chart_config(
            main_groupby,
            main_metrics,
            sort=sort,
            page_size=page_size,
            chart_name=chart_name,
            chart_mode=chart_mode,
            show_frequency=show_frequency,
            **kwargs
        )
        self._append_or_replace(
            "aggregation-charts",
            [self._agg_to_chart_config(**chart_config)],
            default_value=[],
        )

    def charts(self, charts, append=True):
        if isinstance(charts, dict):
            self._append_or_replace(
                "aggregation-charts",
                [self._agg_to_chart_config(**charts)],
                default_value=[],
            )
        elif isinstance(charts, list):
            self._append_or_replace(
                "aggregation-charts",
                [self._agg_to_chart_config(**c) for c in charts],
                default_value=[],
            )
        else:
            raise TypeError("'charts' needs to be a list or dictionary.")

    # Cluster section
    def cluster(self, cluster_field: str):
        self.config["cluster"] = cluster_field

    def cluster_charts(self, charts: List):
        self._append_or_replace("cluster-preview-config", charts, default_value=[])

    def label_clusters(self, cluster_labels: Dict):
        if "cluster" in self.config:
            alias = self.config["cluster"].split(".")[-1]
            vector_fields = ".".join(self.config["cluster"].split(".")[1:-1])
            self.dataset.label_clusters(cluster_labels, alias, vector_fields)
        else:
            raise Exception("Specify your cluster field first with .cluster.")

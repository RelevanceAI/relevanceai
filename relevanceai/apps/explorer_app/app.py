import copy
from typing import Dict, List, Optional, Union
from relevanceai.apps.explorer_app.base import ExplorerBase


class ExplorerApp(ExplorerBase):
    def preview_fields(self, fields: List = None, append=True):
        fields = [] if fields is None else fields
        self._append_or_replace(
            "default-preview-centroids-columns", fields, default_value=[]
        )
        self.document_card(fields=fields)

    def document_card(
        self,
        fields=None,
        primary_field: str = None,
        secondary_field: str = None,
        image_field: str = None,
        layout_mode: str = None,
        page_size: int = None,
        append: bool = True,
    ):
        fields_config = []
        multi_fields = copy.deepcopy(fields)
        if image_field:
            fields_config.append({"id": "image", "value": image_field})
            if image_field in multi_fields:
                multi_fields.remove(image_field)
        if primary_field:
            fields_config.append({"id": "primary", "value": primary_field})
            if primary_field in multi_fields:
                multi_fields.remove(primary_field)
        if secondary_field:
            fields_config.append({"id": "secondary", "value": secondary_field})
            if secondary_field in multi_fields:
                multi_fields.remove(secondary_field)
        if fields:
            fields_config.append({"id": "multiple-fields", "value": multi_fields})

        if "document-view-configuration" not in self.config:
            self.config["document-view-configuration"] = {
                "layout-template": "custom",
                "layout-template-configuration": {"fields": []},
            }
        if layout_mode:
            self.config["document-view-configuration"]["layout-mode"] = layout_mode
        elif "layout-mode" not in self.config["document-view-configuration"]:
            self.config["document-view-configuration"]["layout-mode"] = "grid"
        if page_size:
            self.config["document-view-configuration"]["page-size"] = page_size
        elif "page-size" not in self.config["document-view-configuration"]:
            self.config["document-view-configuration"]["page-size"] = 20

        if append:
            self.config["document-view-configuration"]["layout-template-configuration"][
                "fields"
            ] += fields_config
        else:
            self.config["document-view-configuration"]["layout-template-configuration"][
                "fields"
            ] = fields_config

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

    def search_min_relevance(self, min_relevance: float):
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
    def cluster(self, alias: str, vector_field: str):
        self.config["cluster"] = {"alias": alias, "vector_field": vector_field}

    def cluster_charts(self, charts: List):
        self._append_or_replace("cluster-preview-config", charts, default_value=[])

    def label_clusters(self, cluster_labels: Dict):
        if "cluster" in self.config:
            self.dataset.label_clusters(
                cluster_labels,
                self.config["cluster"]["alias"],
                self.config["cluster"]["vector_field"],
            )
        else:
            raise Exception("Specify your cluster field first with .cluster.")

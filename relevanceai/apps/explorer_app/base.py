import warnings
from typing import Dict, List, Optional, Union


class ExplorerBase:
    def __init__(
        self,
        name: str,
        dataset,
        deployable_id: str = None,
        default_view: str = "charts",
        charts_view_column: int = 2,
        preview_centroids_page_size: int = 2,
        **kwargs,
    ):
        self.name = name
        # if not isinstance(dataset, Dataset):
        #     raise TypeError("'dataset' only accepts Relevance Dataset class. to create one use ds = client.Dataset('example').")
        self.dataset = dataset
        self.dataset_id = dataset.dataset_id
        self.deployable_id = deployable_id
        if default_view not in ["charts", "documents", "categories"]:
            raise KeyError(
                "'default_view' can only be one of ['charts', 'documents', 'categories']"
            )
        self.default_view = default_view
        app_config = None
        self.reloaded = False
        if deployable_id:
            try:
                app_config = self.dataset.get_app(deployable_id)
                self.reloaded = True
            except:
                raise Exception(
                    f"{deployable_id} does not exist in the dataset, the given id will be used for creating a new app."
                )
        if app_config:
            self.config = app_config["configuration"]
        else:
            self.config = {
                "dataset_name": self.dataset_id,
                "deployable_name": name,
                "type": "explore",
                "charts-view-columns": charts_view_column,
                "results-mode": default_view,
                "preview-centroids-page-size": preview_centroids_page_size,
                **kwargs,
            }

    def deploy(self, overwrite: bool = False):
        if self.deployable_id and self.reloaded:
            status = self.dataset.update_app(
                self.deployable_id, self.config, overwrite=overwrite
            )
            if status["status"] == "success":
                return self.dataset.get_app(self.deployable_id)
            else:
                raise Exception("Failed to update app")
        return self.dataset.create_app(self.config)

    def _append_or_replace(
        self, key: str, value, append: bool = True, default_value=None
    ):
        default_value = [] if default_value is None else default_value

        if key not in self.config:
            self.config[key] = default_value
        if append:
            self.config[key] += value
        else:
            self.config[key] = value

    def _agg_to_chart_config(
        self,
        groupby: List[Dict] = None,
        metrics: List[Dict] = None,
        sort: List[Dict] = None,
        page_size=None,
        chart_name: str = None,
        chart_mode: str = "column",
        show_frequency: bool = False,
        **kwargs,
    ):
        groupby = [] if groupby is None else groupby
        metrics = [] if metrics is None else metrics

        chart_config = {
            "groupby": groupby,
            "metrics": metrics,
            "chart-mode": chart_mode,
            "chart-name": chart_name,
            "show-frequency": show_frequency,
            **kwargs,
        }
        if not chart_name:
            chart_name = "Chart "  # can give more flexibility to this
            if groupby:
                groupby_names = [g["name"] for g in groupby]
                chart_name += f"grouped by ({', '.join(groupby_names)}) "
            if metrics:
                metrics_names = [m["name"] for m in metrics]
                chart_name += f"with metrics ({', '.join(metrics_names)}) "
            chart_config["chart-name"] = chart_name
        if sort:
            chart_config["metric-to-sort-by"] = self.dataset._return_sort_in_metrics(
                sort, metrics
            )
        if page_size:
            chart_config["page-size"] = page_size
        return chart_config

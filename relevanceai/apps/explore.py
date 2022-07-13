import warnings
from typing import Dict, List, Optional, Union
class ExploreApp:
    def __init__(self, 
        name, 
        dataset, 
        deployable_id=None,
        default_view="charts", 
        charts_view_column=2,
        preview_centroids_page_size=2,
        client=None,
        **kwargs
    ):
        self.name = name
        # if not isinstance(dataset, Dataset):
        #     raise TypeError("'dataset' only accepts Relevance Dataset class. to create one use ds = client.Dataset('example').")
        self.dataset = dataset
        self.dataset_id = dataset.dataset_id
        self.deployable_id = deployable_id
        if default_view not in ["charts", "documents", "categories"]:
            raise KeyError("'default_view' can only be one of ['charts', 'documents', 'categories']")
        self.default_view = default_view
        app_config = None
        self.reloaded = False
        if deployable_id:
            try:
                app_config = self.dataset.get_app(deployable_id)
                self.reloaded = True
            except:
               raise Exception(f"{deployable_id} does not exist in the dataset, the given id will be used for creating a new app.")            
        if app_config:
            self.config = app_config["configuration"]
        else:
            self.config = {
                "dataset_name" : self.dataset_id,
                "deployable_name" : name,
                "type":"explore", 
                "charts-view-columns" : charts_view_column,
                "results-mode" : default_view,
                "preview-centroids-page-size" : preview_centroids_page_size,
                **kwargs
            }

    def deploy(self, overwrite=False):
        if self.deployable_id and self.reloaded:
            status = self.dataset.update_app(self.deployable_id, self.config, overwrite=overwrite)
            if status["status"] == "success":
                return self.dataset.get_app(self.deployable_id)
            else:
                raise Exception("Failed to update app")
        return self.dataset.create_app(self.config)
        
    def _append_or_replace(self, key, value, append=True, default_value=[]):
        if key not in self.config:
            self.config[key] = []
        if append:
            self.config[key] += value
        else:
            self.config[key] = value

    def preview_fields(self, fields=[], append=True):
        self._append_or_replace("default-preview-centroids-columns", fields, default_value=[])

    #Search and filter section
    def facets(self, fields=[], append=True):
        self._append_or_replace("filtering-facets", fields, default_value=[])

    def sort(self, metrics=[], append=True):
        if "key-metrics" not in self.config:
            self.config["key-metrics"]["previewQuery"] = {"groupby" : [], "metrics" : []}
        if append:
            self.config["key-metrics"]["previewQuery"]["metrics"] += self.dataset._clean_metrics(metrics, name_prefix="")[0]
        else:
            self.config["key-metrics"]["previewQuery"]["metrics"] = self.dataset._clean_metrics(metrics, name_prefix="")[0]

    def sort_default(self, metric_name):
        if "key-metrics" not in self.config or not self.config["key-metrics"]:
            raise TypeError("'sort' needs to be specified to use 'sort_default'")
        if isinstance(metric_name, str):
            self.config['sort-default'] = self.dataset._return_sort_in_metrics(
                metric_name, 
                self.config["key-metrics"]["previewQuery"]["metrics"]
            )
        else:
            raise TypeError("'sort_default' needs to be of type string.")

    def search_fields(self, fields=[]):
        self.config["text-fields-to-search"] = fields
    
    def vector_search_fields(self, vector_fields=[]):
        self.config["vector-fields-to-search"] = vector_fields
    
    def search_min_relevance(self, min_relevance):
        self.config["search-minimum-relevance"] = min_relevance
    
    #Chart section
    def chart(
        self, 
        groupby:List[Dict]=[], 
        metrics:List[Dict]=[], 
        sort:List[Dict]=None, 
        page_size:int=None,
        chart_name:str=None, 
        chart_mode:str="column", 
        show_frequency:bool=False,
        **kwargs
    ):
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
        self._append_or_replace("aggregation-charts", [self._agg_to_chart_config(**chart_config)], default_value=[])

    def charts(self, charts, append=True):
        if isinstance(charts, dict):
            self._append_or_replace("aggregation-charts", [self._agg_to_chart_config(**charts)], default_value=[])
        elif isinstance(charts, list):
            self._append_or_replace("aggregation-charts", [self._agg_to_chart_config(**c) for c in charts], default_value=[])
        else:
            raise TypeError("'charts' needs to be a list or dictionary.")

    #Cluster section
    def cluster(self, cluster_field):
        self.config["cluster"] = cluster_field

    def cluster_charts(self, charts):
        self._append_or_replace("cluster-preview-config", charts, default_value=[])

    def label_clusters(self, cluster_labels):
        if "cluster" in self.config:
            alias = self.config["cluster"].split('.')[-1]
            vector_fields = ".".join(self.config["cluster"].split('.')[1:-1])
            self.dataset.label_clusters(cluster_labels, alias, vector_fields)
        else:
            raise Exception("Specify your cluster field first with .cluster.")

    def _agg_to_chart_config(
        self, 
        groupby:List[Dict]=[], 
        metrics:List[Dict]=[], 
        sort:List[Dict]=None, 
        page_size:int=None,
        chart_name:str=None, 
        chart_mode:str="column", 
        show_frequency:bool=False,
        **kwargs
    ):  
        chart_config = {
            "groupby" : groupby,
            "metrics" : metrics,
            "chart-mode" : chart_mode,
            "chart-name" : chart_name,
            "show-frequency" : show_frequency,
            **kwargs
        }
        if not chart_name:
            chart_name = "Chart " # can give more flexibility to this
            if groupby:
                groupby_names = [g['name'] for g in groupby]
                chart_name += f"grouped by ({', '.join(groupby_names)}) "
            if metrics:
                metrics_names = [m['name'] for m in metrics]
                chart_name += f"with metrics ({', '.join(metrics_names)}) "
            chart_config["chart-name"] = chart_name
        if sort:
            chart_config["metric-to-sort-by"] = self.dataset._return_sort_in_metrics(sort, metrics)
        if page_size:
            chart_config["page-size"] = page_size
        return chart_config
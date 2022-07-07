from relevanceai._api import APIClient
from relevanceai.constants.errors import FieldNotFoundError

class CreateApps(APIClient):
    def list_apps(self, return_config=False):
        print("Note: Deployable is the same as App. Deployables are legacy names of what we call Apps in the backend.")
        if return_config:
            return [
                d
                for d in self.deployables.list()["deployables"]
                if d["dataset_id"] == self.dataset_id
            ]
        else:
            results = []
            for d in self.deployables.list()["deployables"]:
                if d["dataset_id"] == self.dataset_id:
                    result = {
                        "deployable_id" : d["deployable_id"],
                        "url" : f"https://cloud.relevance.ai/dataset/{d['dataset_id']}/deploy/explore/{d['project_id']}/{self.api_key}/{d['deployable_id']}/{self.region}"
                    }
                    if "configuration" in d:
                        if "deployable_name" in d["configuration"]:
                            result["deployable_name"] = d["configuration"]['deployable_name']
                        if "type" in d["configuration"]:
                            result["type"] = d['configuration']["type"]
                    results.append(result)
            return results

    def chart_config_from_agg(self, 
        groupby=[], metrics=[], sort=None, page_size=None,
        chart_name=None, chart_mode="column", 
    ):  
        chart_config = {
            "groupby" : groupby,
            "metrics" : metrics,
            "chart-mode" : chart_mode,
            "chart-name" : chart_name
        }
        if not chart_name:
            chart_name = f"{chart_mode.title()} chart "
            if groupby:
                groupby_names = [g['name'] for g in groupby]
                chart_name += f"grouped by ({', '.join(groupby_names)}) "
            if metrics:
                metrics_names = [m['name'] for m in metrics]
                chart_name += f"with metrics ({', '.join(metrics_names)}) "
            chart_config["chart-name"] = chart_name
        if sort:
            if sort in [m['name'] for m in metrics]:
                chart_config["metric-to-sort-by"] = sort
            else:
                raise KeyError("'sort' used in a chart is not in 'metrics'")
        if page_size:
            chart_config["page-size"] = page_size
        return chart_config

    def create_app_config(self, 
        app_name="",
        default_view="charts",
        sort_default=None,
        sort_default_direction=None, 
        sort=[],
        facets=[], 
        search_fields=[],
        vector_search_fields=[],
        charts=[],
        cluster_charts=[],
        preview_fields=[],
        search_min_relevance=None,
        cluster=None,
        **kwargs):
        
        configuration = {
            "dataset_name" : self.dataset_id,
            "deployable_name" : app_name,
            "type":"explore", 
            "charts-view-columns" : 2,
            "preview-centroids-page-size" : 2,
            **kwargs
        }
        if default_view not in ["charts", "documents", "categories"]:
            raise KeyError("'default_view' can only be one of ['charts', 'documents', 'categories']")
        configuration["results-mode"] = default_view
        configuration["default-preview-centroids-columns"] = preview_fields

        #Filter/Drilldown bar
        configuration["filtering-facets"] = facets
        configuration["key-metrics"] = []
        sort_names = []
        if sort:
            configuration["key-metrics"] = {"previewQuery" : {"groupby":[], "metrics":sort}}
            sort_names = [s['name'] for s in sort]

        if sort_default:
            if not sort_names:
                raise TypeError("'sort' needs to be specified to use 'sort_default'")
            if isinstance(sort_default, str):
                if sort_default not in sort_names:
                    raise TypeError("'sort_default' not in sort.")
                configuration["sort-default"] = sort_default
                if sort_default_direction:
                    configuration["sort-default-direction"] = sort_default_direction
            else:
                raise TypeError("'sort_default' needs to be of type string.")
        
        if search_fields: # make sure its text
            configuration["text-fields-to-search"] = search_fields
        if vector_search_fields:
            configuration["vector-fields-to-search"] = vector_search_fields
        if search_min_relevance:
            configuration["search-minimum-relevance"] = search_min_relevance

        #Charts section
        if charts:
            if isinstance(charts, dict):
                configuration["aggregation-charts"] = [
                    self.chart_config_from_agg(**charts)
                ]
            elif isinstance(charts, list):
                configuration["aggregation-charts"] = [
                    self.chart_config_from_agg(**c)
                    for c in charts
                ]

        #Cluster section
        if cluster_charts:
            configuration["cluster-preview-configuration"] = cluster_charts
        # elif charts:
        #     print("'cluster_charts' is empty, using 'charts' for 'cluster_charts'")
        #     configuration["cluster-preview-configuration"] = [
        #         self.create_chart_from_agg(**c) 
        #         for c in charts
        #     ]

        if cluster:
            configuration["cluster"] = cluster

        return configuration

    def create_app(self, config):
        result = self.deployables.create(dataset_id=self.dataset_id, configuration=config)
        print(f"Your app can be accessed at: https://cloud.relevance.ai/dataset/{result['dataset_id']}/deploy/explore/{result['project_id']}/{self.api_key}/{result['deployable_id']}/{self.region}")
        return result

    def update_app(self, deployable_id, config):
        return self.deployables.update(deployable_id=deployable_id, dataset_id=self.dataset_id, config=config)

    def delete_app(self, deployable_id):
        return self.deployables.delete(deployable_id=deployable_id)

    def get_app(self, deployable_id):
        return self.deployables.get(deployable_id=deployable_id)
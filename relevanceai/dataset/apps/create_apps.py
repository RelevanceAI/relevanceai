import uuid
from copy import deepcopy
from typing import Dict, List, Optional, Union
from relevanceai.dataset.write import Write
from relevanceai.constants.errors import FieldNotFoundError

class CreateApps(Write):
    """
    Set of core functions used to do CRUD on deployables/apps.
    Config = Universal config in Relevance AI for apps
    Config Input = The input to the SDK's create_app_config
    """
    def list_apps(self, return_config:bool=False):
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

    def create_app(self, config):
        result = self.deployables.create(dataset_id=self.dataset_id, configuration=config)
        if config['type'] == "explore":
            print(f"Your app can be accessed at: https://cloud.relevance.ai/dataset/{result['dataset_id']}/deploy/explore/{result['project_id']}/{self.api_key}/{result['deployable_id']}/{self.region}")
        elif config['type'] == "page":
            print(f"Your app can be accessed at: https://cloud.relevance.ai/dataset/{result['dataset_id']}/deploy/page/{result['project_id']}/{self.api_key}/{result['deployable_id']}/{self.region}")
        return result

    def update_app(self, deployable_id, config):
        return self.deployables.update(deployable_id=deployable_id, dataset_id=self.dataset_id, config=config)

    def delete_app(self, deployable_id):
        return self.deployables.delete(deployable_id=deployable_id)

    def get_app(self, deployable_id):
        return self.deployables.get(deployable_id=deployable_id)

    def get_app_ids_by_name(self, name):
        ids = []
        for a in self.list_apps():
            if "configuration" in a:
                if "deployable_name" in a["configuration"] and a['configuration']['deployable_name'] == name:
                    ids.append(a['deployable_id'])
        return ids

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
            if sort in [m['name'] for m in metrics]:
                chart_config["metric-to-sort-by"] = sort
            else:
                raise KeyError("'sort' used in a chart is not in 'metrics'")
        if page_size:
            chart_config["page-size"] = page_size
        return chart_config

    def create_app_config(
        self, 
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
        **kwargs
        ):
        
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
                    self._agg_to_chart_config(**charts)
                ]
            elif isinstance(charts, list):
                configuration["aggregation-charts"] = [
                    self._agg_to_chart_config(**c)
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

    def update_config_inputs(self, inputs, updates):
        """
        A way to combine multiple templates.
        """
        inputs = deepcopy(inputs)
        input_fields_that_append = ["charts", "preview_fields", "facets", "sort"]
        for f in input_fields_that_append:
            if f in updates:
                inputs[f] += [u for u in updates[f] if u not in inputs[f]]
        inputs.update({k:v for k,v in updates.items() if k not in input_fields_that_append})
        return inputs

    def _auto_detect_vector_fields(self, fields, vector_fields="auto", field_type="text"):
        if vector_fields == "auto":
            vector_fields = []
            print(f'Detected "{field_type}_vector_fields" is set as "auto", will try to determine "{field_type}_vector_fields" from "{field_type}_fields"')
            for field, field_type in self.schema.items():
                if isinstance(field_type, dict):
                    for f in fields:
                        if f in field:
                            vector_fields.append(field)
            print(f'The detected vector fields are {str(vector_fields)}, manually specify the `{field_type}_vector_fields` if those are incorrect.')
            if not vector_fields:
                raise(f'No vector fields associated with the given {field_type} fields were found, run `ds.vectorize_{field_type}({field_type}_fields={str(fields)})` to extract vectors for your {field_type} fields.')
        return vector_fields

    def _clean_metrics(self, metrics):
        main_metrics = []
        metric_fields = []
        metric_names = []
        for m in metrics:
            if isinstance(m, str):
                main_metrics.append({"agg" : "avg", "field": m, "name" : f"Average {m}"})
                metric_fields.append(m)
                metric_names.append(f"Average {m}")
            else:
                main_metrics.append(m)
                metric_fields.append(m['field'])
                metric_names.append(m['name'])
        return main_metrics, metric_fields, metric_names

    def _clean_groupby(self, groupby):
        main_groupby = []
        groupby_fields = []
        for m in groupby:
            if isinstance(m, str):
                if self.schema[m] == "text":
                    main_groupby.append({
                        "agg" : "category", "field": m, "name" : f"{m}"
                    })
                elif self.schema[m] == "numeric":
                    main_groupby.append({
                        "agg" : "numeric", "field": m, "name" : f"{m}"
                    })
                groupby_fields.append(m)
            else:
                main_groupby.append(m)
                groupby_fields.append(m['field'])
        return main_groupby, groupby_fields

    def create_report_app_config(self, report):
        return {
            "dataset_name" : self.dataset_id,
            "deployable_name" : report.name,
            "type":"page", 
            "page-content" : {
                "type" :"doc",
                "content" : report.app
            }
        }
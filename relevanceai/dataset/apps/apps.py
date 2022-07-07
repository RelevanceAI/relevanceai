from relevanceai._api import APIClient

class AppsDeployables(APIClient):
    def list_apps(self, return_configs=False):
        print("Note: Deployable is the same as App. Deployables are legacy names of what we call Apps in the backend.")
        if return_configs:
            return [
                d
                for d in self.deployables.list()["deployables"]
                if d["dataset_id"] == self.dataset_id
            ]
        else:
            return [
                {
                    "deployable_id" : d["deployable_id"],
                    "deployable_name" : d["configuration"]['deployable_name'], 
                    "url" : f"https://cloud.relevance.ai/dataset/{d['dataset_id']}/deploy/explore/{d['project_id']}/{self.api_key}/{d['deployable_id']}/{self.region}"
                } 
                for d in self.deployables.list()["deployables"]
                if d["dataset_id"] == self.dataset_id and "configuration" in d and "deployable_name" in d["configuration"]
            ]

    def create_app_config(self, 
        app_name="",
        sort_default=None,
        sort_default_direction=None, 
        sort=[],
        filters=[], 
        search_fields=[],
        vector_search_fields=[],
        charts=[],
        cluster_charts=[],
        cluster_preview_fields=[],
        **kwargs):
        
        configuration = {
            "deployable_name" : app_name,
            "type":"explore", 
            **kwargs
        }
        configuration["filter-facets"] = filters
        configuration["default-preview-centroids-columns"] = cluster_preview_fields
        if sort:
            configuration["key-metrics"] = sort

        if sort_default:
            configuration["sort-default"] = sort_default
            configuration["sort-default-direction"] = sort_default_direction
            if sort_default_direction:
                configuration["sort-default-direction"] = sort_default_direction

        if charts:
            configuration["aggregation-charts"] = charts

        if cluster_charts:
            configuration["cluster-preview-configuration"] = cluster_charts
        elif charts:
            configuration["cluster-preview-configuration"] = charts

        if search_fields:
            configuration["text-fields-to-search"] = search_fields
        if vector_search_fields:
            configuration["vector-fields-to-search"] = vector_search_fields
        return configuration

    def create_app(self, config):
        return self.deployables.create(dataset_id=self.dataset_id, configuration=config)

    def update_app(self, deployable_id, config):
        return self.deployables.update(deployable_id=deployable_id, dataset_id=self.dataset_id, config=config)

    def delete_app(self, deployable_id):
        return self.deployables.delete(deployable_id=deployable_id)

    def get_app(self, deployable_id):
        return self.deployables.get(deployable_id=deployable_id)
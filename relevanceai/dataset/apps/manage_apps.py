import uuid
from copy import deepcopy
from typing import Dict, List, Optional, Union
from relevanceai.dataset.write import Write
from relevanceai.constants.errors import FieldNotFoundError
from relevanceai.apps import ExplorerApp


class ManageApps(Write):
    """
    Set of core functions used to do CRUD on deployables/apps.
    Config = Universal config in Relevance AI for apps
    Config Input = The input to the SDK's create_app_config
    """

    def _app_url(
        self, dataset_id: str, project_id: str, deployable_id: str, app_type=None
    ):
        if app_type:
            return f"https://cloud.relevance.ai/dataset/{dataset_id}/deploy/{app_type}/{project_id}/{self.api_key}/{deployable_id}/{self.region}"
        else:
            return f"https://cloud.relevance.ai/dataset/{dataset_id}/deploy/explore/{project_id}/{self.api_key}/{deployable_id}/{self.region}"

    def list_apps(self, return_config: bool = False):
        # print(
        #     "Note: Deployable is the same as App. Deployables are legacy names of what we call Apps in the backend."
        # )
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
                        "deployable_id": d["deployable_id"],
                    }
                    if "configuration" in d:
                        if "deployable_name" in d["configuration"]:
                            result["deployable_name"] = d["configuration"][
                                "deployable_name"
                            ]
                        if "type" in d["configuration"]:
                            result["type"] = d["configuration"]["type"]
                            result["url"] = self._app_url(
                                d["dataset_id"],
                                d["project_id"],
                                d["configuration"]["type"],
                                d["deployable_id"],
                            )
                    results.append(result)
            return results

    def create_app(self, config: Dict, verbose: bool = True):
        result = self.deployables.create(
            dataset_id=self.dataset_id, configuration=self.json_encoder(config)
        )
        if verbose:
            import click

            click.secho(" You can now view your app in your browser.")
            click.secho(
                f""" {self._app_url(
                    dataset_id=result['dataset_id'],
                    project_id=result['project_id'],
                    deployable_id=result['deployable_id'],
                    app_type=result['configuration']['type']
                )}""",
                bold=True,
            )
        return result

    def update_app(
        self,
        deployable_id: str,
        config: Dict,
        overwrite: bool = False,
        verbose: bool = True,
    ):
        status = self.deployables.update(
            deployable_id=deployable_id,
            dataset_id=self.dataset_id,
            configuration=self.json_encoder(config),
            overwrite=overwrite,
        )
        if status["status"] == "success":
            if "type" in config:
                app_type = config["type"]
            elif "type" in config["configuration"]:
                app_type = config["configuration"]["type"]
            if verbose:
                import click

                click.secho(" You can now view your app in your browser.")
                click.secho(
                    f""" {self._app_url(
                        dataset_id=self.dataset_id,
                        project_id=self.project,
                        deployable_id=deployable_id,
                        app_type=app_type
                    )}""",
                    bold=True,
                )
        else:
            print("Failed to update")
        return status

    def delete_app(self, deployable_id: str):
        return self.deployables.delete(deployable_id=deployable_id)

    def get_app(self, deployable_id: str):
        return self.deployables.get(deployable_id=deployable_id)

    def get_app_ids_by_name(self, name: str):
        ids = []
        for a in self.list_apps():
            if "configuration" in a:
                if (
                    "deployable_name" in a["configuration"]
                    and a["configuration"]["deployable_name"] == name
                ):
                    ids.append(a["deployable_id"])
        return ids

    def create_app_config(
        self,
        name: str = "App",
        default_view: str = "charts",
        preview_fields: List[str] = None,
        facets: List[str] = None,
        search_fields: List[str] = None,
        vector_search_fields: List[str] = None,
        search_min_relevance: float = None,
        sort: Union[List[str], List[dict]] = None,
        sort_default: str = None,
        sort_default_direction: str = None,
        charts: Union[List[str], List[dict]] = None,
        cluster_alias: str = None,
        cluster_vector_field: str = None,
        cluster_charts: Union[List[str], List[dict]] = None,
        charts_view_column: int = 2,
        preview_centroids_page_size: int = 2,
        **kwargs,
    ):
        """
        This is an easy wrapper for creating explorer app
        """
        eapp = ExplorerApp(
            name=name,
            dataset=self,
            default_view=default_view,
            charts_view_column=charts_view_column,
            preview_centroids_page_size=preview_centroids_page_size,
        )
        if preview_fields:
            eapp.preview_fields(fields=preview_fields)

        # Filter/Drilldown bar
        if facets:
            eapp.facets(fields=facets)
        if search_fields:  # make sure its text
            eapp.search_fields(fields=search_fields)
        if vector_search_fields:
            eapp.vector_search_fields(vector_fields=vector_search_fields)
        if search_min_relevance:
            eapp.search_min_relevance(min_relevance=search_min_relevance)
        if sort:
            eapp.sort(metrics=sort)
        if sort_default:
            eapp.sort_default(metric_name=sort_default)

        # Charts section
        if charts:
            eapp.charts(charts=charts)

        # Cluster section
        if cluster_alias and cluster_vector_field:
            eapp.cluster(alias=cluster_alias, vector_field=cluster_vector_field)
        elif (not cluster_alias and cluster_vector_field) or (
            cluster_alias and not cluster_vector_field
        ):
            raise KeyError(
                "Require both 'cluster_alias' and 'cluster_vector_field' for setting cluster explorer."
            )
        if cluster_charts:
            eapp.cluster_charts(charts=cluster_charts)
        return eapp.config

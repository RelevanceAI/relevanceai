from typing import Dict, Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base


class DeployableClient(_Base):
    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    def create_deployable_configuration(self, 
        deployable_name="",
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
            "deployable_name" : deployable_name,
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

    def create(self, dataset_id: str, configuration: Optional[Dict] = None):
        """
        Create a private deployable from an existing dataset.
        TODO: explain what a deployable is

        Parameters
        ------------
        dataset_id: string
            Unique name of dataset
        configuration: None | dict
            A configuration specification

        Returns
        ---------
        On success (200): {
            deployable_id, dataset_id, project_id, api_key, configuration
        }
        On failure (422): {
            loc, msg, type
        }
        """
        if configuration is None:
            configuration = {}

        return self.make_http_request(
            endpoint="/deployables/create",
            method="POST",
            parameters={"dataset_id": dataset_id, "configuration": configuration},
        )

    def share(self, deployable_id: str):
        """
        Share a private deployable.
        The response should be {"status": "success"} or {"status": "failed"}

        Parameters
        ----------
        deployable_id: string
            Unique name of deployable

        Returns
        -------
        On success (200): {
            configuration, status, message
        }
        On failure (422): {
            loc, msg, type
        }
        """
        return self.make_http_request(
            endpoint=f"/deployables/{deployable_id}/share", method="POST"
        )

    def unshare(self, deployable_id: str):
        """
        Unshare a shared deployable, making it private.
        The response should be {"status": "success"} or {"status": "failed"}

        Parameters
        ----------
        deployable_id: string
            Unique name of deployable

        Returns
        -------
        On success (200): {
            configuration, status, message
        }
        On failure (422): {
            loc, msg, type
        }
        """
        return self.make_http_request(
            endpoint=f"/deployables/{deployable_id}/private", method="POST"
        )

    def update(
        self,
        deployable_id: str,
        dataset_id: str,
        configuration: Optional[Dict] = None,
        overwrite: bool = True,
        upsert: bool = True,
    ):
        """
        Update a specified deployable.

        Parameters
        ----------
        deployable_id: string
            The deployable configuration
        dataset_id: string
            Unique name of dataset.
        configuration: None | dict
            The deployable configuration
        overwrite: Boolean
            Whether to overwrite document if it exists
        upsert: Boolean
            If True, adds new fields. This is prioritized over overwrite.

        Returns
        -------
        On success (200): {
            configuration, status, message
        }
        On failure (422): {
            loc, msg, type
        }
        """
        return self.make_http_request(
            endpoint=f"/deployables/{deployable_id}/update",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "configuration": configuration,
                "overwrite": overwrite,
                "upsert": upsert,
            },
        )

    def get(self, deployable_id: str):
        """
        Get a specified deployable.

        Parameters
        ----------
        deployable_id: string
            The deployable configuration

        Returns
        -------
        On success (200): {
            project_id, dataset_id, api_key, configuration
        }
        On failure (422): {
            loc, msg, type
        }
        """
        return self.make_http_request(
            endpoint=f"/deployables/{deployable_id}/get", method="GET"
        )

    def delete(self, deployable_id: str):
        """
        Delete a specified deployable.

        Parameters
        ----------
        deployable_id: string
            The deployable configuration

        Returns
        -------
        On success (200): {
            status, message
        }
        On failure (422): {
            loc, msg, type
        }
        """
        return self.make_http_request(
            endpoint=f"/deployables/delete",
            method="POST",
            parameters={
                "id": deployable_id,
            },
        )

    def list(self):
        """
        List all deployables.

        Parameters
        ----------
            None

        Returns
        -------
        On success (200): {
            deployables, count
        }
        On failure (422): {
            loc, msg, type
        }
        """
        return self.make_http_request(endpoint="/deployables/list", method="GET")

    def url(self, deployable_id: str, dataset_id: str, application: str) -> str:
        """
        Generates the deployable URL.

        Parameters
        ----------
        deployable_id: str
            ID of the deployable
        dataset_id: str
            ID of the dataset
        application: str
            The type of deployable application
        """
        url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
        return url.format(
            dataset_id, self.project, application, self.api_key, deployable_id
        )

from typing import Optional
from relevanceai.client.helpers import Credentials
from relevanceai.dashboard.dashboard import Dashboard


class Clusters(Dashboard):
    def __init__(self, credentials: Credentials, project: str):
        super().__init__(credentials, project, "cluster")

    @classmethod
    def create_dashboard(
        cls,
        credentials: Credentials,
        dataset_id: str,
        vector_field: str,
        alias: str,
        share: bool = False,
        **configuration
    ):
        # TODO: if there is no _cluster_ field in schema, create it here?
        application = "cluster"
        application_configuration = {
            "collection_name": dataset_id,
            "type": application,
            "deployable_name": dataset_id,
            "project_id": credentials.project,
            application: {
                "alias": alias,
                "vector_field": vector_field,
                **{
                    key: value
                    for key, value in configuration.items()
                    if key not in {"alias", "vector_field"}
                },
            },
        }

        return Dashboard.create_dashboard(
            credentials,
            dataset_id,
            vector_field,
            application,
            share,
            application_configuration,
        )

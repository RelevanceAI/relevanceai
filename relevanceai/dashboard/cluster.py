from typing import Optional
from relevanceai.dashboard.dashboard import Dashboard


class Clusters(Dashboard):
    def __init__(self, project: str, api_key: str, deployable_id: str, firebase_uid):
        super().__init__(project, api_key, deployable_id, "cluster", firebase_uid)

    @classmethod
    def create_dashboard(
        cls,
        project: str,
        api_key: str,
        dataset_id: str,
        vector_field: str,
        alias: str,
        share: bool = False,
        firebase_uid: Optional[str] = None,
        **configuration
    ):
        # TODO: if there is no _cluster_ field in schema, create it here?
        application = "cluster"
        application_configuration = {
            "collection_name": dataset_id,
            "type": application,
            "deployable_name": dataset_id,
            "project_id": project,
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
            project,
            api_key,
            dataset_id,
            vector_field,
            application,
            share,
            application_configuration,
            firebase_uid,  # type: ignore
        )

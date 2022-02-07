from relevanceai.dashboard.dashboard import Dashboard
from relevanceai.api.endpoints.deployables.deployables import Deployable


class Clusters(Dashboard):
    def __init__(self, project: str, api_key: str, deployable_id: str):
        super().__init__(project, api_key, deployable_id, "cluster")

    @classmethod
    def create_application(
        cls,
        project: str,
        api_key: str,
        dataset_id: str,
        vector_field: str,
        share: bool = False,
        **configuration
    ):
        # TODO: if there is no _cluster_ field in schema, create it here?
        return super().create_application(
            project,
            api_key,
            dataset_id,
            vector_field,
            "cluster",
            share,
            **configuration
        )

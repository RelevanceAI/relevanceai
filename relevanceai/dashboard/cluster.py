from relevanceai.dashboard.dashboard import Dashboard
from relevanceai.api.endpoints.deployables.deployables import Deployable


class Clusters(Dashboard):
    def __init__(self, project: str, api_key: str, deployable_id: str):
        super().__init__(project, api_key, deployable_id, "cluster")

    @property
    def deployable_url(self):
        deployables = Deployable(self.project, self.api_key)
        configuration = deployables.get(self.deployable_id)["configuration"]
        url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
        return url.format(
            configuration["dataset_id"],
            self.project,
            "cluster",
            self.api_key,
            self.deployable_id,
        )

    @property
    def shareable_url(self):
        if self.shareable_id is None:
            # TODO: write exception
            raise Exception
        else:
            deployables = Deployable(self.project, self.api_key)
            configuration = deployables.get(self.deployable_id)["configuration"]
            url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
            return url.format(
                configuration["dataset_id"],
                self.project,
                "cluster",
                self.share_key,
                self.deployable_id,
            )

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
        return super().create_application(
            project,
            api_key,
            dataset_id,
            vector_field,
            "cluster",
            share,
            **configuration
        )

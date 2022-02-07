from abc import ABC, abstractmethod

from relevanceai.api.endpoints.datasets.datasets import DatasetsClient
from relevanceai.api.endpoints.deployables.deployables import Deployable
from relevanceai.base import _Base


class Dashboard(ABC, _Base):
    def __init__(
        self, project: str, api_key: str, deployable_id: str, application: str
    ):
        deployables = Deployable(project, api_key)
        if deployable_id not in deployables.list():
            raise ValueError(f"No deployable with ID {deployable_id}")
        else:
            configuration = deployables.get(deployable_id)["configuration"]
            if not application == configuration["type"]:
                raise ValueError(f"{deployable_id} is not a {application} application")
            else:
                self.vector_field = configuration["vector_field"]

        super().__init__(project=project, api_key=api_key)
        self.project = project
        self.api_key = api_key
        self.deployable_id = deployable_id
        self.shareable_id = None

    def share_application(self):
        if self.shareable_id is None:
            deployables = Deployable(self.project, self.api_key)
            deployables.share(self.deployable_id)
            response = deployables.get(self.deployable_id)
            self.share_key = response.json()["api_key"]
        else:
            raise Exception("Dashboard is already shareable")

    def unshare_application(self):
        if self.shareable_id is None:
            raise Exception("Dashboard is already unshareable")
        else:
            deployables = Deployable(self.project, self.api_key)
            deployables.unshare(self.deployable_id)
            self.share_key = None

    @classmethod
    def create_application(
        cls,
        project: str,
        api_key: str,
        dataset_id: str,
        vector_field: str,
        application_type: str,
        share: bool = False,
        **configuration,
    ):
        # Validation phase
        schema = DatasetsClient(project, api_key).schema(dataset_id)
        try:
            vector_field_type = schema[vector_field]
            # Since vectors are the only schema types that are objects, this
            # should be a sufficient check for now.
            if not isinstance(vector_field_type, dict):
                raise ValueError(f"{vector_field} is not a vector")
        except KeyError:
            raise ValueError(f"{vector_field} is not a field of {dataset_id}")

        # Creation phase
        deployables = Deployable(project, api_key)
        # TODO: Check if the deployable already exists?
        response = deployables.create(
            dataset_id,
            configuration={
                "type": application_type,
                "vector_field": vector_field,
                **{
                    key: value
                    for key, value in configuration.items()
                    if key not in {"type", "vector_field"}
                },
            },
        )
        deployable_id = response.json()["deployable_id"]

        application = cls(project, api_key, deployable_id, application_type)
        if share:
            application.share_application()

        return application

    @property
    @abstractmethod
    def deployable_url(self):
        pass

    @property
    @abstractmethod
    def shareable_url(self):
        pass

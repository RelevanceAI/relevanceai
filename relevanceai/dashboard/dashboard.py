from abc import ABC, abstractmethod

from relevanceai.api.endpoints.datasets.datasets import DatasetsClient
from relevanceai.api.endpoints.deployables.deployables import Deployable
from relevanceai.base import _Base


class Dashboard(ABC, _Base):
    def __init__(
        self, project: str, api_key: str, deployable_id: str, application: str
    ):
        valid_applications = {"cluster"}
        if application not in valid_applications:
            raise ValueError(
                f"{application} is not a valid application. "
                + "Must be one of the following: "
                + f"{', '.join(valid_applications)}"
            )

        deployables = Deployable(project, api_key)
        if deployable_id not in deployables.list():
            raise ValueError(f"No deployable with ID {deployable_id}")
        else:
            configuration = deployables.get(deployable_id)["configuration"]
            if not application == configuration["type"]:
                raise ValueError(f"{deployable_id} is not a {application} application")
            else:
                self.vector_field = configuration["vector_field"]

        super().__init__(project, api_key)
        self.deployable_id = deployable_id

        self._project = project
        self._api_key = api_key
        self._shareable_id = None
        self._application = application

    def share_application(self):
        if self._shareable_id is None:
            deployables = Deployable(self._project, self._api_key)
            deployables.share(self.deployable_id)
            response = deployables.get(self.deployable_id)
            self._shareable_id = response.json()["api_key"]
        else:
            raise Exception("Dashboard is already shareable")

    def unshare_application(self):
        if self._shareable_id is None:
            raise Exception("Dashboard is already unshareable")
        else:
            deployables = Deployable(self._project, self._api_key)
            deployables.unshare(self.deployable_id)
            self._shareable_id = None

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
        # TODO: Should we if the deployable already exists? I think I would
        # need to check the cluster/field/alias combo
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
    def deployable_url(self):
        deployables = Deployable(self.project, self.api_key)
        configuration = deployables.get(self.deployable_id)["configuration"]
        url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
        return url.format(
            configuration["dataset_id"],
            self._project,
            self._application,
            self._api_key,
            self.deployable_id,
        )

    @property
    def shareable_url(self):
        if self.shareable_id is None:
            raise Exception(f"This {self._application} application is not shareable")
        else:
            deployables = Deployable(self.project, self.api_key)
            configuration = deployables.get(self.deployable_id)["configuration"]
            url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
            return url.format(
                configuration["dataset_id"],
                self._project,
                self._application,
                self._shareable_id,
                self.deployable_id,
            )

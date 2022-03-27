import webbrowser

from abc import ABC
from typing import Optional

from relevanceai._api.endpoints.datasets.datasets import DatasetsClient
from relevanceai._api.endpoints.deployables.deployables import DeployableClient
from relevanceai.utils.base import _Base


class Dashboard(ABC, _Base):
    def __init__(
        self,
        project: str,
        api_key: str,
        deployable_id: str,
        application: str,
        firebase_uid: str,
    ):
        valid_applications = {"cluster"}
        if application not in valid_applications:
            raise ValueError(
                f"{application} is not a valid application. "
                + "Must be one of the following: "
                + f"{', '.join(valid_applications)}"
            )

        deployables = DeployableClient(project, api_key, firebase_uid)
        deployables_ids = map(lambda d: d["_id"], deployables.list()["deployables"])
        if deployable_id not in deployables_ids:
            raise ValueError(f"No deployable with ID {deployable_id}")
        else:
            configuration = deployables.get(deployable_id)["configuration"]
            if not application == configuration["type"]:
                raise ValueError(f"{deployable_id} is not a {application} application")
            else:
                self.vector_field = configuration[application]["vector_field"]

        super().__init__(project, api_key, firebase_uid)
        self.deployable_id = deployable_id

        self._shareable_id = None
        self._application = application

    def share_application(self) -> None:
        if self._shareable_id is None:
            deployables = DeployableClient(
                self.project, self.api_key, self.firebase_uid
            )
            deployables.share(self.deployable_id)
            response = deployables.get(self.deployable_id)
            self._shareable_id = response["api_key"]
        else:
            raise Exception("Dashboard is already shareable")

    def unshare_application(self) -> None:
        if self._shareable_id is None:
            raise Exception("Dashboard is already unshareable")
        else:
            deployables = DeployableClient(
                self.project, self.api_key, self.firebase_uid
            )
            deployables.unshare(self.deployable_id)
            self._shareable_id = None

    @classmethod
    def create_dashboard(
        cls,
        project: str,
        api_key: str,
        dataset_id: str,
        vector_field: str,
        application: str,
        share: bool,
        application_configuration: dict,
        firebase_uid: str,
    ):
        # Validation phase
        schema = DatasetsClient(project, api_key, firebase_uid).schema(dataset_id)
        try:
            vector_field_type = schema[vector_field]
            # Since vectors are the only schema types that are objects, this
            # should be a sufficient check for now.
            if not isinstance(vector_field_type, dict):
                raise ValueError(f"{vector_field} is not a vector")
        except KeyError:
            raise ValueError(f"{vector_field} is not a field of {dataset_id}")

        # Creation phase
        deployables = DeployableClient(project, api_key, firebase_uid)
        # TODO: Should we if the deployable already exists? I think I would
        # need to check the cluster/field/alias combo
        response = deployables.create(
            dataset_id,
            configuration=application_configuration,
        )
        deployable_id = response["deployable_id"]

        dashboard = cls(project, api_key, deployable_id, application, firebase_uid)
        if share:
            dashboard.share_application()

        return dashboard

    @property
    def deployable_url(self) -> str:
        deployables = DeployableClient(self.project, self.api_key, self.firebase_uid)
        deployable = deployables.get(self.deployable_id)
        url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
        return url.format(
            deployable["dataset_id"],
            self.project,
            self._application,
            self.api_key,
            self.deployable_id,
        )

    @property
    def shareable_url(self) -> str:
        if self._shareable_id is None:
            raise Exception(f"This {self._application} application is not shareable")
        else:
            deployables = DeployableClient(
                self.project, self.api_key, self.firebase_uid
            )
            deployable = deployables.get(self.deployable_id)
            url = "https://cloud.relevance.ai/dataset/{}/deploy/{}/{}/{}/{}"
            return url.format(
                deployable["dataset_id"],
                self.project,
                self._application,
                self._shareable_id,
                self.deployable_id,
            )

    def view(self, shareable: bool = False) -> None:
        if shareable:
            url = self.shareable_url
        else:
            url = self.deployable_url

        print(f"Opening {url}...")
        webbrowser.open(url)

"""API Client
"""
from typing import Any, Dict, List, Optional
import uuid
from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base

from relevanceai._api.endpoints.admin.admin import AdminClient
from relevanceai._api.endpoints.datasets.datasets import DatasetsClient
from relevanceai._api.endpoints.services.services import ServicesClient
from relevanceai._api.endpoints.deployables.deployables import DeployableClient
from relevanceai._api.endpoints.workflows.workflows import WorkflowsClient

from relevanceai.constants.errors import FieldNotFoundError

from relevanceai.utils.datasets import ExampleDatasets
from relevanceai.utils import make_id
from relevanceai.utils import DocUtils


class APIEndpointsClient(_Base, DocUtils):
    """API Client"""

    def __init__(
        self,
        credentials: Credentials,
        **kwargs,
    ):
        self.credentials = credentials
        super().__init__(
            credentials=credentials,
            **kwargs,
        )

    # To avoid the inits being missed - we set the endpoints as
    # properties here
    # we set the clients as internal variables
    # so they do not have to be initiated every time

    @property
    def datasets(self):
        if hasattr(self, "_datasets_client"):
            return self._datasets_client
        else:
            self._datasets_client = DatasetsClient(self.credentials)
        return self._datasets_client

    @property
    def services(self):
        if hasattr(self, "_services_client"):
            return self._services_client
        else:
            self._services_client = ServicesClient(self.credentials)
        return self._services_client

    @property
    def example_datasets(self):
        return ExampleDatasets()

    @property
    def admin(self):
        if hasattr(self, "_admin_client"):
            return self._admin_client
        self._admin_client = AdminClient(self.credentials)
        return self._admin_client

    @property
    def deployables(self):
        if hasattr(self, "_deployables_client"):
            self._deployables_client = DeployableClient(self.credentials)
        self._deployables_client = DeployableClient(self.credentials)
        return self._deployables_client

    @property
    def workflows(self):
        if hasattr(self, "_workflows_client"):
            self._workflows_client = WorkflowsClient(self.credentials)
        self._workflows_client = WorkflowsClient(self.credentials)
        return self._workflows_client

    def _validate_ids(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Not inplace anymore
        """
        return [
            {"_id": str(uuid.uuid4()), **document}
            if "_id" not in document
            else document
            for document in documents
        ]

    def _are_fields_in_schema(
        self,
        fields: List[str],
        dataset_id: str,
        schema: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Check fields are in schema
        """
        if schema is None:
            schema = self.datasets.schema(dataset_id)
        invalid_fields = []
        for i in fields:
            if i not in schema:
                invalid_fields.append(i)
        if len(invalid_fields) > 0:
            raise ValueError(
                f"{', '.join(invalid_fields)} are invalid fields. They are not in the dataset schema."
            )
        return

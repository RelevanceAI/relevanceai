"""API Client
"""
from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base

from relevanceai._api.endpoints.admin.admin import AdminClient
from relevanceai._api.endpoints.datasets.datasets import DatasetsClient
from relevanceai._api.endpoints.services.services import ServicesClient
from relevanceai._api.endpoints.reports.reports import ReportsClient
from relevanceai._api.endpoints.deployables.deployables import DeployableClient

from relevanceai.constants.errors import FieldNotFoundError

from relevanceai.utils.datasets import ExampleDatasets
from relevanceai.utils import make_id

from doc_utils import DocUtils


class APIEndpointsClient(_Base, DocUtils):
    """API Client"""

    def __init__(
        self,
        credentials: Credentials,
        **kwargs,
    ):
        self.datasets = DatasetsClient(credentials)
        self.services = ServicesClient(credentials)
        self.example_datasets = ExampleDatasets()
        self.admin = AdminClient(credentials)
        self.reports = ReportsClient(credentials)
        self.deployables = DeployableClient(credentials)
        super().__init__(
            credentials=credentials,
            **kwargs,
        )

    def _convert_id_to_string(self, documents, create_id: bool = False):
        try:
            self.set_field_across_documents(
                "_id", [str(i["_id"]) for i in documents], documents
            )
        except KeyError:
            if create_id:
                self.set_field_across_documents(
                    "_id", [make_id(document) for document in documents], documents
                )
            else:
                raise FieldNotFoundError(
                    "Missing _id field. Set `create_id=True` to automatically generate IDs."
                )

    def _are_fields_in_schema(self, fields, dataset_id, schema=None):
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

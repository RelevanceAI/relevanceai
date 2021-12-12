"""API Client
"""
from relevanceai.base import _Base
from relevanceai.api.endpoints.admin import AdminClient
from relevanceai.api.endpoints.datasets import DatasetsClient
from relevanceai.api.endpoints.services import ServicesClient
from relevanceai.datasets import ExampleDatasets


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class APIClient(_Base):
    """API Client"""

    def __init__(self, project: str, api_key: str):
        self.datasets = DatasetsClient(project=project, api_key=api_key)
        self.services = ServicesClient(project=project, api_key=api_key)
        self.example_datasets = ExampleDatasets()
        self.admin = AdminClient(project=project, api_key=api_key)
        super().__init__(project, api_key)

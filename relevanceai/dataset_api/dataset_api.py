"""
Pandas like dataset API
"""
from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_export import Export
from relevanceai.dataset_api.dataset_stats import Stats
from relevanceai.dataset_api.dataset_operations import Operations


class Dataset(Export, Stats, Operations):
    """Dataset class"""


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project=project, api_key=api_key)

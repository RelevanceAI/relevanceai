"""
Pandas like dataset API
"""

from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_export import Export
from relevanceai.dataset_api.dataset_stats import Stats
from relevanceai.dataset_api.dataset_operations import Operations


class Dataset(Export, Stats, Operations):
    """Dataset class"""

    pass


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project=project, api_key=api_key)

    def __getitem__(self, field):
        """
        Returns a Series Object that selects a particular field within a dataset

        Parameters
        ----------
        field
            the particular field within the dataset

        Returns
        -------
        Tuple
            (N, C)

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            field = "sample_field"
            series = df[field]
        """
        return Series(self.project, self.api_key, self.dataset_id, field)

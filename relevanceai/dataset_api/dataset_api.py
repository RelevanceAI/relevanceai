"""
Pandas like dataset API
"""
from typing import Union, List

from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_export import Export
from relevanceai.dataset_api.dataset_stats import Stats
from relevanceai.dataset_api.dataset_operations import Operations, LabelExperiment
from relevanceai.dataset_api.dataset_series import Series
from relevanceai.dataset_api.dataset_search import Search
from relevanceai.dataset_api.dataset_dr import DR


class Dataset(Export, Stats, LabelExperiment, DR):
    """Dataset class"""

    def __init__(self, project: str, api_key: str, dataset_id: str, fields: list = []):
        self.project = project
        self.api_key = api_key
        self.fields = fields
        self.dataset_id = dataset_id
        super().__init__(
            project=project, api_key=api_key, fields=fields, dataset_id=dataset_id
        )
        self.search = Search(
            project=project, api_key=api_key, fields=fields, dataset_id=dataset_id
        )

    def __getitem__(self, field: Union[List[str], str]):
        """
        Returns a Series Object that selects a particular field within a dataset

        Parameters
        ----------
        field: Union[List, str]
            The particular field within the dataset

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
        if isinstance(field, str):
            return Series(self.project, self.api_key, self.dataset_id, field)
        elif isinstance(field, list):
            return Dataset(
                project=self.project,
                api_key=self.api_key,
                dataset_id=self.dataset_id,
                fields=field,
            )
        else:
            raise TypeError("Field needs to be a list or a string.")


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project=project, api_key=api_key)

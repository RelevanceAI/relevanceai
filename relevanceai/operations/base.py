"""
Base class for operations
"""
from typing import Any, List

from relevanceai.client.helpers import Credentials


class BaseOps:
    """
    Base class for operations
    """

    def init(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)

    def from_credentials(self, credentials: Credentials):
        kwargs = dict(credentials=credentials)
        self.init(**kwargs)

    def from_token(self, token):
        kwargs = dict(token=token)
        self.init(**kwargs)

    def from_details(self, project: str, api_key: str, region: str):
        kwargs = dict(
            project=project,
            api_key=api_key,
            region=region,
        )
        self.init(**kwargs)

    def from_dataset(self, dataset: Any, alias: str, vector_fields: List[str]):
        if isinstance(dataset, str):
            dataset_id = dataset
        else:
            dataset_id = dataset.dataset_id

        kwargs = dict(
            dataset_id=dataset_id,
            alias=alias,
            vector_fields=vector_fields,
        )
        self.init(**kwargs)

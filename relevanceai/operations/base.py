"""
Base class for operations.
"""
from typing import Any, List
from relevanceai.client.helpers import Credentials


class BaseOps:
    """
    Base class for operations
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def init(self, **kwargs):
        return BaseOps(kwargs)

    @classmethod
    def from_credentials(self, credentials: Credentials):
        raise NotImplementedError

    @classmethod
    def from_token(self, token: str):
        """
        If this is from a token, then we use this
        """
        # process the token here using client creds
        raise NotImplementedError

    @classmethod
    def from_details(self, project: str, api_key: str, region: str):
        """
        Use this if you are going to instantiate from details
        """
        kwargs = dict(
            project=project,
            api_key=api_key,
            region=region,
        )
        return BaseOps(**kwargs)

    @classmethod
    def from_client(self, client):
        raise NotImplementedError

    @classmethod
    def from_dataset(self, dataset: Any, alias: str, vector_fields: List[str]):
        """
        Instantiate operations from the workflows.
        """
        raise NotImplementedError

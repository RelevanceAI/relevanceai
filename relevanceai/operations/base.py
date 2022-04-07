"""
Base class for operations.
"""
from typing import Any, List, Optional
from relevanceai.client.helpers import (
    Credentials,
    process_token,
)


class BaseOps:
    """
    Base class for operations
    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def init(self, *args, **kwargs):
        return self(*args, **kwargs)

    @classmethod
    def from_credentials(self, credentials: Credentials):
        return self(credentials=credentials)

    @classmethod
    def from_token(self, token: str):
        """
        If this is from a token, then we use this
        """
        credentials = process_token(token)
        return self(credentials=credentials)

    @classmethod
    def from_client(self, client, *args, **kwargs):
        credentials = client.credentials
        return self(
            credentials=credentials,
            *args,
            **kwargs,
        )

    @classmethod
    def from_dataset(
        self,
        credentials: Credentials,
        dataset_id: str,
        alias: Optional[str],
        vector_fields: Optional[List[str]],
        *args,
        **kwargs,
    ):
        return self(
            credentials=credentials,
            dataset_id=dataset_id,
            alias=alias,
            vector_fields=vector_fields,
            *args,
            **kwargs,
        )

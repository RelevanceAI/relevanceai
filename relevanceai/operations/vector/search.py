"""Search In A Dataset
"""
from typing import List, Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.decorators.analytics import track
from relevanceai._api import APIClient


class SearchOps(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
    ):
        self.dataset_id = dataset_id
        super().__init__(credentials)

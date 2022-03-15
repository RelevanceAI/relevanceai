"""
All read operations for Dataset
"""
import re
import math
import warnings
import pandas as pd
from relevanceai.package_utils.cache import lru_cache
from typing import Dict, List, Optional, Union

from relevanceai.package_utils.analytics_funcs import track
from relevanceai.dataset.crud.helpers import _build_filters
from relevanceai.dataset.crud.groupby import Groupby, Agg
from relevanceai.vector_tools.client import VectorTools
from relevanceai.api.client import BatchAPIClient
from relevanceai.package_utils.constants import MAX_CACHESIZE
from relevanceai.package_utils.list_to_tuple import list_to_tuple
from relevanceai.workflows.cluster_ops.centroids import Centroids
from doc_utils import DocUtils
from relevanceai.package_utils.analytics_funcs import fire_and_forget


class _Metadata(BatchAPIClient):
    """Metadata object"""

    def __init__(self, metadata: dict, project, api_key, firebase_uid, dataset_id):
        self._metadata = metadata
        self.dataset_id = dataset_id
        super().__init__(project, api_key, firebase_uid)

    def __repr__(self):
        return str(self.to_dict())

    def insert_metadata(self, metadata: dict, verbose: bool = False):
        """Insert metadata"""
        results = self.datasets.post_metadata(self.dataset_id, metadata)
        if results == {}:
            return
        else:
            return results

    def upsert_metadata(self, metadata: dict, verbose: bool = False):
        """Upsert metadata."""
        original_metadata: dict = self.datasets.metadata(self.dataset_id)
        original_metadata.update(metadata)

        @fire_and_forget
        async def insert():
            return self.insert_metadata(metadata, verbose=verbose)

        insert()

    def __getitem__(self, key):
        return self.get_field(key, self._metadata)

    def __setitem__(self, key, value):
        self.set_field(key, self._metadata, value)
        self.upsert_metadata(self._metadata)

    def to_dict(self):
        return self._metadata

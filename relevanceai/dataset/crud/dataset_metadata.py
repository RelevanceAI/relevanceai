"""
All read operations for Dataset
"""
import re
import math
import warnings
import pandas as pd
from relevanceai.package_utils.cache import lru_cache
from typing import Dict, List, Optional, Union
from relevanceai.api.client import BatchAPIClient


class Metadata(BatchAPIClient):
    """Metadata object"""

    def __init__(self, metadata: dict, project, api_key, firebase_uid, dataset_id):
        self._metadata = metadata
        self.dataset_id = dataset_id
        super().__init__(project, api_key, firebase_uid)

    def __repr__(self):
        return str(self.to_dict())

    def __contains__(self, m):
        return m in self.to_dict().keys()

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

        # TODO: how do you fire and forget this
        return self.insert_metadata(metadata)

    def __getitem__(self, key):
        return self.get_field(key, self._metadata)

    def __setitem__(self, key, value):
        self.set_field(key, self._metadata, value)
        self.upsert_metadata(self._metadata)

    def to_dict(self):
        return self._metadata

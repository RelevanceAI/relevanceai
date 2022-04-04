# -*- coding: utf-8 -*-
"""Batch client to allow for batch insertions/retrieval and encoding
"""
from typing import Callable

from relevanceai._api.batch.insert import BatchInsertClient
from relevanceai._api.batch.insert_async import BatchInsertAsyncClient

from relevanceai.constants import CONFIG


class APIClient(BatchInsertClient, BatchInsertAsyncClient):
    """Batch API client"""

    def batch_insert(self):
        raise NotImplemented

    def batch_get_and_edit(self, dataset_id: str, chunk_size: int, bulk_edit: Callable):
        """Batch get the documents and return the documents"""
        raise NotImplemented

    @property
    def base_url(self):
        return CONFIG.get_field("api.base_url", CONFIG.config)

    @base_url.setter
    def base_url(self, value):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_url", value)

    @property
    def base_ingest_url(self):
        return CONFIG.get_field("api.base_ingest_url", CONFIG.config)

    @base_ingest_url.setter
    def base_ingest_url(self, value):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_ingest_url", value)

    # Other useful utilities
    def append_metadata_list(self, field: str, value_to_append, verbose: bool = True):
        """
        Easily append to a list value in metadata using this function.
        This prevents overwriting if there is one.
        """
        metadata = self.datasets.metadata(self.dataset_id)["results"]
        if self.is_field(field, metadata):
            value_list = self.get_field(field, metadata)
        else:
            value_list = []

        if not isinstance(value_list, list):
            raise ValueError("Can't append to object that is not a list.")
        value_list.append(value_to_append)
        self.set_field(field, metadata, value_list)
        self.datasets.post_metadata(
            dataset_id=self.dataset_id,
            metadata=metadata,
        )

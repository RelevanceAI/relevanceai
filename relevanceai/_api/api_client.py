# -*- coding: utf-8 -*-
"""Batch client to allow for batch insertions/retrieval and encoding
"""
from typing import Callable

from relevanceai._api.batch.insert import BatchInsertClient
from relevanceai._api.batch.insert_async import BatchInsertAsyncClient


class APIClient(BatchInsertClient, BatchInsertAsyncClient):
    """Batch API client"""

    def batch_insert(self):
        raise NotImplemented

    def batch_get_and_edit(self, dataset_id: str, chunk_size: int, bulk_edit: Callable):
        """Batch get the documents and return the documents"""
        raise NotImplemented

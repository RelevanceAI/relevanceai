"""Batch client to allow for batch insertions/retrieval and encoding
"""
from typing import Callable

from relevanceai.api.batch.batch_insert import BatchInsertClient
from relevanceai.api.batch.batch_retrieve import BatchRetrieveClient
from relevanceai.api.endpoints.client import APIClient
from relevanceai.api.batch.chunk import Chunker


class BatchAPIClient(BatchInsertClient, BatchRetrieveClient, APIClient):
    """Batch API client"""

    def batch_insert(self):
        raise NotImplemented

    def batch_get_and_edit(self, dataset_id: str, chunk_size: int, bulk_edit: Callable):
        """Batch get the documents and return the documents"""
        raise NotImplemented

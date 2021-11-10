"""Batch client to allow for batch insertions/retrieval and encoding
"""
from typing import Callable

from .batch_insert import BatchInsert
from .chunk import Chunker


class BatchAPIClient(BatchInsert):
    """Batch API client"""

    def batch_insert(self):
        raise NotImplemented

    def batch_get_and_edit(self, dataset_id: str, chunk_size: int, bulk_edit: Callable):
        """Batch get the documents and return the documents"""
        raise NotImplemented

"""Batch operations
"""
from typing import Callable
from ..api.client import APIClient

class BatchInsert(APIClient):
    def insert_documents(self, dataset_id: str, docs: list, bulk_encode: Callable=None, verbose: bool=True):
        for c in self.chunk(docs, chunk_size=20):
            # If you want to encode as you insert
            if bulk_encode is not None:
                bulk_encode(c)
            if verbose:
                print(self.datasets.bulk_insert(dataset_id, c))
            else:
                self.datasets.bulk_insert(dataset_id, c)

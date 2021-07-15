"""Batch operations
"""
from typing import Callable
from ..api.client import APIClient
from .chunk import Chunker
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from ..concurrency import multithread, multiprocess

class BatchInsert(APIClient, Chunker):
    def insert_documents(self, dataset_id: str, docs: list, 
        bulk_fn: Callable=None, verbose: bool=True,
        chunksize: int=20, max_workers:int =8, *args, **kwargs):
        """
        Insert a list of documents with multi-threading automatically
        enabled.
        """
        def bulk_insert_func(docs):
            return self.datasets.bulk_insert(
                dataset_id,
                docs, *args, **kwargs)
        
        if bulk_fn is not None:
            return multiprocess(
                func=bulk_fn,
                iterables=docs,
                post_func_hook=bulk_insert_func,
                max_workers=max_workers,
                chunksize=chunksize)

        return multithread(bulk_insert_func, docs, 
            max_workers=max_workers, chunksize=chunksize)

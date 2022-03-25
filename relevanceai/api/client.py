# -*- coding: utf-8 -*-
"""Batch client to allow for batch insertions/retrieval and encoding
"""
from typing import Callable

from relevanceai.api.batch.batch_insert import BatchInsertClient
from relevanceai.api.batch.batch_insert_async import BatchInsertAsync
from relevanceai.api.batch.batch_retrieve import BatchRetrieveClient
from relevanceai.api.endpoints.client import APIClient
from relevanceai.package_utils.config import CONFIG
from relevanceai.package_utils.utils import Utils


class BatchAPIClient(
    BatchInsertClient, BatchInsertAsync, Utils, BatchRetrieveClient, APIClient
):
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

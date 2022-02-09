"""Testing code for batch inserting
"""
import pytest

import numpy as np

from typing import Dict, List

from relevanceai import Client

from tests.globals.constants import generate_dataset_id


def do_nothing(documents):
    return documents


def cause_error(documents):
    for d in documents:
        d["value"] = np.nan
    return documents


def cause_some_error(documents):
    MAX_ERRORS = 5
    ERROR_COUNT = 0
    for d in documents:
        if ERROR_COUNT < MAX_ERRORS:
            d["value"] = np.nan
            ERROR_COUNT += 1
    return documents


class TestInsert:
    test_dataset_id = generate_dataset_id()

    def test_batch_insert(self, vector_documents: List[Dict], test_client: Client):
        test_client.config.reset()
        results = test_client._insert_documents(self.test_dataset_id, vector_documents)
        test_client.datasets.monitor.health(self.test_dataset_id)
        assert len(results["failed_documents"]) == 0

    def test_datetime_upload(self, datetime_documents: List[Dict], test_client: Client):
        test_client.config.reset()
        results = test_client._insert_documents(
            self.test_dataset_id, datetime_documents
        )
        assert results["inserted"] == len(datetime_documents)

    def test_numpy_upload(
        self,
        test_client: Client,
        numpy_documents: List[Dict],
    ):
        test_client.config.reset()
        results = test_client._insert_documents(self.test_dataset_id, numpy_documents)
        test_client.datasets.monitor.health(self.test_dataset_id)
        assert len(results["failed_documents"]) == 0

    def test_pandas_upload(self, test_client: Client, pandas_documents: List[Dict]):
        test_client.config.reset()
        results = test_client._insert_documents(self.test_dataset_id, pandas_documents)
        test_client.datasets.monitor.health(self.test_dataset_id)
        assert len(results["failed_documents"]) == 0

    def test_assorted_nested_upload(
        self,
        test_client: Client,
        assorted_nested_documents: List[Dict],
    ):
        test_client.config.reset()
        results = test_client._insert_documents(
            self.test_dataset_id, assorted_nested_documents
        )
        assert len(results["failed_documents"]) == 0


class TestPullUpdatePush:
    def test_pull_update_push_simple(self, test_client: Client, sample_dataset_id: str):
        results = test_client.pull_update_push(sample_dataset_id, do_nothing)
        assert len(results["failed_documents"]) == 0

    @pytest.mark.xfail
    def test_pull_update_push_with_errors(
        self, test_client: Client, sample_dataset_id: str
    ):
        with pytest.raises(Exception) as execinfo:
            test_client.pull_update_push(sample_dataset_id, cause_error)

    @pytest.mark.xfail
    def test_with_some_errors(self, test_client: Client, sample_dataset_id: str):
        import requests

        with pytest.raises(requests.exceptions.InvalidJSONError) as execinfo:
            test_client.pull_update_push(sample_dataset_id, cause_some_error)

    @pytest.mark.slow
    def test_pull_update_push_loaded(self, test_client: Client, sample_dataset_id: str):

        response = test_client.pull_update_push(sample_dataset_id, do_nothing)
        assert len(response["failed_documents"]) == 0, "Failed to insert documents"

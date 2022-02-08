"""Testing code for batch inserting
"""
import pytest

import numpy as np

from typing import Dict, List

from relevanceai import Client


class TestInsert:
    def test_batch_insert(
        self, vector_documents: List[Dict], test_dataset_id: str, test_client: Client
    ):
        results = test_client._insert_documents(test_dataset_id, vector_documents)
        assert len(results["failed_documents"]) == 0

    def test_health(self, test_dataset_id: str, test_client: Client):
        health = test_client.datasets.monitor.health(test_dataset_id)
        assert health["_chunk_.label"]["exists"] == 100

    def test_csv_upload(self, test_csv_dataset):
        response, original_length = test_csv_dataset
        assert response["inserted"] == original_length

    def test_datetime_upload(
        self, datetime_documents: List[Dict], test_dataset_id: str, test_client: Client
    ):
        results = test_client._insert_documents(test_dataset_id, datetime_documents)
        assert len(results["failed_documents"]) == 0

    def test_numpy_upload(
        self, numpy_documents: List[Dict], test_dataset_id: str, test_client: Client
    ):
        results = test_client._insert_documents(test_dataset_id, numpy_documents)
        assert len(results["failed_documents"]) == 0

    def test_pandas_upload(
        self, pandas_documents: List[Dict], test_dataset_id: str, test_client: Client
    ):
        results = test_client._insert_documents(test_dataset_id, pandas_documents)
        assert len(results["failed_documents"]) == 0

    def test_assorted_nested_upload(
        self,
        assorted_nested_documents: List[Dict],
        test_dataset_id: str,
        test_client: Client,
    ):
        results = test_client._insert_documents(
            test_dataset_id, assorted_nested_documents
        )
        assert len(results["failed_documents"]) == 0


# Mock a callable For pull update push
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


class TestPullUpdatePush:
    """Testing Pull Update Push"""

    def test_pull_update_push_simple(
        self, test_client: Client, sample_dataset: List[Dict]
    ):
        """Simple test for pull update push"""
        results = test_client.pull_update_push(sample_dataset, do_nothing)
        assert len(results["failed_documents"]) == 0

    @pytest.mark.xfail
    def test_pull_update_push_with_errors(
        self, test_client: Client, sample_dataset: List[Dict]
    ):
        """Simple test for pull update push with an errored update function"""
        with pytest.raises(Exception) as execinfo:
            test_client.pull_update_push(sample_dataset, cause_error)

    @pytest.mark.xfail
    def test_with_some_errors(self, test_client: Client, sample_dataset: List[Dict]):
        """Test with some errors"""
        import requests

        with pytest.raises(requests.exceptions.InvalidJSONError) as execinfo:
            test_client.pull_update_push(sample_dataset, cause_some_error)

    @pytest.mark.slow
    def test_pull_update_push_loaded(
        self, test_client: Client, sample_dataset: List[Dict]
    ):
        """Stress testing pull update push."""

        def do_nothing(documents):
            return documents

        response = test_client.pull_update_push(sample_dataset, do_nothing)
        assert len(response["failed_documents"]) == 0, "Failed to insert documents"


# class TestCleanUp:
#     def test_clean_up(self, test_client, test_dataset_id):
#         assert test_client.datasets.delete(test_dataset_id)

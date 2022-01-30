"""Testing code for batch inserting
"""
import numpy as np
import pytest
from relevanceai import Client


class TestInsert:
    """Testing the insert functionalities"""

    def test_batch_insert(self, sample_vector_documents, test_dataset_id, test_client):
        """Batch insert"""
        results = test_client._insert_documents(
            test_dataset_id, sample_vector_documents
        )
        assert len(results["failed_documents"]) == 0

    def test_health(self, test_dataset_id, test_client: Client):
        """Batch insert"""
        health = test_client.datasets.monitor.health(test_dataset_id)
        assert health["_chunk_.label"]["exists"] == 100

    def test_csv_upload(self, test_csv_dataset):
        response, original_length = test_csv_dataset
        assert response["inserted"] == original_length

    def test_datetime_upload(self, test_datetime_dataset):
        response, original_length = test_datetime_dataset
        assert response["inserted"] == original_length

    def test_numpy_upload(self, test_numpy_dataset):
        response, original_length = test_numpy_dataset
        assert response["inserted"] == original_length

    def test_pandas_upload(self, test_pandas_dataset):
        response, original_length = test_pandas_dataset
        assert response["inserted"] == original_length

    def test_nested_assorted_upload(self, test_nested_assorted_dataset):
        response, original_length = test_nested_assorted_dataset
        assert response["inserted"] == original_length


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

    def test_pull_update_push_simple(self, test_client, test_sample_dataset):
        """Simple test for pull update push"""
        results = test_client.pull_update_push(test_sample_dataset, do_nothing)
        assert len(results["failed_documents"]) == 0

    @pytest.mark.xfail
    def test_pull_update_push_with_errors(self, test_client, test_sample_dataset):
        """Simple test for pull update push with an errored update function"""
        with pytest.raises(Exception) as execinfo:
            results = test_client.pull_update_push(test_sample_dataset, cause_error)

    @pytest.mark.xfail
    def test_with_some_errors(self, test_client, test_sample_dataset):
        """Test with some errors"""
        import requests

        with pytest.raises(requests.exceptions.InvalidJSONError) as execinfo:
            results = test_client.pull_update_push(
                test_sample_dataset, cause_some_error
            )

    @pytest.mark.slow
    def test_pull_update_push_loaded(self, test_sample_dataset, test_client):
        """Stress testing pull update push."""

        def do_nothing(documents):
            return documents

        response = test_client.pull_update_push(test_sample_dataset, do_nothing)
        assert len(response["failed_documents"]) == 0, "Failed to insert documents"


# class TestCleanUp:
#     def test_clean_up(self, test_client, test_dataset_id):
#         assert test_client.datasets.delete(test_dataset_id)

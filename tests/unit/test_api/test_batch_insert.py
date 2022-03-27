"""Testing code for batch inserting
"""
import pytest

import numpy as np

from functools import partial
from typing import Dict, List

from relevanceai import Client

from tests.globals.constants import generate_dataset_id, generate_random_integer
from tests.conftest import correct_client_config, test_client


def do_nothing(documents):
    return documents


def add_new_column(documents, column, value):
    for document in documents:
        document[column] = value
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
        results = test_client._insert_documents(self.test_dataset_id, vector_documents)
        test_client.datasets.monitor.health(self.test_dataset_id)
        assert len(results["failed_documents"]) == 0

    def test_datetime_upload(self, datetime_documents: List[Dict], test_client: Client):
        results = test_client._insert_documents(
            self.test_dataset_id, datetime_documents
        )
        assert results["inserted"] == len(datetime_documents)

    def test_numpy_upload(
        self,
        test_client: Client,
        numpy_documents: List[Dict],
    ):
        results = test_client._insert_documents(self.test_dataset_id, numpy_documents)
        test_client.datasets.monitor.health(self.test_dataset_id)
        assert len(results["failed_documents"]) == 0

    def test_pandas_upload(self, test_client: Client, pandas_documents: List[Dict]):
        results = test_client._insert_documents(self.test_dataset_id, pandas_documents)
        test_client.datasets.monitor.health(self.test_dataset_id)
        assert len(results["failed_documents"]) == 0

    def test_assorted_nested_upload(
        self,
        test_client: Client,
        assorted_nested_documents: List[Dict],
    ):
        results = test_client._insert_documents(
            self.test_dataset_id, assorted_nested_documents
        )
        assert len(results["failed_documents"]) == 0


class TestInsertImages:
    def setup(self):
        from pathlib import Path
        from uuid import uuid4

        self.filename = "lovelace.jpg"

        self.directory = Path(str(uuid4()))
        self.directory.mkdir()
        with open(self.filename, "wb") as f:
            f.write(b"ghuewiogahweuaioghweqrofleuwaiolfheaswufg9oeawhfgaeuw")

    def test_insert_media_folder(self, test_client: Client):
        self.ds = test_client.Dataset(generate_dataset_id())
        results = self.ds.insert_media_folder(
            field="images",
            path=self.directory,
            recurse=False,  # No subdirectories exist anyway
        )
        assert results is None

    def teardown(self):
        self.ds.delete()

        try:
            import os

            os.remove(self.directory / self.filename)
        except FileNotFoundError:
            pass

        self.directory.rmdir()


class TestPullUpdatePush:
    def test_pull_update_push_simple(
        self, test_client: Client, simple_documents: List[Dict], sample_dataset_id: str
    ):
        test_client._insert_documents(sample_dataset_id, simple_documents)
        results = test_client.pull_update_push(sample_dataset_id, do_nothing)
        assert len(results["failed_documents"]) == 0

    @pytest.mark.xfail
    def test_pull_update_push_with_errors(
        self, test_client: Client, simple_documents: List[Dict], sample_dataset_id: str
    ):
        test_client._insert_documents(sample_dataset_id, simple_documents)
        with pytest.raises(Exception) as execinfo:
            test_client.pull_update_push(sample_dataset_id, cause_error)

    @pytest.mark.xfail
    def test_with_some_errors(
        self, test_client: Client, simple_documents: List[Dict], sample_dataset_id: str
    ):
        import requests

        test_client._insert_documents(sample_dataset_id, simple_documents)
        with pytest.raises(requests.exceptions.InvalidJSONError) as execinfo:
            test_client.pull_update_push(sample_dataset_id, cause_some_error)

    @pytest.mark.slow
    def test_pull_update_push_loaded(
        self, test_client: Client, simple_documents: List[Dict], sample_dataset_id: str
    ):
        test_client._insert_documents(sample_dataset_id, simple_documents)
        response = test_client.pull_update_push(sample_dataset_id, do_nothing)
        assert len(response["failed_documents"]) == 0, "Failed to insert documents"


class TestPullUpdatePushAsync:
    def test_pull_update_push_async(
        self, test_client: Client, vector_documents: List[Dict], sample_dataset_id: str
    ):
        column = "sample_7_value"
        value = -1
        fn = partial(add_new_column, column=column, value=value)

        test_client._insert_documents(sample_dataset_id, vector_documents)

        ds = test_client.Dataset(sample_dataset_id)
        ds.pull_update_push_async(sample_dataset_id, fn)

        documents = ds.get_all_documents()
        for document in documents:
            assert document[column] == value

"""Test labels
"""
import pytest
from relevanceai import Client
from relevanceai import mock_documents
from tests.globals.constants import (
    SAMPLE_DATASET_DATASET_PREFIX,
    generate_random_label,
    generate_random_string,
    generate_random_vector,
    generate_random_integer,
)
import random


def random_vector(vector_length: int = 5):
    return [random.uniform(0, 1) for _ in range(5)]


@pytest.fixture
def test_label_ds(test_client: Client):

    dataset = test_client.Dataset(SAMPLE_DATASET_DATASET_PREFIX + "_label")
    dataset.insert_documents(
        documents=mock_documents(100),
        create_id=True,
    )
    yield dataset
    dataset.delete()


@pytest.fixture
def label_documents():
    return [
        {"label": "value", "label_vector_": random_vector()},
        {"label": "value-2", "label_vector_": random_vector()},
    ]


class TestLabelOps:
    def test_label_ops(self, test_client: Client, label_documents):
        # Add testing for label ops
        from relevanceai.operations_new.label import LabelOps

        ops = LabelOps(
            label_documents=label_documents,
            credentials=test_client.credentials,
            vector_field="sample_1_vector_",
            expanded=True,
        )
        # TODO: Test for expanded = True and False
        # Test for differnet label documents (is an intuitive error returned)
        documents = mock_documents(100)
        docs = ops.transform(
            documents=documents,
        )
        assert True

"""Testing for dataset labels.
"""
import pytest
import random
from relevanceai.http_client import Dataset, Client

from tests.globals.constants import (
    generate_random_vector,
    generate_random_string,
    SAMPLE_DATASET_DATASET_PREFIX,
)

LABEL_DATSET_ID = SAMPLE_DATASET_DATASET_PREFIX + generate_random_string().lower()


@pytest.fixture(scope="function")
def test_label_df(test_client: Client, vector_documents):
    df = test_client.Dataset(LABEL_DATSET_ID)
    df.upsert_documents(vector_documents)
    yield df
    df.delete()


def test_dataset_labelling(test_label_df: Dataset):
    results = test_label_df.label_from_dataset(
        vector_field="sample_1_vector_",
        alias="example",
        label_dataset_id=test_label_df.dataset_id,
        label_fields=["sample_1_label"],
        label_vector_field="sample_1_vector_",
        filters=[
            # {
            #     "field": "sample_1_label",
            #     "filter_type": "exists",
            #     "condition": ">=",
            #     "condition_value": " ",
            # },
            # {
            #     "field": "sample_1_vector_",
            #     "filter_type": "exists",
            #     "condition": ">=",
            #     "condition_value": " ",
            # },
        ],
    )
    assert "_label_.example" in test_label_df.schema, "schema is incorrect"
    assert len(results["failed_documents"]) == 0, "failed to label documents :("


def test_labelling_vector(test_label_df: Dataset):
    ALIAS = "sample"
    result = test_label_df.label_vector(
        generate_random_vector(),
        label_vector_field="sample_1_vector_",
        alias=ALIAS,
        label_dataset_id=test_label_df.dataset_id,
        label_fields=["sample_1_label"],
        number_of_labels=1,
    )
    assert len(result["_label_"]["sample"]) > 0, "missing_labels"


@pytest.mark.skip(reason="we dont really use this.")
def test_labelling_document(test_df: Dataset):
    result = test_df.label_document(
        {},
        label_vector_field="sample_1_vector_",
        alias="sample",
        label_dataset_id=test_df.dataset_id,
        label_fields=["path"],
        number_of_labels=1,
    )
    assert len(result["_label_"]["sample"]) > 0, "missing_labels"


def test_labelling_by_model(test_df: Dataset):

    LABEL_LIST = ["cat", "dog"]

    def random_vector(labels):
        return [generate_random_vector() for _ in labels]

    test_df.label_from_list(
        vector_field="sample_1_vector_",
        model=random_vector,
        label_list=LABEL_LIST,
        alias="pets",
    )
    assert "_label_.pets" in test_df.schema

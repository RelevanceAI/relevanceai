"""Testing for dataset labels.
"""
import pytest
from relevanceai.http_client import Dataset, Client
from ...utils import generate_random_vector, generate_random_string
from ...conftest import SAMPLE_DATASET_DATASET_PREFIX

LABEL_DATSET_ID = SAMPLE_DATASET_DATASET_PREFIX + generate_random_string().lower()


@pytest.fixture(scope="session")
def test_label_df(test_client: Client, sample_vector_documents):
    df = test_client.Dataset(LABEL_DATSET_ID)
    df.upsert_documents(sample_vector_documents)
    yield df
    df.delete()


def test_dataset_labelling(test_label_df: Dataset):
    results = test_label_df.label(
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
        generate_random_vector(100),
        label_vector_field="sample_1_vector_",
        alias=ALIAS,
        label_dataset_id=test_label_df.dataset_id,
        label_fields=["sample_1_label"],
        number_of_labels=1,
    )
    assert len(result["_label_"]["sample"]) > 0, "missing_labels"


@pytest.mark.skip(reason="we dont really use this.")
def test_labelling_document(test_dataset_df: Dataset):
    result = test_dataset_df.label_document(
        {},
        label_vector_field="sample_1_vector_",
        alias="sample",
        label_dataset_id=test_dataset_df.dataset_id,
        label_fields=["path"],
        number_of_labels=1,
    )
    assert len(result["_label_"]["sample"]) > 0, "missing_labels"

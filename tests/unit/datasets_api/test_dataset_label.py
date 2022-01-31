"""Testing for dataset labels 
"""
import pytest
from relevanceai.http_client import Dataset
from ...utils import generate_random_vector


def test_dataset_labelling(test_dataset_df: Dataset):
    results = test_dataset_df.label(
        vector_field="sample_1_vector_",
        alias="example",
        label_dataset_id=test_dataset_df.dataset_id,
        label_fields=["sample_1_label"],
        label_vector_field="sample_1_vector_",
        filters=[
            {
                "field": "sample_1_label",
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            },
            {
                "field": "sample_1_vector_",
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            },
        ],
    )
    assert "_label_.example" in test_dataset_df.schema, "schema is incorrect"
    assert len(results["failed_documents"]) == 0, "failed to label documents :("


def test_labelling_vector(test_dataset_df: Dataset):
    ALIAS = "sample"
    result = test_dataset_df.label_vector(
        generate_random_vector(100),
        label_vector_field="sample_1_vector_",
        alias=ALIAS,
        label_dataset_id=test_dataset_df.dataset_id,
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

"""Testing for dataset labels 
"""
from ...utils import generate_random_vector
from relevanceai.http_client import Dataset


def test_dataset_labelling(test_dataset_df: Dataset):
    results = test_dataset_df.label(
        vector_field="documentation_vector_",
        alias="example",
        label_dataset="_github_repo_clip",
        label_fields=["path"],
        label_vector_field="documentation_vector_",
    )
    assert "_label_.example" in test_dataset_df.schema, "schema"


def test_labelling_vector(test_dataset_df: Dataset):
    result = test_dataset_df.label_vector(
        generate_random_vector(),
        label_vector_field="sample_1_vector_",
        alias="sample",
        label_dataset=test_dataset_df.dataset_id,
        label_fields=["path"],
        number_of_labels=1,
    )
    assert len(result["_label_"]["sample"]) > 0, "missing_labels"


def test_labelling_document(test_dataset_df: Dataset):
    result = test_dataset_df.label_vector(
        generate_random_vector(),
        label_vector_field="sample_1_vector_",
        alias="sample",
        label_dataset=test_dataset_df.dataset_id,
        label_fields=["path"],
        number_of_labels=1,
    )
    assert len(result["_label_"]["sample"]) > 0, "missing_labels"

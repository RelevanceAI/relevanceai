"""
    Testing dataset read operations
"""

import pandas as pd
from relevanceai.http_client import Dataset, Client


def test_apply(test_dataset_df: Dataset):
    RANDOM_STRING = "you are the kingj"
    test_dataset_df["sample_1_label"].apply(
        lambda x: x + RANDOM_STRING, output_field="sample_1_label_2"
    )
    assert test_dataset_df["sample_1_label_2"][0].endswith(RANDOM_STRING)


def test_apply(test_dataset_df: Dataset):
    RANDOM_STRING = "you are the queen"
    LABEL = "sample_output"

    def bulk_fn(docs):
        for d in docs:
            d[LABEL] = d["sample_1_label"] + RANDOM_STRING
        return docs

    test_dataset_df["sample_1_label"].bulk_apply(bulk_fn, output_field=LABEL)
    assert test_dataset_df[LABEL][0].endswith(RANDOM_STRING)


def test_df_insert_csv_successful(test_csv_df: Dataset):
    """Test Insert CSv successful"""
    response, original_length = test_csv_df
    assert response["inserted"] == original_length, "incorrect insertion"

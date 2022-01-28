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


def test_bulk_apply(test_dataset_df: Dataset):
    RANDOM_STRING = "you are the queen"
    LABEL = "sample_output"

    def bulk_fn(docs):
        for d in docs:
            d[LABEL] = d.get("sample_1_label", "") + RANDOM_STRING
        return docs

    test_dataset_df.bulk_apply(bulk_fn)
    assert test_dataset_df[LABEL][0].endswith(RANDOM_STRING)


def test_df_insert_csv_successful(test_csv_df: Dataset):
    """Test Insert CSv successful"""
    response, original_length = test_csv_df
    assert response["inserted"] == original_length, "incorrect insertion"


def test_insert_df(test_dataset_df: Dataset):
    pandas_df = pd.DataFrame({"pandas_value": [3, 2, 1], "_id": ["10", "11", "12"]})
    test_dataset_df.insert_pandas_dataframe(pandas_df)
    assert "pandas_value" in pandas_df.columns

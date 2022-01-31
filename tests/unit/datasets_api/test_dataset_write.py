"""
    Testing dataset read operations
"""

import pandas as pd
from relevanceai.http_client import Dataset, Client


def test_apply(test_client: Client, test_sample_vector_dataset):
    RANDOM_STRING = "you are the kingj"
    df = test_client.Dataset(test_sample_vector_dataset)
    df["sample_1_label"].apply(
        lambda x: x + RANDOM_STRING, output_field="sample_1_label_2"
    )
    filtered_documents = df.datasets.documents.get_where(
        df.dataset_id,
        filters=[
            {
                "field": "sample_1_label_2",
                "filter_type": "contains",
                "condition": "==",
                "condition_value": RANDOM_STRING,
            }
        ],
    )
    assert len(filtered_documents["documents"]) > 0


def test_bulk_apply(test_client: Client, test_sample_vector_dataset):
    RANDOM_STRING = "you are the queen"
    LABEL = "sample_output"
    df = test_client.Dataset(test_sample_vector_dataset)

    def bulk_fn(docs):
        for d in docs:
            d[LABEL] = d.get("sample_1_label", "") + RANDOM_STRING
        return docs

    df.bulk_apply(bulk_fn)
    filtered_documents = df.datasets.documents.get_where(
        df.dataset_id,
        filters=[
            {
                "field": "sample_output",
                "filter_type": "contains",
                "condition": "==",
                "condition_value": RANDOM_STRING,
            }
        ],
    )
    assert len(filtered_documents["documents"]) > 0


def test_df_insert_csv_successful(test_csv_df: Dataset):
    """Test Insert CSv successful"""
    response, original_length = test_csv_df
    assert response["inserted"] == original_length, "incorrect insertion"


def test_insert_df(test_dataset_df: Dataset):
    pandas_df = pd.DataFrame({"pandas_value": [3, 2, 1], "_id": ["10", "11", "12"]})
    test_dataset_df.insert_pandas_dataframe(pandas_df)
    assert "pandas_value" in pandas_df.columns

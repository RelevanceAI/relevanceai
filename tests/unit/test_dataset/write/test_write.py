import pytest

import pandas as pd
from relevanceai.constants.errors import FieldNotFoundError

from relevanceai.dataset import Dataset


def test_apply(test_dataset: Dataset):
    random_string = "you are the kingj"
    test_dataset["sample_1_label"].apply(
        lambda x: x + random_string, output_field="sample_1_label_2"
    )
    filtered_documents = test_dataset.datasets.documents.get_where(
        test_dataset.dataset_id,
        filters=[
            {
                "field": "sample_1_label_2",
                "filter_type": "contains",
                "condition": "==",
                "condition_value": random_string,
            }
        ],
    )
    assert len(filtered_documents["documents"]) > 0


def test_create_id_in_documents(test_dataset: Dataset):
    docs = [{"value": 2}]
    with pytest.raises(FieldNotFoundError):
        test_dataset.insert_documents(docs)

    results = test_dataset.insert_documents(docs, create_id=True)
    assert results is None, "Documents are inserted."


def test_bulk_apply(test_dataset: Dataset):
    random_string = "you are the queen"
    label = "sample_output"

    def bulk_fn(docs):
        for d in docs:
            d[label] = d.get("sample_1_label", "") + random_string
        return docs

    test_dataset.bulk_apply(bulk_fn)
    filtered_documents = test_dataset.datasets.documents.get_where(
        test_dataset.dataset_id,
        filters=[
            {
                "field": "sample_output",
                "filter_type": "contains",
                "condition": "==",
                "condition_value": random_string,
            }
        ],
    )
    assert len(filtered_documents["documents"]) > 0


def test_insert_df(test_dataset: Dataset):
    pandas_df = pd.DataFrame({"pandas_value": [3, 2, 1], "_id": ["10", "11", "12"]})
    test_dataset.insert_pandas_dataframe(pandas_df)
    assert "pandas_value" in pandas_df.columns

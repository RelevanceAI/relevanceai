"""
    Testing dataset read operations
"""

import os
import uuid
import pandas as pd

import pytest

from relevanceai import Client

from relevanceai.dataset import Dataset

from tests.globals.constants import NOT_IMPLEMENTED, generate_dataset_id


class TestDatasetExport:
    def test_to_csv(self, test_dataset: Dataset):
        fname = f"{uuid.uuid4().__str__()}.csv"
        test_dataset.to_csv(fname)
        df = pd.read_csv(fname)

        assert df.values.size > 0

        os.remove(fname)

    def test_to_dict(self, test_dataset: Dataset):
        data = test_dataset.to_dict("records")

        assert data[0]

    def test_to_dataset(self, test_client: Client, test_dataset: Dataset):
        child_dataset_id = generate_dataset_id()

        test_dataset.to_dataset(child_dataset_id=child_dataset_id)
        child_dataset = test_client.Dataset(child_dataset_id)

        test_ids = [document["_id"] for document in test_dataset.get_all_documents()]
        child_ids = [document["_id"] for document in child_dataset.get_all_documents()]

        assert all(_id in test_ids for _id in child_ids)

    def test_to_pandas_dataframe(self, test_dataset: Dataset):
        df = test_dataset.to_pandas_dataframe()

        dataframe_ids = df.index.values.tolist()
        test_ids = [document["_id"] for document in test_dataset.get_all_documents()]

        assert all(_id in test_ids for _id in dataframe_ids)


class TestDatasetImport:
    pass


class TestDatasetSeries:
    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_list_aliases(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_head(self):
        assert False

    def test_sample(self, test_dataset: Dataset):
        sample_n = test_dataset[["sample_1_label", "sample_2_label"]].sample(n=10)
        assert len(sample_n) == 10
        assert len(sample_n[0].keys()) == 3

        sample_n = test_dataset["sample_1_label"].sample(n=10)
        assert len(sample_n) == 10

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_all(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_apply(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_bulk_apply(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_numpy(self):
        assert False

    def test_add(self, test_dataset: Dataset):
        series_1 = test_dataset["sample_1_value"]
        series_2 = test_dataset["sample_2_value"]
        series_3 = test_dataset["sample_1_label"]

        series_1 + series_2

        try:
            series_1 + series_3
        except ValueError:
            assert True

        try:
            series_3 + series_2
        except ValueError:
            assert True

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def value_counts(self):
        assert False


class TestDatasetStats:
    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_value_counts(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_describe(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_corr(self):
        assert False

    def test_health(self, test_dataset: Dataset):
        dataframe_output = test_dataset.health(output_format="dataframe")
        assert type(dataframe_output) == pd.DataFrame

        json_output = test_dataset.health(output_format="json")
        assert type(json_output) == dict

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_aggregate(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_facets(self):
        assert False


class TestDatasetMetadata:
    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_insert_metadata(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_upsert_metadata(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_to_dict(self):
        assert False


class TestDatasetRead:
    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_shape(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_info(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_head(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_get(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_schema(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_columns(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_filter(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_list_vector_fields(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_list_cluster_aliases(self):
        assert False


class TestDatasetWrite:
    def test_apply(self, test_dataset: Dataset):
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

    def test_bulk_apply(self, test_dataset: Dataset):
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

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_concat(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_create(self):
        assert False

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_delete(self):
        assert False

    def test_insert_dataframe(self, test_dataset: Dataset):
        pandas_df = pd.DataFrame({"pandas_value": [3, 2, 1], "_id": ["10", "11", "12"]})
        test_dataset.insert_pandas_dataframe(pandas_df)
        assert "pandas_value" in pandas_df.columns

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_insert_csv(self):
        assert False

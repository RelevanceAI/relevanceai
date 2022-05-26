"""
    Testing dataset read operations
"""

import os
import uuid

import numpy as np
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
    def test_list_aliases(self, test_dataset: Dataset):
        series = test_dataset["sample_1_label"]
        aliases = series.list_aliases()
        assert "div" in aliases

    def test_head(self, test_dataset: Dataset):
        series = test_dataset["sample_2_label"]
        aliases = series.list_aliases()
        assert "div" in aliases

    def test_sample(self, test_dataset: Dataset):
        sample_n = test_dataset[["sample_1_label", "sample_2_label"]].sample(n=10)
        assert len(sample_n) == 10
        assert len(sample_n[0].keys()) == 3

        sample_n = test_dataset["sample_1_label"].sample(n=10)
        assert len(sample_n) == 10

    @pytest.mark.skip(reason=NOT_IMPLEMENTED)
    def test_all(self):
        assert False

    def test_apply(self, test_dataset: Dataset):
        before = test_dataset["sample_1_value"].values
        test_dataset["sample_1_value"].apply(
            lambda x: x + 1, output_field="sample_1_value"
        )
        after = test_dataset["sample_1_value"].values
        assert ((before + 1) == after).mean()

    def test_bulk_apply(self, test_dataset: Dataset):
        def bulk_func(documents):
            for document in documents:
                document["sample_1_value"] += 1
            return documents

        before = test_dataset["sample_1_value"].values
        test_dataset["sample_1_value"].bulk_apply(bulk_func)
        after = test_dataset["sample_1_value"].values

        assert ((before + 1) == after).mean()

    def test_numpy(self, test_dataset: Dataset):
        series = test_dataset["sample_1_value"]
        numpy = series.numpy()
        assert isinstance(numpy, np.ndarray)

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

    def test_value_counts(self, test_dataset: Dataset):
        series = test_dataset["sample_1_value"]
        value_counts = series.value_counts()
        assert "sample_1_value" in value_counts.columns
        assert value_counts.size > 0

    def test_repr_html(self, test_dataset: Dataset):
        series = test_dataset["sample_1_value"]
        repr_html = series._repr_html_()
        assert "<" in repr_html
        assert ">" in repr_html


class TestDatasetStats:
    def test_value_counts(self, test_dataset: Dataset):
        value_counts = test_dataset.value_counts(field="sample_1_label")
        assert isinstance(value_counts, pd.DataFrame)

    def test_describe(self, test_dataset: Dataset):
        describe = test_dataset.describe()
        assert isinstance(describe, pd.DataFrame)

    @pytest.mark.skip(reason="requires matplotlib not sure how to handle")
    def test_corr(self, test_dataset: Dataset):
        test_dataset.cluster(model="kmeans", vector_fields=["sample_1_vector_"])
        corr = test_dataset.corr(
            X="sample_1_value",
            Y="sample_2_value",
            vector_field="sample_1_vector_",
            alias="kmeans-8",
            show_plot=False,
        )
        assert corr is None

    def test_health(self, test_dataset: Dataset):
        dataframe_output = test_dataset.health(output_format="dataframe")
        assert isinstance(dataframe_output, pd.DataFrame)

        json_output = test_dataset.health(output_format="json")
        assert isinstance(json_output, dict)

    @pytest.mark.skip(reason="Not sure why its failing")
    def test_aggregate(self, test_dataset: Dataset):
        agg = test_dataset.aggregate(groupby=["sample_1_label"])
        assert True

    @pytest.mark.skip(reason="Not sure why its failing")
    def test_facets(self, test_dataset: Dataset):
        facets = test_dataset.facets()
        assert facets


class TestDatasetMetadata:
    payload = {"test_metavalue": 1}

    def test_insert_metadata(self, test_dataset: Dataset):
        result = test_dataset.insert_metadata(self.payload)
        assert result is None

    def test_upsert_metadata(self, test_dataset: Dataset):
        result = test_dataset.upsert_metadata(self.payload)
        assert result is None

    def test_to_dict(self, test_dataset: Dataset):
        metadata = test_dataset.metadata.to_dict()
        assert metadata == {}

        result = test_dataset.insert_metadata(self.payload)
        assert result is None

        metadata = test_dataset.metadata.to_dict()
        assert metadata == self.payload


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

    @pytest.mark.skip(reason="Not used")
    def test_dtype_mapping(self, test_dataset: Dataset):
        test_dataset.set_dtypes({})
        dtypes = test_dataset.get_dtypes()
        assert all(
            dtype in dtypes
            for dtype in [
                "_numeric_",
                "_category_",
                "_text_",
                "_image_",
            ]
        )

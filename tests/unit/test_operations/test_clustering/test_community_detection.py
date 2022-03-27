import pytest

from relevanceai import Client, mock_documents


@pytest.mark.skip(
    reason="community detection needs to be selected as a model under cluster"
)
def test_community_detection(test_client: Client, test_dataset_id: str):
    ds = test_client.Dataset(test_dataset_id + "_community-detection")
    ds.upsert_documents(mock_documents(100, 10))

    text_field = "sample_1_label"
    ds.community_detection(field=text_field)
    assert f"_cluster_.{text_field}.community-detection" in ds.schema

    vector_field = "sample_1_vector_"
    ds.community_detection(field=vector_field)
    assert f"_cluster_.{vector_field}.community-detection" in ds.schema

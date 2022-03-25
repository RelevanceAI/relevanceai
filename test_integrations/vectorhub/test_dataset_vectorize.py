from typing import Dict, List

from relevanceai import Client


def test_dataset_vectorize(
    test_client: Client, test_dataset_id: str, test_documents: List[Dict]
):
    ds = test_client.Dataset(test_dataset_id)
    ds.upsert_documents(test_documents)

    results = ds.vectorize(image_fields=["image_url"], text_fields=["data"])
    # This means the results are good yay!
    assert "image_url_clip_vector_" in ds.schema
    assert "data_use_vector_" in ds.schema

    # Make sure the metadata was updated as well
    assert "image_url" in ds.metadata["_vector_"]
    assert "data" in ds.metadata["_vector_"]

    results = ds.vectorize(image_fields=["image_url"])
    assert "image_url_clip_vector_" in results["skipped_vectors"]


def test_dataset_auto_text_cluster_dashboard(
    test_client: Client, test_dataset_id: str, test_documents: List[Dict]
):
    ds = test_client.Dataset(test_dataset_id)
    ds.upsert_documents(test_documents)

    alias = "kmeans-3"
    ds.auto_text_cluster_dashboard(text_fields=["data"], alias=alias)

    vector = "data_use_vector_"
    assert vector in ds.schema
    assert ".".join(["_cluster_", vector, alias]) in ds.schema

    # Make sure the metadata was updated as well
    assert "data" in ds.metadata["_vector_"]

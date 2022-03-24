import hdbscan

from relevanceai import Client, mock_documents


def test_hdbscan(test_client: Client, test_dataset_id: str):
    ds = test_client.Dataset(test_dataset_id + "_hdbscan")
    ds.upsert_documents(mock_documents(100))
    clusterer = test_client.ClusterOps(alias="hdbscan", model=hdbscan.HDBSCAN())
    clusterer.fit_predict_update(ds, vector_fields=["sample_1_vector_"])
    docs = clusterer.get_centroid_documents()
    all_ids = [d["_id"] for d in docs]
    assert "cluster--1" in all_ids

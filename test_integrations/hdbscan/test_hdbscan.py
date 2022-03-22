from relevanceai import Client


def test_hdbscan(test_client):
    import hdbscan
    from relevanceai import mock_documents

    DATASET_ID = "_test_sample_hdbscan"
    ALIAS = "hdbscan"
    ds = test_client.Dataset(DATASET_ID)
    ds.upsert_documents(mock_documents(100))
    model = hdbscan.HDBSCAN()
    clusterer = test_client.ClusterOps(alias="hdbscan", model=model)
    clusterer.fit(ds, vector_fields=["sample_1_vector_"])
    docs = clusterer.get_centroid_documents()
    all_ids = [d["_id"] for d in docs]
    assert "cluster--1" in all_ids

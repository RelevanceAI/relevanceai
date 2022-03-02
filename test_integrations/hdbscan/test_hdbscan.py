from relevanceai import Client


def test_hdbscan(test_df):
    import hdbscan

    model = hdbscan.HDBSCAN()
    clusterer = client.ClusterOps(alias="hdbscan", model=model)
    clusterer.fit_predict_update(test_df, vector_fields=["sample_1_vector_"])
    docs = clusterer.get_centroid_documents()
    all_ids = [d["_id"] for d in docs]
    assert "cluster--1" in all_ids

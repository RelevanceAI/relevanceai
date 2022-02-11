"""
Sklearn Integration Test
"""
from relevanceai import Client
from relevanceai.dataset_api import Dataset

from tests.globals.constants import generate_random_string


def test_cluster(test_df: Dataset):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = generate_random_string().lower()

    # check they're not in first
    assert f"_cluster_.{vector_field}.{alias}" not in test_df.schema

    model = KMeans()
    clusterer = test_df.cluster(
        model=model, alias=alias, vector_fields=[vector_field], overwrite=True
    )
    assert f"_cluster_.{vector_field}.{alias}" in test_df.schema
    assert len(clusterer.list_closest_to_center()) > 0


def test_dbscan(test_client: Client, test_df: Dataset):
    from sklearn.cluster import DBSCAN

    ALIAS = "dbscan"

    model = DBSCAN()
    clusterer = test_client.ClusterOps(alias=ALIAS, model=model)
    clusterer.fit_predict_update(test_df, ["sample_3_vector_"])
    assert any([x for x in test_df.schema if ALIAS in x])

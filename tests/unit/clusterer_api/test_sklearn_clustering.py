"""
Sklearn Integration Test
"""
from relevanceai import Client
from relevanceai.dataset_api import Dataset

from tests.globals.utils import generate_random_string


def test_cluster(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = generate_random_string().lower()

    # check they're not in first
    assert f"_cluster_.{vector_field}.{alias}" not in df.schema

    model = KMeans()
    clusterer = df.cluster(
        model=model, alias=alias, vector_fields=[vector_field], overwrite=True
    )
    assert f"_cluster_.{vector_field}.{alias}" in df.schema
    assert len(clusterer.centroids) > 0

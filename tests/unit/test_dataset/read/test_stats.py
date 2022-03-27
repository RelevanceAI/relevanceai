from relevanceai.dataset import Dataset


# Skip cluster tests
# def test_cluster(test_dataset: Dataset):
#     from relevanceai.operations.cluster.models.kmeans import KMeansModel

#     vector_field = "sample_1_vector_"
#     alias = "test_alias"

#     model = KMeansModel()

#     test_dataset.cluster(model=model, alias=alias, vector_fields=[vector_field])
#     assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema


# def test_centroids(test_clustered_df: Dataset):
#     closest = test_clustered_df.centroids(["sample_1_vector_"], "kmeans-10").closest()
#     assert "results" in closest
#     assert all("cluster" in cluster for cluster in list(closest["results"]))

#     furthest = test_clustered_df.centroids(["sample_1_vector_"], "kmeans-10").furthest()
#     assert "results" in furthest
#     assert all("cluster" in cluster for cluster in list(furthest["results"]))


def test_health(test_dataset: Dataset):
    import pandas as pd

    dataframe_output = test_dataset.health(output_format="dataframe")
    assert type(dataframe_output) == pd.DataFrame
    json_output = test_dataset.health(output_format="json")
    assert type(json_output) == dict

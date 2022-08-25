import pytest

from relevanceai.dataset import Dataset


@pytest.mark.xfail(reason="not sure")
def test_batch_cluster(test_dataset: Dataset):

    vector_field = "sample_1_vector_"
    n_clusters = 15
    model = "minibatchkmeans"

    test_dataset.batch_cluster(
        vector_fields=[vector_field],
        model=model,
        model_kwargs={"n_clusters": n_clusters},
    )
    assert f"_cluster_.{vector_field}.{model}-{n_clusters}" in test_dataset.schema


@pytest.mark.xfail(reason="not sure")
def test_batch_cluster_kmeans_integration(test_dataset: Dataset):
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=20)
    test_dataset.batch_cluster(
        vector_fields=["sample_1_vector_"],
        model=model,
    )
    assert f"_cluster_.sample_1_vector_.minibatchkmeans-20" in test_dataset.schema

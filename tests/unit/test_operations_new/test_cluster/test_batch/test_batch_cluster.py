from relevanceai import Client
from relevanceai.dataset import Dataset


def test_batch_cluster(test_dataset: Dataset):

    test_dataset.batch_cluster(
        vector_fields=["sample_1_vector_"],
        model="minibatchkmeans",
        model_kwargs={"n_clusters": 15},
    )
    assert f"_cluster_.sample_1_vector_.minibatchkmeans-15" in test_dataset.schema


def test_batch_cluster_kmeans_integration(test_dataset: Dataset):
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=20)
    test_dataset.batch_cluster(
        vector_fields=["sample_1_vector_"],
        model=model,
    )
    assert f"_cluster_.sample_1_vector_.minibatchkmeans-20" in test_dataset.schema

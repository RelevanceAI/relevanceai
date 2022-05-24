from relevanceai import Client
from relevanceai.dataset import Dataset


def test_batch_cluster(test_dataset: Dataset):

    test_dataset.batch_cluster(
        vector_fields=["sample_1_vector_"],
        model="minibatchkmeans",
        model_kwargs={"n_clusters": 15},
    )
    alias = "minibatchkmeans-15"
    assert alias in test_dataset.schema


def test_batch_cluster_kmeans_integration(test_dataset: Dataset):
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=20)
    test_dataset.batch_cluster(
        vector_fields=["sample_1_vector_"],
        model=model,
    )
    alias = "minibatchkmeans-20"
    assert alias in test_dataset.schema

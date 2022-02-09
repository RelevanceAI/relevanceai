from relevanceai.http_client import ClusterOps


def test_minibatch_clusterer(minibatch_clusterer: ClusterOps):
    assert len(minibatch_clusterer.list_closest_to_center()) > 0
    assert len(minibatch_clusterer.centroids) > 0


def test_kmeans_clusterer(kmeans_clusterer: ClusterOps):
    assert len(kmeans_clusterer.list_closest_to_center()) > 0
    assert len(kmeans_clusterer.centroids) > 0

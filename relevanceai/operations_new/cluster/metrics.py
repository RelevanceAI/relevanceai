from relevanceai.utils.integration_checks import (
    is_sklearn_available,
)

if is_sklearn_available():
    from sklearn.metrics import silhouette_samples
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.neighbors import NearestNeighbors

class ClusterMetrics:
    def silouhette_scoring(self, vectors, labels):
        silhouette_scores = silhouette_samples(vectors, labels)
        return silhouette_scores
    
    def compactness_scoring(self, centroids, labels, method="distance"):
        #WIP
        distances = []
        for v in labels:
            distances = pairwise_distances(centroids, labels)
        return distances
    
    def seperation_scoring(self, centroids, labels):
        #WIP
        distances = []
        for v in labels:
            distances = pairwise_distances(centroids, labels)
        return distances
from relevanceai import Client
from relevanceai.base import ClusterBase

from sklearn.cluster import KMeans

client = Client()

df = client.Dataset("iris")

n_clusters = 3
vector_field = "feature_vector_"


class Clusterer(ClusterBase):
    def fit_transform(self, vectors):
        return self.clusterer.fit_transform(vectors)

    def get_centroids(self):
        return self.clusterer.cluster_centers_


kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusterer = Clusterer(kmeans)

vectors = clusterer.fit_transform(df[vector_field].numpy())
centroids = clusterer.get_centroids()

print()

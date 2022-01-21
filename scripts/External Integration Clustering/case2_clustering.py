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

    def get_alias(self):
        return type(self.clusterer).__name__

    def get_labels(self):
        return self.clusterer.labels_


kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusterer = Clusterer(kmeans)

vectors = clusterer.fit_transform(df[vector_field].numpy())
df.set_cluster_labels(
    vector_field=vector_field,
    alias=clusterer.get_alias(),
    labels=clusterer.get_labels(),
)

from relevanceai import Client
from relevanceai.base import ClusterBase

from faiss import Kmeans

client = Client()

df = client.Dataset("set6_team_comps")
vector_field = "comp_vector_"
n_clusters = 10


class Clusterer(ClusterBase):
    def fit_transform(self, vectors):
        self.vectors = vectors
        return self.clusterer.train(vectors)

    def get_centroids(self):
        return self.clusterer.centroids

    def get_alias(self):
        return type(self.clusterer).__name__

    def get_labels(self):
        return self.clusterer.assign(self.vectors)[1]


vectors = df[vector_field].numpy().astype("float32")
vector_length = vectors.shape[1]

kmeans = Kmeans(d=vector_length, k=n_clusters, gpu=True, niter=1000, nredo=100)
clusterer = Clusterer(kmeans)

vectors = clusterer.fit_transform(vectors)

df.set_cluster_labels(
    vector_field=vector_field,
    alias=clusterer.get_alias(),
    labels=clusterer.get_labels(),
)

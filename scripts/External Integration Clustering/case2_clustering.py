"""
This script demonstrates a class based approach for clustering with KMeans Using the new Pandas-Like Dataset API for RelevanceAI Python Package
"""

import argparse

from relevanceai import Client
from relevanceai.base import ClusterBase

from sklearn.cluster import KMeans


def main():
    client = Client()

    df = client.Dataset(args.dataset_id)
    vector_field = args.vector_field
    n_clusters = args.n_clusters

    class Clusterer(ClusterBase):
        def fit_dataset(self, df, vector_field):

            vectors = df[vector_field].numpy()
            self.fit_transform(self, vectors)
            df.set_cluster_labels(
                vector_field=vector_field,
                alias=clusterer.get_alias(),
                labels=clusterer.get_labels(),
            )

        def fit_transform(self, vectors):
            return self.clusterer.fit_transform(vectors)

        def get_centroids(self):
            return self.clusterer.cluster_centers_

        def get_alias(self):
            return type(self.clusterer).__name__

        def get_labels(self):
            return self.clusterer.labels_

    kmeans = KMeans(n_clusters=n_clusters)
    clusterer = Clusterer(kmeans)

    clusterer.fit_dataset(df, vector_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faiss_clustering")

    parser.add_argument("dataset_id", help="The dataset_id of the dataset to cluster")
    parser.add_argument("vector_field", help="The vector field over which to cluster")
    parser.add_argument("n_clusters", help="The number of clusters to find")

    args = parser.parse_args()
    main(args)

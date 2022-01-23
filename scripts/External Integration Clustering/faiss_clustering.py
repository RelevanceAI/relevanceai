"""
This script demonstrates a class based approach for clustering with faiss Kmeans Using the new Pandas-Like Dataset API for RelevanceAI Python Package
"""

import argparse

from relevanceai import Client
from relevanceai.base import ClusterBase

from faiss import Kmeans


def main(args):

    client = Client()

    df = client.Dataset(args.dataset_id)
    vector_field = args.vector_field
    n_clusters = args.n_clusters

    class Clusterer(ClusterBase):
        def fit_dataset(self, df, vector_field):

            vectors = df[vector_field].numpy().astype("float32")
            self.fit_transform(self, vectors)
            df.set_cluster_labels(
                vector_field=vector_field,
                alias=clusterer.get_alias(),
                labels=clusterer.get_labels(),
            )

        def fit_transform(self, vectors):
            self.vectors = vectors
            return self.clusterer.train(vectors)

        def get_centroids(self):
            return self.clusterer.centroids

        def get_alias(self):
            return type(self.clusterer).__name__

        def get_labels(self):
            return self.clusterer.assign(self.vectors)[1]

    kmeans = Kmeans(d=args.dimensions, k=n_clusters, gpu=True, niter=1000, nredo=100)
    clusterer = Clusterer(kmeans)

    clusterer.fit_dataset(df, vector_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faiss_clustering")

    parser.add_argument("dataset_id", help="The dataset_id of the dataset to cluster")
    parser.add_argument("vector_field", help="The vector field over which to cluster")
    parser.add_argument("n_clusters", help="The number of clusters to find")
    parser.add_argument("dimensions", help="dimensions in vector_field")

    args = parser.parse_args()
    main(args)

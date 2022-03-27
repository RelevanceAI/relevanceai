# -*- coding: utf-8 -*-
"""
This script demonstrates a class based approach for clustering with faiss Kmeans Using the new Pandas-Like Dataset API for RelevanceAI Python Package
"""

import argparse

import numpy as np

from relevanceai import Client
from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.cluster.base import ClusterBase

from faiss import Kmeans


def main(args):

    client = Client()

    df = client.Dataset(args.dataset_id)
    vector_field = args.vector_field
    n_clusters = int(args.n_clusters)

    class FaissKMeans(ClusterBase):
        def __init__(self, model):
            self.model = model

        def fit_predict(self, vectors):
            vectors = np.array(vectors).astype("float32")

            self.model.train(vectors)
            cluster_labels = self.model.assign(vectors)[1]

            return cluster_labels

        def metadata(self):
            return self.model.__dict__

    model = FaissKMeans(model=Kmeans(d=4, k=n_clusters))

    clusterer = ClusterOps(model=model, alias=f"kmeans_{n_clusters}")

    clusterer.fit(dataset=df, vector_fields=[vector_field])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faiss_clustering")

    parser.add_argument("dataset_id", help="The dataset_id of the dataset to cluster")
    parser.add_argument("vector_field", help="The vector field over which to cluster")
    parser.add_argument("n_clusters", help="The number of clusters to find")

    args = parser.parse_args()
    main(args)

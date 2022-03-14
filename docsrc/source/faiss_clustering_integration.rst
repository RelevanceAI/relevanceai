.. _integration:


Faiss Kmeans (Facebook AI Similarity Search)
=================

Relevance AI comes with a number of integrations. Below are a few examples of such
integrations. It is also relatively easy to build your own!

Clustering
-----------------------------

Faiss Kmeans Example
#######################

.. code-block::

    import numpy as np
    from relevanceai import Client
    from relevanceai.ops.clusterops.cluster import ClusterOps
    from relevanceai.ops.clusterops.cluster import ClusterBase

    from faiss import Kmeans

    # instantiate the client
    client = Client()

    df = client.Dataset("dataset_id")
    vector_field = "vector_field"
    n_clusters = 3
    vector_dims = 4

    class FaissKMeans(ClusterBase):
        def __init__(self, model):
            self.model = model

        def fit_transform(self, vectors):
            vectors = np.array(vectors).astype("float32")

            self.model.train(vectors)
            cluster_labels = self.model.assign(vectors)[1]

            return cluster_labels

    model = FaissKMeans(model=Kmeans(d=vector_dims, k=n_clusters))

    clusterer = ClusterOps(model=model, alias=f"kmeans-{n_clusters}")

    clusterer.fit(dataset=df, vector_fields=[vector_field])

ClusterBase
=============================

.. code-block

    from relevanceai import ClusterBase

    from faiss import Kmeans

    class FaissKMeans(ClusterBase):
        def __init__(self, model):
            self.model = model

        def fit_transform(self, vectors):
            vectors = np.array(vectors).astype("float32")

            self.model.train(vectors)
            cluster_labels = self.model.assign(vectors)[1]

            return cluster_labels

        def metadata(self):
            return self.model.__dict__

    n_clusters = 10
    model = FaissKMeans(model=Kmeans(d=4, k=n_clusters))

    clusterer = Clusterer(model=model, alias=f"kmeans-{n_clusters}")

    clusterer.fit(dataset=df, vector_fields=[vector_field])

.. automodule:: relevanceai.clusterer.cluster_base
    :members:

.. automodule:: relevanceai.clusterer.kmeans_clusterer
    :members:
    :show-inheritance:
    :exclude-members: __init__


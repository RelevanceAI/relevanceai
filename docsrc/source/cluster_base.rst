Custom Cluster Models
-------------------------

The ClusterBase class is intended to be inherited so that users can add their own clustering algorithms 
and models. A cluster base has the following abstractmethods (methods to be overwritten):

- fit_transform
- metadata (optional if you want to store cluster metadata)
- get_cluster_documents(optional if you want to store cluster centroid documents)

.. code-block::

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

.. autoclass:: relevanceai.clusterer.cluster_base.ClusterBase
    :members:

.. autoclass:: relevanceai.clusterer.cluster_base.CentroidClusterBase
    :members:

.. autoclass:: relevanceai.clusterer.cluster_base.AdvancedCentroidClusterBase
    :members:

.. autoclass:: relevanceai.clusterer.kmeans_clusterer.KMeansModel
    :members:
    :show-inheritance:
    :exclude-members: __init__

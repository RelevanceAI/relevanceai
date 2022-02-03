Custom Cluster Models
-------------------------

The ClusterBase class is intended to be inherited so that users can add their own clustering algorithms 
and models. A cluster base has the following abstractmethods (methods to be overwritten):

- fit_transform
- metadata (optional if you want to store cluster metadata)
- get_cluster_documents(optional if you want to store cluster centroid documents)


`Centroidbase` is the most basic class to inherit. Use this class if you have an 
in-memory fitting algorithm.

If your clusters return centroids, you will want to inherit
`CentroidClusterBase`.

If your clusters can fit on batches, you will want to inherit 
`BatchClusterBase`.

If you have both Batches and Centroids, you will want to inherit both.

.. code-block::

    from relevanceai import CentroidClusterBase

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
        
        def get_centers(self):
            return 

    n_clusters = 10
    model = FaissKMeans(model=Kmeans(d=4, k=n_clusters))

    clusterer = Clusterer(model=model, alias=f"kmeans-{n_clusters}")

    clusterer.fit(dataset=df, vector_fields=[vector_field])

.. automodule:: relevanceai.clusterer.cluster_base

.. autoclass:: relevanceai.clusterer.cluster_base.ClusterBase
    :members:

.. autoclass:: relevanceai.clusterer.cluster_base.CentroidClusterBase
    :members:

.. autoclass:: relevanceai.clusterer.cluster_base.AdvancedCentroidClusterBase
    :members:

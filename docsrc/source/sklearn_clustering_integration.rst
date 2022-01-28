.. _integration:


Scikit Learn
=================

Relevance AI comes with a number of integrations. Below are a few examples of such
integrations. It is also relatively easy to build your own!

Clustering
-----------------------------

DBSCAN Example
################

.. code-block::

    from relevanceai import Client
    from sklearn.cluster import DBSCAN

    # instantiate the client
    client = Client()

    # Retrieve the relevant dataset
    df = client.Dataset("sample_dataset")

    model = DBSCAN()

    clusterer = df.cluster(model, alias="dbscan")

    # check that cluster is now in schema
    df.schema


Faiss KMeans Example
######################

.. code-block::

    from relevanceai import Client
    from relevanceai.clusterer import Clusterer
    from relevanceai.clusterer import CentroidClusterBase

    client = Client()

    df = client.Dataset(args.dataset_id)
    vector_field = args.vector_field
    n_clusters = int(args.n_clusters)

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

    model = FaissKMeans(model=Kmeans(d=4, k=n_clusters))

    clusterer = df.cluster(
        model=model,
        alias=f"kmeans-{n_clusters}",
        vector_fields=[vector_field]
    )

    clusterer.fit(dataset=df, vector_fields=[vector_field])

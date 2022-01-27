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

    clusterer = client.Clusterer(model, alias="dbscan")

    clusterer.fit(df, ["sample_vector_"])

    # check that cluster is now in schema
    df.schema


KMeans Example
################

.. code-block:: 

    from relevanceai import Client
    from relevanceai.clusterer import Clusterer
    from relevanceai.clusterer import CentroidClusterBase

    from sklearn.cluster import KMeans

    client = Client()

    df = client.Dataset("iris")
    vector_field = "feature_vector_"
    n_clusters = 3


    class KMeansModel(CentroidClusterBase):
        def __init__(self, model):
            self.model = model

        def fit_transform(self, vectors):
            return self.model.fit_predict(vectors)

        def get_centers(self):
            return self.model.cluster_centers_


    model = KMeansModel(model=KMeans(n_clusters=3))

    clusterer = Clusterer(model=model, alias="wofasdfho")

    clusterer.fit(dataset=df, vector_fields=[vector_field])


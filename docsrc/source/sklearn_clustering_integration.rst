.. _integration:


Native Scikit Learn
============================

Relevance AI integrates nicely with SKLearn! Below are a few examples of such
integrations. It is also relatively easy to build your own!

Clustering Algorithms
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

    clusterer = client.ClusterOps(model, alias="dbscan")

    # check that cluster is now in schema
    df.schema

K Means Example
#################

.. code-block::

    from relevanceai import Client
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=5)

    df = client.Dataset('sample')

    clusterer = client.ClusterOps(model)
    clusterer.fit_predict_update(df, vector_fields=["sample_vector_"])
    clusterer.list_closest_to_center()

Mini Batch K-Means Example
##############################

.. code-block::

    from relevanceai import Client
    from sklearn.cluster import MiniBatchKMeans
    model = MiniBatchKMeans()

    df = client.Dataset('sample')

    clusterer = client.ClusterOps(model)
    clusterer.fit_partial_predict_update(df, vector_fields=['sample_vector_'])
    clusterer.list_closest_to_center()

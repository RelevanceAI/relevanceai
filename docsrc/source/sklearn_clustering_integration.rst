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

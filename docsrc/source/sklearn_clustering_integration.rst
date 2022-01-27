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
    from sklearn.cluster import KMeans

    # instantiate the client
    client = Client()

    # Retrieve the relevant dataset
    df = client.Dataset("_github_repo_vectorai")

    from sklearn.cluster import DBSCAN
    model = DBSCAN()

    clusterer = client.Clusterer(model, alias="dbscan-integration")

    clusterer.fit(df, ['documentation_vector_'])

    # check that cluster is now in schema
    df.schema

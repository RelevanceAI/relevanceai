.. _integration:


Native Scikit Learn
============================

Relevance AI integrates nicely with SKLearn! Below are a few examples of such
integrations. It is also relatively easy to build your own!

Clustering Algorithms
-----------------------------

DBSCAN
################

.. code-block::

    from relevanceai import Client
    from relevanceai.datasets import mock_documents
    from sklearn.cluster import DBSCAN

    model = DBSCAN()

    clusterer = client.ClusterOps(model=model, alias="dbscan")

    # Retrieve the relevant dataset
    df = client.Dataset("sample_dataset_id")
    df.upsert_documents(mock_documents(10))
    clusterer.fit_predict_update(df, vector_fields=['sample_1_vector_'])

OPTICS
#################

.. code-block::

    from relevanceai import Client
    from relevanceai.datasets import mock_documents

    df = client.Dataset('sample')
    df.upsert_documents(mock_documents(100))

    from sklearn.cluster import OPTICS
    model = OPTICS()

    clusterer = client.ClusterOps(alias="optics", model=model)
    clusterer.fit_predict_update(df, vector_fields=["sample_1_vector_"])
    clusterer.list_closest_to_center()

Birch
##############################

.. code-block::

    from relevanceai import Client
    from relevanceai.datasets import mock_documents

    df = client.Dataset('sample')
    df.upsert_documents(mock_documents(100))

    from sklearn.cluster import Birch

    model = Birch()
    clusterer = client.ClusterOps(alias="birch", model=model)
    clusterer.fit_predict_update(df, vector_fields=["sample_1_vector_"])

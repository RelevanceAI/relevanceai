Explain Text Clusters
========================

This uses marginal similarity measure to explain clusters.

.. code-block::

    cluster_ops = ds.cluster(
        vector_fields=["product_title_all-mpnet-base-v2_vector_"],
        model=model
    )

    cluster_ops.explain_text_clusters(
        text_field="product_title", encode_fn_or_model="all-mpnet-base-v2", algorithm="relational"
    )

.. code-block::

    from relevanceai import Client
    client = Client()

    from relevanceai.utils.datasets import get_ecommerce_1_dataset
    docs = get_ecommerce_1_dataset()
    ds = client.Dataset("ecommerce-test")

    ds.upsert_documents(docs)

    ds.vectorize_text(["product_title"])

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=25)

    cluster_ops = ds.cluster(
        vector_fields=["product_title_all-mpnet-base-v2_vector_"],
        model=model
    )

    cluster_ops.explain_text_clusters(
        text_field="product_title", encode_fn_or_model="all-mpnet-base-v2", algorithm="relational"
    )
    # These text explanations need can be either `relational` or `centroid`

The **relational** will compare the first document against the rest and is best used to explain
why a document has been placed into this cluster with the other documents.

The **centroid** algorithm is best used to explain why a document has been placed into this cluster
based on comparing to the center vector. This has down the downside of being noisy but is a more
faithful representation of the cluster.

Explain Text Clusters
========================

This uses a marginal similarity measure to explain clusters.

.. code-block::

    from relevanceai.utils.datasets import get_ecommerce_1_dataset
    docs = get_ecommerce_1_dataset()
    ds = client.Dataset("ecommerce")

    ds.upsert_documents(docs)


    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(text):
        return enc.encode(text).tolist()

    ds['product_title'].apply(encode, output_field="product_title_minilm_vector_")

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=25)

    cluster_ops = ds.cluster(vector_fields=["product_title_minilm_vector_"], model=model)

    # we can now run text explanations
    cluster_ops.explain_text_clusters(
        text_field="product_title", encode_fn_or_model=encode, algorithm="relational"
    )

    # These text explanations need can be either `relational` or `centroid`

The **relational** will compare the first document against the rest and is best used to explain
why a document has been placed into this cluster with the other documents.

The **centroid** algorithm is best used to explain why a document has been placed into this cluster
based on comparing to the center vector. This has down the downside of being noisy but is a more
faithful representation of the cluster.

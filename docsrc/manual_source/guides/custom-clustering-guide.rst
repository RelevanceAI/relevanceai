.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0
    !pip install faiss-cpu


First, you need to set up a client object to interact with RelevanceAI.
You will need to have a dataset under your Relevance AI account. You can
either use our dummy sample data as shown below or follow the tutorial
on how to create your own dataset to create your own database.

.. code:: python

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()



Dataset
=======

.. code:: python

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    documents = get_ecommerce_dataset_encoded()
    {k:v for k, v in documents[0].items() if '_vector_' not in k}


.. code:: python

    ds = client.Dataset("faiss_kmeans_clustering")
    ds.insert_documents(documents)


.. code:: python

    ds.health()


Custom Clustering
=================

RelevanceAI supports the integration of custom clustering algorithms.
The custom algorithm can be created as the fit_transform method of the
*ClusterBase* class.

The following code shows an example of a custom clustering algorithm
that chooses randomly between Cluster 0 and Cluster 1.

.. code:: python

    # Inherit from ClusterBase to keep all the goodies!
    import numpy as np
    from faiss import Kmeans
    from relevanceai import CentroidClusterBase

    class FaissKMeans(CentroidClusterBase):
        def __init__(self, model):
            self.model = model

        def fit_predict(self, vectors):
            vectors = np.array(vectors).astype("float32")
            self.model.train(vectors)
            cluster_labels = self.model.assign(vectors)[1]
            return cluster_labels

        def metadata(self):
            return self.model.__dict__

        def get_centers(self):
            return self.model.centroids

    n_clusters = 10
    d = 512
    alias = f"faiss-kmeans_{n_clusters}"
    vector_fields = ["product_title_clip_vector_"]

    model = FaissKMeans(model=Kmeans(d=d, k=n_clusters))
    clusterer = client.ClusterOps(model=model, alias=alias)
    clusterer.operate(dataset_id="faiss_kmeans_clustering", vector_fields=vector_fields)

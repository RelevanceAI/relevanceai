Cluster
=============================


K-Means clustering
----------------------------

You can run K Means clustering using

.. code-block:: python

    from relevanceai import Client 
    client = Client()

    cluster_flow = client.KMeansClusterFlow()
    df = client.Dataset("sample")
    cluster_flow.fit(df, vector_fields=["sample_vector_"])

Custom Clustering
-----------------------------

Modifying fit_transform
**************************

You can also add custom clustering using 

.. code-block:: python 

    from relevanceai import Client 
    client = Client()
    
    from relevanceai import ClusterBase

    import random
    class RandomClusterFlow(ClusterBase):
        def __init__(self):
            pass

        # update this to update documents
        def fit_transform(self, X):
            return random.randint(0, 100)
         
    cluster_flow = client.KMeansClusterFlow()
    df = client.Dataset("sample") 
    cluster_flow.fit(df)

Modifying fit_documents
**************************

You can also modify the fitting to document behavior if your clustering algorithm
requires more than 1 field of information. For example:

.. code-block:: python 

    from relevanceai import Client 
    client = Client()
    
    from relevanceai import ClusterBase

    import random
    class RandomClusterFlow(ClusterBase):
        def __init__(self):
            pass

        # update this to update documents
        def fit_documents(self, documents, *args, **kw):
            X = self.get_field_across_documents("sample_vector_", documents)
            y = self.get_field_across_documents("entropy", documents)
            cluster_labels = self.fit_transform(documents, entropy)
            self.set_cluster_labels_across_documents(cluster_labels, documents)
        
        def fit_transform(self, X, y):
            cluster_labels = []
            for y_value in y:
                if y_value == "auto":
                    cluster_labels.append(1)
                else:
                    cluster_labels.append(random.randint(0, 100))
            return cluster_labels
            
    cluster_flow = client.KMeansClusterFlow()
    df = client.Dataset("sample") 
    cluster_flow.fit(df)


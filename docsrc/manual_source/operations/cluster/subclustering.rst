Sub Clustering
=================

Sub-clustering refers to when you are running clustering on clusters that
already exist. This can be helpful for users who want to dive deeper
into a cluster or want to cluster within a specific subset.

With Relevance AI, `sub-cluster` values are given by appending a "-{cluster_id}"
where the `cluster_id` is usually a number. For example, if you set the 
`parent_field` to be `cars` which have values `mercedes`, `tesla`, `honda` , the 
subcluster values could potentially be `mercedes-1`, `mercedes-2`, `tesla-1`, `honda-6`.

Basic
-------

The easiest to cluster is to run this: 

.. code-block::

   from relevanceai import Client 
   client = Client()
   ds = client.Dataset("sample")
   
   # Insert a dummy dataset for now
   from relevanceai.utils.datasets import mock_documents
   ds.upsert_documents(mock_documents(100))

   from sklearn.cluster import KMeans
   model = KMeans(n_clusters=5)

   cluster_ops = ds.cluster(
      model, 
      vector_fields=["sample_1_vector_"],
      alias="sample_1")
   
   # You can find the parent field in the schema or alternatively provide a field
   parent_field = "_cluster_.sample_1_vector_.sample_1"

   # Given the parent field - we now run subclustering
   ds.subcluster(
      model=model,
      parent_field=parent_field,
      vector_fields=["sample_2_vector_"],
      alias="subcluster-kmeans-2"
   )

   # You should also be able to track your subclusters using
   ds.metadata

   # Should output something like this:
   # {'_subcluster_': [{'parent_field': '_cluster_.sample_1_vector_.sample_1', 'cluster_field': '_cluster_.sample_2_vector_.subcluster-kmeans-2'}]}


   # You can also view your subcluster results using
   ds['_cluster_.sample_2_vector_.subcluster-kmeans-2']
   
You can read more about it from the API reference here:

.. automodule:: relevanceai.operations.cluster.sub
   :members:
   :exclude-members: __init__

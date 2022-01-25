:py:mod:`relevanceai.clusterer.kmeans_clusterer`
================================================

.. py:module:: relevanceai.clusterer.kmeans_clusterer

.. autoapi-nested-parse::

   KMeans Clustering



Module Contents
---------------

.. py:class:: KMeansModel(k=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')



   A Cluster Base for models to be copied off.

   .. py:method:: fit_transform(self, vectors: Union[numpy.ndarray, List])

      Fit and transform transform the vectors


   .. py:method:: metadata(self)
      :property:

      Editing the metadata of the function


   .. py:method:: get_centers(self)

      Returns centroids of clusters


   .. py:method:: get_centroid_documents(self, centroid_vector_field_name: str = 'centroid_vector_') -> List

      Get the centroid documents to store.
      If single vector field returns this:
          {
              "_id": "document-id-1",
              "centroid_vector_": [0.23, 0.24, 0.23]
          }
      If multiple vector fields returns this:
      Returns multiple
      ```
      {
          "_id": "document-id-1",
          "blue_vector_": [0.12, 0.312, 0.42],
          "red_vector_": [0.23, 0.41, 0.3]
      }

      ```




.. py:class:: KMeansClusterer(alias: str, project: str, api_key: str, k: Union[None, int] = 10, init: str = 'k-means++', n_init: int = 10, max_iter: int = 300, tol: float = 0.0001, verbose: bool = False, random_state: Optional[int] = None, copy_x: bool = True, algorithm: str = 'auto', cluster_field: str = '_cluster_')



   Run KMeans Clustering.

   :param alias: The name to call your cluster.  This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}
   :type alias: str
   :param k: The number of clusters in your K Means
   :type k: str
   :param cluster_field: The field from which to store the cluster. This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}
   :type cluster_field: str
   :param You can read about the other parameters here:
   :type You can read about the other parameters here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

   .. rubric:: Example

   >>> from relevanceai import Client
   >>> client = Client()
   >>>
   >>> clusterer = client.KMeansClusterer(alias="kmeans_cluster_sample")
   >>> df = client.Dataset("sample")
   >>> clusterer.fit(df, vector_fields=["sample_vector_"])

   .. py:method:: fit(self, dataset: Union[relevanceai.dataset_api.Dataset, str], vector_fields: List)

      Train clustering algorithm on documents and then store the labels
      inside the documents.

      :param dataset: The dataset to fit it. If string, it will create a dataset
      :type dataset: Union[str, Dataset]
      :param vector_field: The vector field of the documents
      :type vector_field: list




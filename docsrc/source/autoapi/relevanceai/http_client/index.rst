:py:mod:`relevanceai.http_client`
=================================

.. py:module:: relevanceai.http_client

.. autoapi-nested-parse::

   access the client via this class



Module Contents
---------------

.. py:data:: vis_requirements
   :annotation: = False

   

.. py:data:: vis_requirements
   :annotation: = True

   

.. py:function:: str2bool(v)


.. py:class:: Client(project=os.getenv('RELEVANCE_PROJECT'), api_key=os.getenv('RELEVANCE_API_KEY'), authenticate: bool = False)



   Batch API client

   .. py:attribute:: FAIL_MESSAGE
      :annotation: = Your API key is invalid. Please login again

      

   .. py:attribute:: build_and_plot_clusters
      

      

   .. py:method:: base_url(self)
      :property:


   .. py:method:: base_ingest_url(self)
      :property:


   .. py:method:: login(self, authenticate: bool = True)


   .. py:method:: auth_header(self)
      :property:


   .. py:method:: make_search_suggestion(self)


   .. py:method:: check_auth(self)


   .. py:method:: list_datasets(self)

      List Datasets

      .. rubric:: Example

      .. code-block::

          from relevanceai import Client
          client = Client()
          client.list_datasets()


   .. py:method:: delete_dataset(self, dataset_id)

      Delete a dataset

      :param dataset_id: The ID of a dataset
      :type dataset_id: str

      .. rubric:: Example

      .. code-block::

          from relevanceai import Client
          client = Client()
          client.delete_dataset("sample_dataset")


   .. py:method:: Clusterer(self, model: relevanceai.clusterer.ClusterBase, alias: str, cluster_field: str = '_cluster_')


   .. py:method:: KMeansClusterer(self, alias: str, k: Union[None, int] = 10, init: str = 'k-means++', n_init: int = 10, max_iter: int = 300, tol: float = 0.0001, verbose: bool = True, random_state: Optional[int] = None, copy_x: bool = True, algorithm: str = 'auto', cluster_field: str = '_cluster_')




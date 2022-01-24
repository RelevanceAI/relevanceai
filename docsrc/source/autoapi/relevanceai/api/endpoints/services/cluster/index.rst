:py:mod:`relevanceai.api.endpoints.services.cluster`
====================================================

.. py:module:: relevanceai.api.endpoints.services.cluster


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.endpoints.services.cluster.ClusterClient




.. py:class:: ClusterClient(project, api_key)

   Bases: :py:obj:`relevanceai.base._Base`

   Base class for all relevanceai client utilities

   .. py:method:: aggregate(self, dataset_id: str, vector_fields: list, metrics: list = [], groupby: list = [], sort: list = [], filters: list = [], page_size: int = 20, page: int = 1, asc: bool = False, flatten: bool = True, alias: str = 'default')

      Takes an aggregation query and gets the aggregate of each cluster in a collection. This helps you interpret each cluster and what is in them.
      It can only can be used after a vector field has been clustered.


      For more information about aggregations check out services.aggregate.aggregate.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_fields: The vector field that was clustered on
      :type vector_fields: list
      :param metrics: Fields and metrics you want to calculate
      :type metrics: list
      :param groupby: Fields you want to split the data into
      :type groupby: list
      :param filters: Query for filtering the search results
      :type filters: list
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param asc: Whether to sort results by ascending or descending order
      :type asc: bool
      :param flatten: Whether to flatten
      :type flatten: bool
      :param alias: Alias used to name a vector field. Belongs in field_{alias}vector
      :type alias: string


   .. py:method:: facets(self, dataset_id: str, facets_fields: list = [], page_size: int = 20, page: int = 1, asc: bool = False, date_interval: str = 'monthly')

      Takes a high level aggregation of every field and every cluster in a collection. This helps you interpret each cluster and what is in them.

      Only can be used after a vector field has been clustered.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param facets_fields: Fields to include in the facets, if [] then all
      :type facets_fields: list
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param asc: Whether to sort results by ascending or descending order
      :type asc: bool
      :param date_interval: Interval for date facets
      :type date_interval: string




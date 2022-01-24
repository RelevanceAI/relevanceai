:py:mod:`relevanceai.dataset_api.groupby`
=========================================

.. py:module:: relevanceai.dataset_api.groupby


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.dataset_api.groupby.Groupby
   relevanceai.dataset_api.groupby.Agg




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.dataset_api.groupby.GROUPBY_MAPPING


.. py:data:: GROUPBY_MAPPING
   

   

.. py:class:: Groupby(project, api_key, dataset_id, _pre_groupby=None)

   Bases: :py:obj:`relevanceai.api.client.BatchAPIClient`

   Batch API client

   .. py:method:: __call__(self, by: list = [])

      Instaniates Groupby Class which stores a groupby call

      :param by: List of fields to groupby
      :type by: list


   .. py:method:: _get_groupby_fields(self)

      Get what type of groupby field to use


   .. py:method:: _check_groupby_value_type(self, fields_schema)

      Check groupby fields can be grouped


   .. py:method:: _create_groupby_call(self)

      Create groupby call


   .. py:method:: mean(self, field: str)

      Convenience method to call avg metric on groupby.

      :param field: The field name to apply the mean aggregation.
      :type field: str



.. py:class:: Agg(project, api_key, dataset_id, groupby_call=[])

   Bases: :py:obj:`relevanceai.api.client.BatchAPIClient`

   Batch API client

   .. py:method:: __call__(self, metrics: dict = {}, page_size: int = 20, page: int = 1, asc: bool = False, flatten: bool = True, alias: str = 'default')

      Return aggregation query from metrics

      :param metrics: Dictionary of field and metric pairs to get
      :type metrics: dict
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param asc: Whether to sort results by ascending or descending order
      :type asc: bool
      :param flatten: Whether to flatten
      :type flatten: bool
      :param alias: Alias used to name a vector field. Belongs in field_{alias} vector
      :type alias: string


   .. py:method:: _create_metrics(self)

      Create metric call




:py:mod:`relevanceai.api.endpoints.datasets.monitor`
====================================================

.. py:module:: relevanceai.api.endpoints.datasets.monitor

.. autoapi-nested-parse::

   All Dataset related functions



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.endpoints.datasets.monitor.MonitorClient




.. py:class:: MonitorClient(project, api_key)

   Bases: :py:obj:`relevanceai.base._Base`

   Base class for all relevanceai client utilities

   .. py:method:: health(self, dataset_id: str)

      Gives you a summary of the health of your vectors, e.g. how many documents with vectors are missing, how many documents with zero vectors

      :param dataset_id: Unique name of dataset
      :type dataset_id: string


   .. py:method:: stats(self, dataset_id: str)

      All operations related to monitoring

      :param dataset_id: Unique name of dataset
      :type dataset_id: string


   .. py:method:: usage(self, dataset_id: str, filters: list = [], page_size: int = 20, page: int = 1, asc: bool = False, flatten: bool = True, log_ids: list = [])

      Aggregate the logs for a dataset.


      The response returned has the following fields:

      >>> [{'frequency': 958, 'insert_date': 1630159200000},...]

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
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
      :param log_ids: The log dataset IDs to aggregate with - one or more of logs, logs-write, logs-search, logs-task or js-logs
      :type log_ids: list




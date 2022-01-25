:py:mod:`relevanceai.dataset_api.groupby`
=========================================

.. py:module:: relevanceai.dataset_api.groupby


Module Contents
---------------

.. py:data:: GROUPBY_MAPPING
   

   

.. py:class:: Groupby(project, api_key, dataset_id, _pre_groupby=None)



   Batch API client

   .. py:method:: mean(self, field: str)

      Convenience method to call avg metric on groupby.

      :param field: The field name to apply the mean aggregation.
      :type field: str



.. py:class:: Agg(project, api_key, dataset_id, groupby_call=[])



   Batch API client



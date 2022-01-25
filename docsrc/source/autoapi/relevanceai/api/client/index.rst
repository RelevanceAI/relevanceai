:py:mod:`relevanceai.api.client`
================================

.. py:module:: relevanceai.api.client

.. autoapi-nested-parse::

   Batch client to allow for batch insertions/retrieval and encoding



Module Contents
---------------

.. py:class:: BatchAPIClient(project, api_key)



   Batch API client

   .. py:method:: batch_insert(self)


   .. py:method:: batch_get_and_edit(self, dataset_id: str, chunk_size: int, bulk_edit: Callable)

      Batch get the documents and return the documents




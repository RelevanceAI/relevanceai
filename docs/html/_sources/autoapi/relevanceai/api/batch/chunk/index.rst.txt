:py:mod:`relevanceai.api.batch.chunk`
=====================================

.. py:module:: relevanceai.api.batch.chunk

.. autoapi-nested-parse::

   Chunk Helper functions



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.chunk.Chunker




.. py:class:: Chunker

   Update the chunk Mixins

   .. py:method:: chunk(self, documents: Union[pandas.DataFrame, List], chunksize: int = 20)

      Chunk an iterable object in Python.

      Example:

      >>> documents = [{...}]
      >>> ViClient.chunk(documents)

      :param documents: List of dictionaries/Pandas dataframe
      :param chunksize: The chunk size of an object.




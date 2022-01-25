:py:mod:`relevanceai.vector_tools.nearest_neighbours`
=====================================================

.. py:module:: relevanceai.vector_tools.nearest_neighbours


Module Contents
---------------

.. py:data:: doc_utils
   

   

.. py:class:: NearestNeighbours(project: str, api_key: str)



   Base class for all relevanceai client utilities

   .. py:method:: get_nearest_neighbours(docs: list, vector: list, vector_field: str, distance_measure_mode: relevanceai.vector_tools.constants.NEAREST_NEIGHBOURS = 'cosine', callable_distance=None)
      :staticmethod:




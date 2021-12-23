:py:mod:`relevanceai.vector_tools.nearest_neighbours`
=====================================================

.. py:module:: relevanceai.vector_tools.nearest_neighbours


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.vector_tools.nearest_neighbours.NearestNeighbours




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.vector_tools.nearest_neighbours.doc_utils


.. py:data:: doc_utils
   

   

.. py:class:: NearestNeighbours(project: str, api_key: str)

   Bases: :py:obj:`relevanceai.base._Base`, :py:obj:`doc_utils.doc_utils.DocUtils`

   Base class for all relevanceai client utilities

   .. py:method:: get_nearest_neighbours(docs: list, vector: list, vector_field: str, distance_measure_mode: relevanceai.vector_tools.constants.NEAREST_NEIGHBOURS = 'cosine', callable_distance=None)
      :staticmethod:




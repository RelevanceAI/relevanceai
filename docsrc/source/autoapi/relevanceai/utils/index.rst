:py:mod:`relevanceai.utils`
===========================

.. py:module:: relevanceai.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.utils.Utils




.. py:class:: Utils(project, api_key)

   Bases: :py:obj:`relevanceai.api.endpoints.client.APIClient`, :py:obj:`relevanceai.base._Base`, :py:obj:`doc_utils.DocUtils`

   API Client

   .. py:method:: _is_valid_vector_name(self, dataset_id, vector_name: str) -> bool

      Check vector field name is valid


   .. py:method:: _is_valid_label_name(self, dataset_id, label_name: str) -> bool

      Check vector label name is valid. Checks that it is either numeric or text


   .. py:method:: _remove_empty_vector_fields(self, docs, vector_field: str) -> List[Dict]

      Remove documents with empty vector fields


   .. py:method:: _convert_id_to_string(self, docs)


   .. py:method:: _are_fields_in_schema(self, fields, dataset_id, schema=None)

      Check fields are in schema




:py:mod:`relevanceai.api.endpoints.services.services`
=====================================================

.. py:module:: relevanceai.api.endpoints.services.services

.. autoapi-nested-parse::

   Services class



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.endpoints.services.services.ServicesClient




.. py:class:: ServicesClient(project: str, api_key: str)

   Bases: :py:obj:`relevanceai.base._Base`

   Base class for all relevanceai client utilities

   .. py:method:: document_diff(self, doc: dict, docs_to_compare: list, difference_fields: list = [])

      Find differences between documents

      :param doc: Main document to compare other documents against.
      :type doc: dict
      :param docs_to_compare: Other documents to compare against the main document.
      :type docs_to_compare: list
      :param difference_fields: Fields to compare. Defaults to [], which compares all fields.
      :type difference_fields: list




:py:mod:`relevanceai.errors`
============================

.. py:module:: relevanceai.errors

.. autoapi-nested-parse::

   Missing field error



Module Contents
---------------

.. py:exception:: RelevanceAIError

   Bases: :py:obj:`Exception`

   Base class for all errors


.. py:exception:: MissingFieldError

   Bases: :py:obj:`RelevanceAIError`

   Error handling for missing fields


.. py:exception:: APIError

   Bases: :py:obj:`RelevanceAIError`

   Error related to API


.. py:exception:: ClusteringResultsAlreadyExistsError(field_name, message='Clustering results for %s already exist')

   Bases: :py:obj:`RelevanceAIError`

   Exception raised for existing clustering results

   .. attribute:: message -- explanation of the error

      

   .. py:method:: __str__(self)

      Return str(self).




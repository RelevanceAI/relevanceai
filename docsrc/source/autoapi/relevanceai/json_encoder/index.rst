:py:mod:`relevanceai.json_encoder`
==================================

.. py:module:: relevanceai.json_encoder

.. autoapi-nested-parse::

   Json Encoder utility

   To invoke JSON encoder:

   ```
       from relevanceai import json_encoder
   ```



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.json_encoder.JSONEncoderUtils




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.json_encoder.ENCODERS_BY_TYPE


.. py:data:: ENCODERS_BY_TYPE
   

   

.. py:class:: JSONEncoderUtils

   .. py:method:: json_encoder(self, obj)

      Converts object so it is json serializable
      If you want to add your own mapping,
      customize it this way;

      .. rubric:: Example

      YOu can use our JSON encoder easily.
      >>> docs = [{"value": np.nan}]
      >>> client.json_encoder(docs)

      If you want to use FastAPI's json encoder, do this:
      >>> from fastapi import jsonable_encoder
      >>> client.json_encoder = jsonable_encoder




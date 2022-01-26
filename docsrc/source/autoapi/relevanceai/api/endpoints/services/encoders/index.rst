:py:mod:`relevanceai.api.endpoints.services.encoders`
=====================================================

.. py:module:: relevanceai.api.endpoints.services.encoders


Module Contents
---------------

.. py:class:: EncodersClient(project: str, api_key: str)



   Base class for all relevanceai client utilities

   .. py:method:: textimage(self, text: str)

      Encode text to make searchable with images

      :param text: Text to encode
      :type text: string


   .. py:method:: text(self, text: str)

      Encode text

      :param text: Text to encode
      :type text: string


   .. py:method:: multi_text(self, text)

      Encode multilingual text

      :param text: Text to encode
      :type text: string


   .. py:method:: image(self, image)

      Encode an image

      :param image: URL of image to encode
      :type image: string


   .. py:method:: imagetext(self, image)

      Encode an image to make searchable with text

      :param image: URL of image to encode
      :type image: string




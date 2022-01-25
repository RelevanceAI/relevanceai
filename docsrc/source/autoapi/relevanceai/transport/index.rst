:py:mod:`relevanceai.transport`
===============================

.. py:module:: relevanceai.transport

.. autoapi-nested-parse::

   The Transport Class defines a transport as used by the Channel class to communicate with the network.



Module Contents
---------------

.. py:data:: DO_NOT_REPEAT_STATUS_CODES
   

   

.. py:class:: Transport



   Base class for all relevanceai objects

   .. py:attribute:: project
      :annotation: :str

      

   .. py:attribute:: api_key
      :annotation: :str

      

   .. py:attribute:: config
      :annotation: :relevanceai.config.Config

      

   .. py:attribute:: logger
      :annotation: :relevanceai.logger.AbstractLogger

      

   .. py:method:: auth_header(self)
      :property:


   .. py:method:: DASHBOARD_TYPES(self)
      :property:


   .. py:method:: print_dashboard_url(self, dashboard_url)


   .. py:method:: make_http_request(self, endpoint: str, method: str = 'GET', parameters: dict = {}, base_url: str = None, output_format=None)

      Make the HTTP request
      :param endpoint: The endpoint from the documentation to use
      :type endpoint: string
      :param method_type: POST or GET request
      :type method_type: string




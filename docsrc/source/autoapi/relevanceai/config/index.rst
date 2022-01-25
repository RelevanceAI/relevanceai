:py:mod:`relevanceai.config`
============================

.. py:module:: relevanceai.config

.. autoapi-nested-parse::

   Configuration Settings



Module Contents
---------------

.. py:data:: PATH
   

   

.. py:data:: CONFIG_PATH
   

   

.. py:class:: Config



   Set and change configuration settings

   - Retries - Set the behaviour of retries for failed responses from the API
       - number_of_retries - Number of retries to attempt
       - seconds_between_retries - Seconds to wait between retries

   - Logging - Set the behaviour of logging
        - enable_logging - Whether to log
        - log_to_file - Whether to log to file
        - log_file_name - The name of the file to log to, if logging to file
        - logging_level - Minimum level to log

   - Upload - Set the behaviour of uploads to RelevanceAI
       - target_chunk_mb - Maximum upload size per request

   - API - Set the behaviour of API requests
       - base_url - The base url to access
       - output_format - The format of API responses

   - Dashboard - URLS to various things


   .. py:method:: options(self)
      :property:

      View all current config settings


   .. py:method:: get_option(self, option)

      View current config settings

      :param option: Setting key
      :type option: string


   .. py:method:: set_option(self, option, value)

      Change a config settings

      :param option: Setting key
      :type option: string
      :param value: New setting
      :type value: string


   .. py:method:: reset_to_default(self)

      Reset config to default



.. py:data:: CONFIG
   

   


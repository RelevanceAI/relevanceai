:py:mod:`relevanceai.api.batch.local_logger`
============================================

.. py:module:: relevanceai.api.batch.local_logger

.. autoapi-nested-parse::

   Local logger for pull_update_push.



Module Contents
---------------

.. py:class:: PullUpdatePushLocalLogger(filename: Union[str, bytes])



   This logger class is specifically for pull_update_push to log
   failures locally as opposed to on the cloud.

   .. py:method:: log_ids(self, id_list, verbose: bool = True)

      Log the failed IDs to the file


   .. py:method:: count_ids_in_fn(self) -> int

      Returns total count of failed IDs




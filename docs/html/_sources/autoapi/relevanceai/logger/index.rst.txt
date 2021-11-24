:py:mod:`relevanceai.logger`
============================

.. py:module:: relevanceai.logger


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.logger.AbstractLogger
   relevanceai.logger.LoguruLogger



Functions
~~~~~~~~~

.. autoapisummary::

   relevanceai.logger.str2bool



.. py:function:: str2bool(v)


.. py:class:: AbstractLogger

   Base Logging Instance

   .. py:attribute:: info
      :annotation: :Callable

      

   .. py:attribute:: error
      :annotation: :Callable

      

   .. py:attribute:: success
      :annotation: :Callable

      

   .. py:attribute:: debug
      :annotation: :Callable

      

   .. py:attribute:: warning
      :annotation: :Callable

      

   .. py:attribute:: critical
      :annotation: :Callable

      

   .. py:attribute:: warn
      :annotation: :Callable

      


.. py:class:: LoguruLogger

   Bases: :py:obj:`AbstractLogger`

   Using verbose loguru as base logger for now

   .. py:method:: logger(self)
      :property:


   .. py:method:: _init_logger(self)




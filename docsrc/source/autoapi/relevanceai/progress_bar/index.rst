:py:mod:`relevanceai.progress_bar`
==================================

.. py:module:: relevanceai.progress_bar

.. autoapi-nested-parse::

   Get a good progress bar



Module Contents
---------------

.. py:class:: ProgressBar

   .. py:method:: is_in_ipython()
      :staticmethod:

      Determines if current code is executed within an ipython session.


   .. py:method:: is_in_notebook(self) -> bool

      Determines if current code is executed from an ipython notebook.


   .. py:method:: get_bar(self)



.. py:class:: NullProgressBar(iterable: int = None)



   Context manager that does no additional processing.

   Used as a stand-in for a normal context manager, when a particular
   block of code is only sometimes used with a normal context manager:

   cm = optional_cm if condition else nullcontext()
   with cm:
       # Perform operation, using optional_cm if condition is True


.. py:function:: progress_bar(iterable, show_progress_bar: bool = False)



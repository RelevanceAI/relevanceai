:py:mod:`relevanceai.data_tools.base_text_processing`
=====================================================

.. py:module:: relevanceai.data_tools.base_text_processing


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.data_tools.base_text_processing.BaseTextProcessing




.. py:class:: BaseTextProcessing

   .. py:method:: normalize_text(txt: str, lower: bool = True, remove_digit: bool = True, remove_punct: bool = True) -> str
      :staticmethod:

      * Lower-casing
      * Digit removal
      * Punctuation removal


   .. py:method:: get_word_frequency(str_list: List[str], remove_stop_words: bool = True, additional_stop_words: List[str] = [], language='english') -> List
      :staticmethod:

      Returns a sorted word frequency in Python




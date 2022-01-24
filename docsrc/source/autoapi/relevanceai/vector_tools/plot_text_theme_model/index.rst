:py:mod:`relevanceai.vector_tools.plot_text_theme_model`
========================================================

.. py:module:: relevanceai.vector_tools.plot_text_theme_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.vector_tools.plot_text_theme_model.PlotTextThemeModel



Functions
~~~~~~~~~

.. autoapisummary::

   relevanceai.vector_tools.plot_text_theme_model.build_and_plot_clusters



.. py:class:: PlotTextThemeModel(project: str, api_key: str, dataset_id: str, upload_chunksize: int = 50, cluster_field: str = '_cluster_', embedding_dims: int = 2, dim_red_k: int = 800, n_epochs_without_progress: int = 100, language: str = 'english')

   Bases: :py:obj:`relevanceai.api.client.BatchAPIClient`, :py:obj:`relevanceai.data_tools.base_text_processing.BaseTextProcessing`, :py:obj:`relevanceai.logger.LoguruLogger`, :py:obj:`doc_utils.DocUtils`

   Batch API client

   .. py:method:: _build_and_plot_clusters(self, vector_fields: List[str], text_fields: List[str], max_doc_num: int = None, k: int = 10, alias: str = 'kmeans', lower: bool = True, remove_digit: bool = True, remove_punct: bool = True, remove_stop_words: bool = True, additional_stop_words: List[str] = [], cluster_representative_cnt: int = 3, plot_axis: str = 'off', figsize: Tuple[int, Ellipsis] = (20, 10), cmap: str = 'plasma', alpha: float = 0.2)


   .. py:method:: _get_documents(self, vector_fields: List[str], text_fields: List[str], max_doc_num: int = None)


   .. py:method:: _batch_load_docs(self, fields: List[str], filters: List[dict] = [], page_size: int = 200, cursor: str = None)


   .. py:method:: _kmeans_clustering(self, docs: List[dict], vector_fields: List[str], k: int = 10, alias: str = 'kmeans')


   .. py:method:: _get_fields_value_by_id(docs: List[dict], id: str, fields: List[str])
      :staticmethod:


   .. py:method:: _get_cluster_datafield(self, docs: List[dict], vector_fields: List[str], text_fields: List[str], alias: str, lower: bool = True, remove_digit: bool = True, remove_punct: bool = True)


   .. py:method:: _get_cluster_population(cluster_data: dict)
      :staticmethod:


   .. py:method:: _get_cluster_word_freq(self, cluster_data: dict, remove_stop_words: bool = True, additional_stop_words: List[str] = [], cluster_representative_cnt: int = 3)


   .. py:method:: _dim_reduction(self, vector_data: List, embedding_dims=2, k=800, n_epochs_without_progress=100, run_fit=True)


   .. py:method:: _plot_clusters(self, docs: List[dict], dr_docs: List, centers: List[dict], cluster_data: dict, vector_fields: List[str], alias: str, cluster_representative_cnt: int, plot_axis: str = 'off', figsize: Tuple[int, Ellipsis] = (20, 10), cmap: str = 'plasma', alpha: float = 0.2)



.. py:function:: build_and_plot_clusters(self, project: str, api_key: str, dataset_id: str, vector_fields: List[str], text_fields: List[str], upload_chunksize: int = 50, cluster_field: str = '_cluster_', embedding_dims: int = 2, dim_red_k: int = 800, n_epochs_without_progress: int = 100, language: str = 'english', max_doc_num: int = None, k: int = 10, alias: str = 'kmeans', lower: bool = True, remove_digit: bool = True, remove_punct: bool = True, remove_stop_words: bool = True, additional_stop_words: List[str] = [], cluster_representative_cnt: int = 3, plot_axis: str = 'off', figsize: Tuple[int, Ellipsis] = (20, 10), cmap: str = 'plasma', alpha: float = 0.2)



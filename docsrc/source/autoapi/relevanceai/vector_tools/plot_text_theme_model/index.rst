:py:mod:`relevanceai.vector_tools.plot_text_theme_model`
========================================================

.. py:module:: relevanceai.vector_tools.plot_text_theme_model


Module Contents
---------------

.. py:class:: PlotTextThemeModel(project: str, api_key: str, dataset_id: str, upload_chunksize: int = 50, cluster_field: str = '_cluster_', embedding_dims: int = 2, dim_red_k: int = 800, n_epochs_without_progress: int = 100, language: str = 'english')



   Batch API client


.. py:function:: build_and_plot_clusters(self, project: str, api_key: str, dataset_id: str, vector_fields: List[str], text_fields: List[str], upload_chunksize: int = 50, cluster_field: str = '_cluster_', embedding_dims: int = 2, dim_red_k: int = 800, n_epochs_without_progress: int = 100, language: str = 'english', max_doc_num: int = None, k: int = 10, alias: str = 'kmeans', lower: bool = True, remove_digit: bool = True, remove_punct: bool = True, remove_stop_words: bool = True, additional_stop_words: List[str] = [], cluster_representative_cnt: int = 3, plot_axis: str = 'off', figsize: Tuple[int, Ellipsis] = (20, 10), cmap: str = 'plasma', alpha: float = 0.2)



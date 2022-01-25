:py:mod:`relevanceai.vector_tools.dim_reduction`
================================================

.. py:module:: relevanceai.vector_tools.dim_reduction


Module Contents
---------------

.. py:class:: DimReductionBase



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: numpy.ndarray, dr_args: Dict[Any, Any], dims: int) -> numpy.ndarray
      :abstractmethod:



.. py:class:: PCA



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: numpy.ndarray, dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS['pca'], dims: int = 3) -> numpy.ndarray



.. py:class:: TSNE



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: numpy.ndarray, dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS['tsne'], dims: int = 3) -> numpy.ndarray



.. py:class:: UMAP



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: numpy.ndarray, dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS['umap'], dims: int = 3) -> numpy.ndarray



.. py:class:: Ivis



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: numpy.ndarray, dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS['tsne'], dims: int = 3) -> numpy.ndarray



.. py:class:: DimReduction(project, api_key)



   Base class for all relevanceai client utilities

   .. py:method:: dim_reduce(vectors: numpy.ndarray, dr: Union[relevanceai.vector_tools.constants.DIM_REDUCTION, DimReductionBase], dr_args: Union[None, dict], dims: typing_extensions.Literal[2, 3]) -> numpy.ndarray
      :staticmethod:

      Dimensionality reduction




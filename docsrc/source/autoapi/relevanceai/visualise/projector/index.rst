:py:mod:`relevanceai.visualise.projector`
=========================================

.. py:module:: relevanceai.visualise.projector


Module Contents
---------------

.. py:data:: chained_assignment
   

   

.. py:data:: RELEVANCEAI_BLUE
   :annotation: = #1854FF

   

.. py:class:: Projector(project, api_key)



   Projector class.

   .. rubric:: Example

   >>> from relevanceai import Client
   >>> project = input()
   >>> api_key = input()
   >>> client = Client(project, api_key)
   >>> client.projector.plot(
           dataset_id, vector_field, number_of_points_to_render, random_state,
           dr, dr_args, dims,
           vector_label, label_char_length,
           color_label, colour_label_char_length,
           hover_label,
           cluster, cluster_args,
           )

   .. py:method:: plot(self, dataset_id: str, vector_field: str, number_of_points_to_render: int = 1000, vector_label: relevanceai.vector_tools.constants.Union[None, str] = None, dr: relevanceai.vector_tools.constants.Union[relevanceai.vector_tools.constants.DIM_REDUCTION, relevanceai.vector_tools.dim_reduction.DimReductionBase] = 'pca', dims: relevanceai.vector_tools.constants.Literal[2, 3] = 3, dr_args: relevanceai.vector_tools.constants.Union[None, relevanceai.vector_tools.constants.Dict] = None, cluster: relevanceai.vector_tools.constants.Union[relevanceai.vector_tools.constants.CLUSTER, relevanceai.vector_tools.cluster.ClusterBase] = None, num_clusters: relevanceai.vector_tools.constants.Union[None, int] = 10, cluster_args: relevanceai.vector_tools.constants.Dict = {}, cluster_on_dr: bool = False, hover_label: list = [], show_image: bool = False, label_char_length: int = 50, marker_size: int = 5)

      Dimension reduce vectors and plot them

      To write your own custom dimensionality reduction, you should inherit from DimReductionBase:
      from relevanceai.visualise.dim_reduction import DimReductionBase
      class CustomDimReduction(DimReductionBase):
          def fit_transform(self, vectors):
              return np.arange(512, 2)

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> project = input()
      >>> api_key = input()
      >>> client = Client(project, api_key)
      >>> client.projector.plot(
              dataset_id, vector_field, number_of_points_to_render, random_state,
              dr, dr_args, dims,
              vector_label, label_char_length,
              color_label, colour_label_char_length,
              hover_label,
              cluster, cluster_args,
              )

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_field: Vector field to plot
      :type vector_field: list
      :param number_of_points_to_render: Number of vector fields to plot
      :type number_of_points_to_render: int
      :param vector_label: Field to use as label to describe vector on plot
      :type vector_label: string
      :param dr: Method of dimension reduction for vectors
      :type dr: string
      :param dims: Number of dimensions to reduce to
      :type dims: int
      :param dr_args: Additional arguments for dimension reduction
      :type dr_args: dict
      :param cluster: Method of clustering for vectors
      :type cluster: string
      :param num_clusters: Number of clusters to create
      :type num_clusters: string
      :param cluster_args: Additional arguments for clustering
      :type cluster_args: dict
      :param cluster_on_dr: Whether to cluster on the dimension reduced or original vectors
      :type cluster_on_dr: int
      :param hover_label: Additional labels to include as plot labels
      :type hover_label: list
      :param show_image: Whether vector labels are image urls
      :type show_image: bool
      :param label_char_length: Maximum length of text for each hover label
      :type label_char_length: int
      :param marker_size: Marker size of the plot
      :type marker_size: int


   .. py:method:: plot_with_jupyter_dash(self, dataset_id: str, vector_field: str, number_of_points_to_render: int = 1000, vector_label: relevanceai.vector_tools.constants.Union[None, str] = None, dr: relevanceai.vector_tools.constants.Union[relevanceai.vector_tools.constants.DIM_REDUCTION, relevanceai.vector_tools.dim_reduction.DimReductionBase] = 'pca', dims: relevanceai.vector_tools.constants.Literal[2, 3] = 3, dr_args: relevanceai.vector_tools.constants.Union[None, relevanceai.vector_tools.constants.Dict] = None, cluster: relevanceai.vector_tools.constants.Union[relevanceai.vector_tools.constants.CLUSTER, relevanceai.vector_tools.cluster.ClusterBase] = None, num_clusters: relevanceai.vector_tools.constants.Union[None, int] = 10, cluster_args: relevanceai.vector_tools.constants.Dict = {}, cluster_on_dr: bool = False, hover_label: list = [], show_image: bool = False, label_char_length: int = 50, marker_size: int = 5, interactive: bool = True)

      Dimension reduce vectors and plot them using Jupyter Dash, with functionality to visualise different clusters and nearest neighbours

      To write your own custom dimensionality reduction, you should inherit from DimReductionBase:
      from relevanceai.visualise.dim_reduction import DimReductionBase
      class CustomDimReduction(DimReductionBase):
          def fit_transform(self, vectors):
              return np.arange(512, 2)

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> project = input()
      >>> api_key = input()
      >>> client = Client(project, api_key)
      >>> client.projector.plot(
              dataset_id, vector_field, number_of_points_to_render, random_state,
              dr, dr_args, dims,
              vector_label, label_char_length,
              color_label, colour_label_char_length,
              hover_label,
              cluster, cluster_args,
              )

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_field: Vector field to plot
      :type vector_field: list
      :param number_of_points_to_render: Number of vector fields to plot
      :type number_of_points_to_render: int
      :param vector_label: Field to use as label to describe vector on plot
      :type vector_label: string
      :param dr: Method of dimension reduction for vectors
      :type dr: string
      :param dims: Number of dimensions to reduce to
      :type dims: int
      :param dr_args: Additional arguments for dimension reduction
      :type dr_args: dict
      :param cluster: Method of clustering for vectors
      :type cluster: string
      :param num_clusters: Number of clusters to create
      :type num_clusters: string
      :param cluster_args: Additional arguments for clustering
      :type cluster_args: dict
      :param cluster_on_dr: Whether to cluster on the dimension reduced or original vectors
      :type cluster_on_dr: int
      :param hover_label: Additional labels to include as plot labels
      :type hover_label: list
      :param show_image: Whether vector labels are image urls
      :type show_image: bool
      :param label_char_length: Maximum length of text for each hover label
      :type label_char_length: int
      :param marker_size: Marker size of the plot
      :type marker_size: int
      :param interactive: Whether to include interactive features including nearest neighbours
      :type interactive: bool


   .. py:method:: plot_from_docs(self, docs: relevanceai.vector_tools.constants.List[relevanceai.vector_tools.constants.Dict], vector_field: str, vector_label: relevanceai.vector_tools.constants.Union[None, str] = None, dr: relevanceai.vector_tools.constants.Union[relevanceai.vector_tools.constants.DIM_REDUCTION, relevanceai.vector_tools.dim_reduction.DimReductionBase] = 'pca', dims: relevanceai.vector_tools.constants.Literal[2, 3] = 3, dr_args: relevanceai.vector_tools.constants.Union[None, relevanceai.vector_tools.constants.Dict] = None, cluster: relevanceai.vector_tools.constants.Union[relevanceai.vector_tools.constants.CLUSTER, relevanceai.vector_tools.cluster.ClusterBase] = None, num_clusters: relevanceai.vector_tools.constants.Union[None, int] = 10, cluster_args: relevanceai.vector_tools.constants.Dict = {}, cluster_on_dr: bool = False, hover_label: list = [], show_image: bool = False, label_char_length: int = 50, marker_size: int = 5, dataset_name: relevanceai.vector_tools.constants.Union[None, str] = None, jupyter_dash=False, interactive: bool = True)




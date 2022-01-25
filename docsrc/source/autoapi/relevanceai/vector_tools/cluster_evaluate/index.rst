:py:mod:`relevanceai.vector_tools.cluster_evaluate`
===================================================

.. py:module:: relevanceai.vector_tools.cluster_evaluate


Module Contents
---------------

.. py:data:: SILHOUETTE_INFO
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        
        Good clusters have clusters which are highly seperated and elements within which are highly cohesive. <br/>
        <b>Silohuette Score</b> is a metric from <b>-1 to 1</b> that calculates the average cohesion and seperation of each element, with <b>1</b> being clustered perfectly, <b>0</b> being indifferent and <b>-1</b> being clustered the wrong way

    .. raw:: html

        </details>

   

.. py:data:: RAND_INFO
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        Good clusters have elements, which, when paired, belong to the same cluster label and same ground truth label. <br/>
        <b>Rand Index</b> is a metric from <b>0 to 1</b> that represents the percentage of element pairs that have a matching cluster and ground truth labels with <b>1</b> matching perfect and <b>0</b> matching randomly. <br/> <i>Note: This measure is adjusted for randomness so does not equal the exact numerical percentage.</i>

    .. raw:: html

        </details>

   

.. py:data:: HOMOGENEITY_INFO
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        Good clusters only have elements from the same ground truth within the same cluster<br/>
        <b>Homogeneity</b> is a metric from <b>0 to 1</b> that represents whether clusters contain only elements in the same ground truth with <b>1</b> being perfect and <b>0</b> being absolutely incorrect.

    .. raw:: html

        </details>

   

.. py:data:: COMPLETENESS_INFO
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        Good clusters have all elements from the same ground truth within the same cluster <br/>
        <b>Completeness</b> is a metric from <b>0 to 1</b> that represents whether clusters contain all elements in the same ground truth with <b>1</b> being perfect and <b>0</b> being absolutely incorrect.

    .. raw:: html

        </details>

   

.. py:data:: METRIC_DESCRIPTION
   

   

.. py:function:: sort_dict(dict, reverse: bool = True, cut_off=0)


.. py:class:: ClusterEvaluate(project, api_key)



   Batch API client

   .. py:method:: plot(self, dataset_id: str, vector_field: str, cluster_alias: str, ground_truth_field: str = None, description_fields: list = [], marker_size: int = 5)

      Plot the vectors in a collection to compare performance of cluster labels, optionally, against ground truth labels

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_field: The vector field that was clustered upon
      :type vector_field: string
      :param cluster_alias: The alias of the clustered labels
      :type cluster_alias: string
      :param ground_truth_field: The field to use as ground truth
      :type ground_truth_field: string
      :param description_fields: List of fields to use as additional labels on plot
      :type description_fields: list
      :param marker_size: Size of scatterplot marker
      :type marker_size: int


   .. py:method:: metrics(self, dataset_id: str, vector_field: str, cluster_alias: str, ground_truth_field: str = None)

      Determine the performance of clusters through the Silhouette Score, and optionally against ground truth labels through Rand Index, Homogeneity and Completeness

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_field: The vector field that was clustered upon
      :type vector_field: string
      :param cluster_alias: The alias of the clustered labels
      :type cluster_alias: string
      :param ground_truth_field: The field to use as ground truth
      :type ground_truth_field: string


   .. py:method:: distribution(self, dataset_id: str, vector_field: str, cluster_alias: str, ground_truth_field: str = None, transpose=False)

      Determine the distribution of clusters, optionally against the ground truth

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_field: The vector field that was clustered upon
      :type vector_field: string
      :param cluster_alias: The alias of the clustered labels
      :type cluster_alias: string
      :param ground_truth_field: The field to use as ground truth
      :type ground_truth_field: string
      :param transpose: Whether to transpose cluster and ground truth perspectives
      :type transpose: bool


   .. py:method:: centroid_distances(self, dataset_id: str, vector_field: str, cluster_alias: str, distance_measure_mode: relevanceai.vector_tools.constants.CENTROID_DISTANCES = 'cosine', callable_distance=None)

      Determine the distances of centroid from each other

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param vector_field: The vector field that was clustered upon
      :type vector_field: string
      :param cluster_alias: The alias of the clustered labels
      :type cluster_alias: string
      :param distance_measure_mode: Distance measure to compare cluster centroids
      :type distance_measure_mode: string
      :param callable_distance: Optional function to use for distance measure
      :type callable_distance: func


   .. py:method:: plot_from_docs(vectors: list, cluster_labels: list, ground_truth: list = None, vector_description: dict = None, marker_size: int = 5)
      :staticmethod:

      Plot the vectors in a collection to compare performance of cluster labels, optionally, against ground truth labels

      :param vectors: List of vectors which were clustered upon
      :type vectors: list
      :param cluster_labels: List of cluster labels corresponding to the vectors
      :type cluster_labels: list
      :param ground_truth: List of ground truth labels for the vectors
      :type ground_truth: list
      :param vector_description: Dictionary of fields and their values to describe the vectors
      :type vector_description: dict
      :param marker_size: Size of scatterplot marker
      :type marker_size: int


   .. py:method:: metrics_from_docs(vectors, cluster_labels, ground_truth=None)
      :staticmethod:

      Determine the performance of clusters through the Silhouette Score, and optionally against ground truth labels through Rand Index, Homogeneity and Completeness

      :param vectors: List of vectors which were clustered upon
      :type vectors: list
      :param cluster_labels: List of cluster labels corresponding to the vectors
      :type cluster_labels: list
      :param ground_truth: List of ground truth labels for the vectors
      :type ground_truth: list


   .. py:method:: label_distribution_from_docs(label)
      :staticmethod:

      Determine the distribution of a label

      :param label: List of labels
      :type label: list


   .. py:method:: label_joint_distribution_from_docs(label_1, label_2)
      :staticmethod:

      Determine the distribution of a label against another label

      :param label_1: List of labels
      :type label_1: list
      :param label_2: List of labels
      :type label_2: list


   .. py:method:: centroid_distances_from_docs(centroids, distance_measure_mode: relevanceai.vector_tools.constants.CENTROID_DISTANCES = 'cosine', callable_distance=None)
      :staticmethod:

      Determine the distances of centroid from each other

      :param centroids: Dictionary containing cluster name and centroid
      :type centroids: dict
      :param distance_measure_mode: Distance measure to compare cluster centroids
      :type distance_measure_mode: string
      :param callable_distance: Optional function to use for distance measure
      :type callable_distance: func


   .. py:method:: silhouette_score(vectors, cluster_labels)
      :staticmethod:


   .. py:method:: adjusted_rand_score(ground_truth, cluster_labels)
      :staticmethod:


   .. py:method:: completeness_score(ground_truth, cluster_labels)
      :staticmethod:


   .. py:method:: homogeneity_score(ground_truth, cluster_labels)
      :staticmethod:




Visualisation
================

Embedding Projector
------------------

To use the Embedding projector - 

See [`relevanceai/visualise/constants.py`]('./relevanceai/visualise/constants.py') for default args . available

.. code-block:: python

    from relevanceai import Client

    client = Client()

    '''
    Retrieve docs in dataset  set `number_of_points_to_render = None` to retrieve all docs
    '''

    vector_label = "product_name"
    vector_field = "product_name_imagetext_vector_"

    dr = 'pca'
    cluster = 'kmeans'

    client.projector.plot(
        dataset_id="ecommerce-6", 
        vector_field=vector_field,
        number_of_points_to_render=1000,
    )  


Full options and more details on functionality, see [this notebook](https://colab.research.google.com/drive/1ONEjcIf1CqUhXy8dknlyAnp1DnSAYHnR?usp=sharing) here - 


.. code-block:: python

    '''
    If `cluster` specified, will override `colour_label` option and render cluster as legend
    '''

    dr = 'tsne'
    cluster = 'kmedoids'

    dataset_id = "ecommerce-6"
    vector_label = "product_name"
    vector_field = "product_name_imagetext_vector_"

    client.projector.plot(
        dataset_id = dataset_id,
        vector_field = vector_field,
        number_of_points_to_render=1000,
        
        ### Dimensionality reduction args
        dr = dr,
        dr_args = DIM_REDUCTION_DEFAULT_ARGS[ dr ], 

        ## Plot rendering args
        vector_label = None, 
        colour_label = vector_label,
        hover_label = None,
        
        ### Cluster args
        cluster = cluster,
        cluster_args = CLUSTER_DEFAULT_ARGS[ cluster ],
        num_clusters = 20
    )

Full options and more details on functionality, see [this notebook](https://colab.research.google.com/drive/1ONEjcIf1CqUhXy8dknlyAnp1DnSAYHnR?usp=sharing) here.

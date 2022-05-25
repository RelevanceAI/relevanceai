🔢 Vectorizing Guide
====================

Firstly, we must import what we need from Relevance AI

.. code:: python

    from relevanceai import Client
    from relevanceai.utils.datasets import (
        get_iris_dataset,
        get_palmer_penguins_dataset,
        get_online_ecommerce_dataset,
    )

.. code:: python

    client = Client()

Example 1
---------

For this first example we going to work with a purely numeric dataset.
The Iris dataset contains 4 numeric features and another text column
with the label

.. code:: python

    iris_documents = get_iris_dataset()

.. code:: python

    dataset = client.Dataset("iris")


.. parsed-literal::

    ⚠️ Your dataset has no documents. Make sure to insert some!


.. code:: python

    dataset.insert_documents(iris_documents, create_id=True)


.. parsed-literal::

    while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/iris/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


Here we can see the dataset schema, pre-vectorization

.. code:: python

    dataset.schema




.. parsed-literal::

    {'PetalLengthCm': 'numeric',
     'PetalWidthCm': 'numeric',
     'SepalLengthCm': 'numeric',
     'SepalWidthCm': 'numeric',
     'Species': 'text',
     'insert_date_': 'date'}



Vectorizing is as simple specifying ``create_feature_vector=True``

While species is a text feature, we do not need to vectorize this.
Besides, smart typechecking recognises this field as a text field we
would not usually vectorize.

``create_feature_vector=True`` is what creates our “document” vectors.
This concatenates all numeric/vector fields in a single “document”
vector. This new vector_field is always called
``f"_dim{n_dims}_feature_vector_"``, with n_dims being the size of the
concatenated vector.

Furthermore, for nuermic stability accross algorithms, sklearn’s
StandardScaler is applied to the concatenated vector field. If the
concatenated size of a vector field is >512 dims, PCA is automatically
applied.

.. code:: python

    dataset.vectorize(create_feature_vector=True)


.. parsed-literal::

    No fields were given, vectorizing the following field(s):
    Concatenating the following fields to form a feature vector: PetalLengthCm, PetalWidthCm, SepalLengthCm, SepalWidthCm



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Concatenated field is called _dim4_feature_vector_



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    ✅ All documents inserted/edited successfully.
    The following vector fields were added: _dim4_feature_vector_
    Concatenating the following fields to form a feature vector: PetalLengthCm, PetalWidthCm, SepalLengthCm, SepalWidthCm



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Concatenated field is called _dim4_feature_vector_



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


or
~~

.. code:: python

    dataset.vectorize(fields=["numeric"], create_feature_vector=True)

You can see below that the dataset schema has been altered accordingly

.. code:: python

    dataset.schema




.. parsed-literal::

    {'PetalLengthCm': 'numeric',
     'PetalWidthCm': 'numeric',
     'SepalLengthCm': 'numeric',
     'SepalWidthCm': 'numeric',
     'Species': 'text',
     '_dim4_feature_vector_': {'vector': 4},
     'insert_date_': 'date'}



Example 2
---------

For this second example we going to work with a mixed numeric and text
dataset. The Palmer Penguins dataset contains several numeric features
and another text column called “Comments”

.. code:: python

    penguins_documents = get_palmer_penguins_dataset()

.. code:: python

    dataset.insert_documents(penguins_documents, create_id=True)


.. parsed-literal::

    while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/iris/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


We must install the default Encoders for text vectorizing from vectorhub

.. code:: python

    !pip install vectorhub[encoders-text-tfhub-windows] # If you are on windows

.. code:: python

    !pip install vectorhub[encoders-text-tfhub] # other

No arguments automatically detects what text and image fieds are presetn
in your dataset. Since this is a new function, its typechecking could be
faulty. If need be, specifiy the data types in the same format as the
schema with ``_text_`` denoting text_fields and ``_image_`` denoting
image fields.

.. code:: python

    dataset.vectorize()


.. parsed-literal::

    No fields were given, vectorizing the following field(s): Comments, Species, Stage
    This operation will create the following vector_fields: ['Comments_use_vector_', 'Species_use_vector_', 'Stage_use_vector_']



.. parsed-literal::

      0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    📌 Your logs have been saved to iris_13-04-2022-04-11-09_pull_update_push.log. If you are debugging, you can turn file logging off by setting `log_to_file=False`.📌
    ✅ All documents inserted/edited successfully.
    The following vector fields were added: Species_use_vector_, Stage_use_vector_


or
~~

.. code:: python

    dataset.vectorize(fields=["Comments"], create_feature_vector=True)


.. parsed-literal::

    This operation will create the following vector_fields: ['Comments_use_vector_']



.. parsed-literal::

      0%|          | 0/3 [00:00<?, ?it/s]


.. parsed-literal::

    Concatenating the following fields to form a feature vector: Comments_use_vector_



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Concatenated field is called _dim512_feature_vector_



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    ✅ All documents inserted/edited successfully.
    The following vector fields were added: _dim512_feature_vector_

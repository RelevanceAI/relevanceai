🔢 Vectorizing Guide
===================

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



Vectorizing is as simple specifying ``feature_vector=True``

While species is a text feature, we do not need to vectorize this.
Besides, smart typechecking recognises this field as a text field we
would not usually vectorize.

``feature_vector=True`` is what creates our “document” vectors. This
concatenates all numeric/vector fields in a single “document” vector.
This new vector_field is always called
``f"_dim{n_dims}_feature_vector_"``, with n_dims being the size of the
concatenated vector.

Furthermore, for nuermic stability accross algorithms, sklearn’s
StandardScaler is applied to the concatenated vector field. If the
concatenated size of a vector field is >512 dims, PCA is automatically
applied.

.. code:: python

    dataset.vectorize(feature_vector=True)


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
--

.. code:: python

    dataset.vectorize(fields=["numeric"], feature_vector=True)

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


.. parsed-literal::

    Requirement already satisfied: vectorhub[encoders-text-tfhub-windows] in /usr/local/lib/python3.7/dist-packages (1.8.3)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (1.19.5)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (2.23.0)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (6.0)
    Requirement already satisfied: document-utils in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (1.7.1)
    Requirement already satisfied: tf-models-official==2.4.0 in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (2.4.0)
    Requirement already satisfied: tensorflow-hub~=0.12.0 in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (0.12.0)
    Requirement already satisfied: tensorflow~=2.4.3 in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (2.4.4)
    Requirement already satisfied: bert-for-tf2==0.14.9 in /usr/local/lib/python3.7/dist-packages (from vectorhub[encoders-text-tfhub-windows]) (0.14.9)
    Requirement already satisfied: params-flow>=0.8.0 in /usr/local/lib/python3.7/dist-packages (from bert-for-tf2==0.14.9->vectorhub[encoders-text-tfhub-windows]) (0.8.2)
    Requirement already satisfied: py-params>=0.9.6 in /usr/local/lib/python3.7/dist-packages (from bert-for-tf2==0.14.9->vectorhub[encoders-text-tfhub-windows]) (0.10.2)
    Requirement already satisfied: oauth2client in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (4.1.3)
    Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (4.0.1)
    Requirement already satisfied: pandas>=0.22.0 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.3.5)
    Requirement already satisfied: tensorflow-addons in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.16.1)
    Requirement already satisfied: tensorflow-model-optimization>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.7.2)
    Requirement already satisfied: seqeval in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.2.2)
    Requirement already satisfied: py-cpuinfo>=3.3.0 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (8.0.0)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.15.0)
    Requirement already satisfied: google-api-python-client>=1.6.7 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.12.11)
    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.1.96)
    Requirement already satisfied: google-cloud-bigquery>=0.31.0 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.21.0)
    Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (4.5.5.64)
    Requirement already satisfied: pycocotools in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (2.0.4)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.4.1)
    Requirement already satisfied: gin-config in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.5.0)
    Requirement already satisfied: psutil>=5.4.3 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (5.4.8)
    Requirement already satisfied: tf-slim>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.1.0)
    Requirement already satisfied: Cython in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.29.28)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (3.2.2)
    Requirement already satisfied: dataclasses in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.6)
    Requirement already satisfied: kaggle>=1.3.9 in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.5.12)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (7.1.2)
    Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (3.0.1)
    Requirement already satisfied: google-api-core<3dev,>=1.21.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.26.3)
    Requirement already satisfied: google-auth<3dev,>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.35.0)
    Requirement already satisfied: google-auth-httplib2>=0.0.3 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.0.4)
    Requirement already satisfied: httplib2<1dev,>=0.15.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.17.4)
    Requirement already satisfied: protobuf>=3.12.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (3.17.3)
    Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.56.0)
    Requirement already satisfied: packaging>=14.3 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (21.3)
    Requirement already satisfied: pytz in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (2018.9)
    Requirement already satisfied: setuptools>=40.3.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (57.4.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (4.2.4)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (4.8)
    Requirement already satisfied: google-resumable-media!=0.4.0,<0.5.0dev,>=0.3.1 in /usr/local/lib/python3.7/dist-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.4.1)
    Requirement already satisfied: google-cloud-core<2.0dev,>=1.0.3 in /usr/local/lib/python3.7/dist-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.0.3)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (6.1.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (4.63.0)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (2.8.2)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.24.3)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (2021.10.8)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=14.3->google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (3.0.7)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.4.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->vectorhub[encoders-text-tfhub-windows]) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->vectorhub[encoders-text-tfhub-windows]) (2.10)
    Requirement already satisfied: absl-py~=0.10 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (0.15.0)
    Requirement already satisfied: termcolor~=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.1.0)
    Requirement already satisfied: h5py~=2.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (2.10.0)
    Requirement already satisfied: typing-extensions~=3.7.4 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (3.7.4.3)
    Requirement already satisfied: wheel~=0.35 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (0.37.1)
    Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (0.3.3)
    Requirement already satisfied: tensorflow-estimator<2.5.0,>=2.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (2.4.0)
    Requirement already satisfied: tensorboard~=2.4 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (2.8.0)
    Requirement already satisfied: wrapt~=1.12.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.12.1)
    Requirement already satisfied: astunparse~=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.6.3)
    Requirement already satisfied: opt-einsum~=3.3.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (3.3.0)
    Requirement already satisfied: grpcio~=1.32.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.32.0)
    Requirement already satisfied: keras-preprocessing~=1.1.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.1.2)
    Requirement already satisfied: flatbuffers~=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.12)
    Requirement already satisfied: google-pasta~=0.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (0.2.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.0.1)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (3.3.6)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.8.1)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (0.6.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (0.4.6)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (4.11.3)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (3.7.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.4->tensorflow~=2.4.3->vectorhub[encoders-text-tfhub-windows]) (3.2.0)
    Requirement already satisfied: dm-tree~=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow-model-optimization>=0.4.1->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.1.6)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.4.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle>=1.3.9->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.3)
    Requirement already satisfied: scikit-learn>=0.21.3 in /usr/local/lib/python3.7/dist-packages (from seqeval->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.0.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (3.1.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.1.0)
    Requirement already satisfied: typeguard>=2.7 in /usr/local/lib/python3.7/dist-packages (from tensorflow-addons->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (2.7.1)
    Requirement already satisfied: promise in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (2.3)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.16.0)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (5.4.0)
    Requirement already satisfied: attrs>=18.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (21.4.0)
    Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (1.7.0)
    Requirement already satisfied: dill in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official==2.4.0->vectorhub[encoders-text-tfhub-windows]) (0.3.4)


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
--

.. code:: python

    dataset.vectorize(fields=["Comments"], feature_vector=True)


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


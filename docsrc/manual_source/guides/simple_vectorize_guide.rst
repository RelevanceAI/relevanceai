😊🔢 Simple Vectorizing Guide
===========================

This notebook will be a beginners guide to getting start with vectorized
unstructured data with RelevanceAI

Firstly, we must import what we need from Relevance AI

.. code:: python

    from relevanceai import Client
    from relevanceai.utils.datasets import (
        get_online_ecommerce_dataset,
    )

Prerequisites
~~~~~~~~~~~~~

This tutorial assumes that you have instantiaed the RelevanceAI python
SDK client and are working with the provided
``online_ecommerce_dataset``

.. code:: python

    client = Client()

    dataset = client.Dataset("ecommerce")

    documents = get_online_ecommerce_dataset()

    dataset.insert_documents(documents, create_id=True)

Vectorize with One Line of Code
-------------------------------

In the ``online_ecommerce_dataset``, there are a few text fields; namely
``product_title`` and ``product_description``. To vectorize with the
default text model (``all-MiniLM-L6-v2``), simply run the code below:

.. code:: python

    dataset.vectorize_text(fields=["product_title", "product_description"])

Also in ``online_ecommerce_dataset`` is an image field called
``product_image``. To vectorize with the default image model (``clip``),
simply run the code below:

.. code:: python

    dataset.vectorize_image(fields=["product_image"])

HuggingFace Transformers Integration
------------------------------------

Our vectorize functions come with built in integrations with most models
from hugging face. Simply specify the model string from the hugging face
model page and run the code below as such.

Say I wanted to use ``bert-base-uncased`` instead of the default…

.. code:: python

    dataset.vectorize_text(
        fields=["product_title"], models=["bert-base-uncased"]
    )  # creates one new vector

    dataset.vectorize_text(
        fields=["product_title"], models=["bert-base-uncased", "all-MiniLM-L6-v2"]
    )  # creates two new vectors

The above also applies to vectorizing images

How to bring your own Vectorizer
--------------------------------

RelevanceAI also supports the bringing your own custom vectorizers.
Simply import the base model class and write your own logic around it

.. code:: python

    import random

    from relevanceai.operations_new.vectorize.ops import VectorizeOps


    class CustomVectorizeOps(VectorizeOps):
        def __init__(self, field):
            super().__init__()

            self.field = field

            self.vector_length = 64
            self.model = lambda field: [random.random() for _ in range(self.vector_length)]

        @property
        def vector_name(self):
            return "custom_vector_"

        def transform(self, documents):

            for document in documents:
                vector = self.model(document[self.field])
                document[self.vector_name] = vector

            return documents


    custom_field = "sample_text_field"
    ops = CustomVectorizeOps(custom_field)

    chunksize = 20  # You can increase this value depending on how computationally expensive your vectorizer is
    ops.run(
        dataset,
        batched=True,
        chunksize=chunksize,
    )

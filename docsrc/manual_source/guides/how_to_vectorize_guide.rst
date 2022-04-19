⚡ How To Use Custom Vectorizers
===============================

|Open In Colab|

Installation
------------

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/general-features/how-to-vectorize/_notebooks/RelevanceAI_ReadMe_How_to_Vectorize.ipynb

.. code:: ipython3

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0

.. code:: ipython3

    from relevanceai import Client
    
    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()

Vectorizing using Vectorhub
---------------------------

.. code:: ipython3

    ## Installing Vectorhub with access to Tfhub models!
    # remove `!` if running the line in a terminal
    !pip install vectorhub[encoders-text-tfhub]

.. code:: ipython3

    ## Installing Vectorhub with access to Sentence Transformer models
    # remove `!` if running the line in a terminal
    !pip install vectorhub[sentence-transformers]

.. code:: ipython3

    ## Installing Vectorhub with access to Huggingface models
    # remove `!` if running the line in a terminal
    !pip install vectorhub[transformers]

Setup
=====

.. code:: ipython3

    ## Let's use a sample Sentence Transformer model for encoding
    from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec
    
    model = SentenceTransformer2Vec("all-mpnet-base-v2")

Encoding
~~~~~~~~

.. code:: ipython3

    # Encode a single input
    model.encode("I love working with vectors.")
    
    # documents are saved as a list of dictionaries
    documents = [
        {"sentence": '"This is the first sentence."', "_id": 1},
        {"sentence": '"This is the second sentence."', "_id": 2},
    ]
<<<<<<< HEAD
    
=======

>>>>>>> development
    # Encode the `"sentence"` field in a list of documents
    encoded_documents = model.encode_documents(["sentence"], documents)

Encoding an entire dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to update an existing dataset with encoding results is
to run ``encode_documents``. This function fetches all the data-points
in a dataset, runs the specified function (i.e. encoding in this case)
and writes the result back to the dataset.

For instance, in the sample code below, we use a dataset called
``ecommerce_dataset``, and encodes the ``product_description`` field
using the ``USE2Vec`` encoder. You can see the list of the available
list of models for vectorising here using
`Vectorhub <https://github.com/RelevanceAI/vectorhub>`__ or feel free to
bring your own model(s).

.. code:: ipython3

    import pandas as pd
    from relevanceai.utils.datasets import get_ecommerce_dataset_clean
    
    # Retrieve our sample dataset. - This comes in the form of a list of documents.
    documents = get_ecommerce_dataset_clean()
    
    pd.DataFrame.from_dict(documents).head()
    ds = client.Dataset("quickstart_example_encoding")
    ds.insert_documents(documents)
<<<<<<< HEAD
    
=======

>>>>>>> development
    ds["product_title"].apply(
        lambda x: model.encode(x), output_field="product_title_vector_"
    )

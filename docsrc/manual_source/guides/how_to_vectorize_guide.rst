⚡ How To Vectorize
================================

|Open In Colab|

Installation
------------

You can install the Relevance AI using

::

   pip install -q RelevanceAI

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/general-features/how-to-vectorize/_notebooks/RelevanceAI_ReadMe_How_to_Vectorize.ipynb

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

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()

.. code:: ipython3

    from relevanceai.utils import get_ecommerce_1_dataset
    dataset_id = "ecommerce-2"
    documents = get_ecommerce_1_dataset(number_of_documents=100)

.. code:: ipython3

    ds = client.Dataset(dataset_id)
    ds.delete()
    ds.insert_documents(documents, create_id=True)

.. code:: ipython3

    ds.vectorize_text(
        models=["princeton-nlp/sup-simcse-roberta-large"],
        fields=["product_text"]
    )

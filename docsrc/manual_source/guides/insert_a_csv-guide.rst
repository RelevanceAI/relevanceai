|Open In Colab|

Installation
============

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/general-features/creating-a-dataset/_notebooks/creating-a-dataset.ipynb

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0


Setup
=====

.. code:: python

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()




Create a dataset
================

Uploading a CSV file is as simple as specifying the CSV path to your
file. If there is no ``_id`` field in your dataset, we will
automatically generate it for you, unless you specify otherwise via
``auto_generate_id=False``. If there is a unique identifier in your
dataset which can be used as the ``_id`` field, pass it to the insert
function through the ``col_for_id`` field (e.g. ``col_for_id="ref_no"``)

.. code:: python

    ds = client.Dataset('quickstart_insert_csv')
    csv_fpath = "./sample_data/california_housing_test.csv"
    ds.insert_csv(filepath_or_buffer = csv_fpath)


.. code:: python

    ds.schema

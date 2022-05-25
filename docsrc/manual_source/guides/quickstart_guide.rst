🏃‍♀️ Quickstart
===============

Use `Relevance AI <https://cloud.relevance.ai/>`__ for clustering and
gaining meaning from your unstructured data.

✨ Example
---------

An example cluster app that showcases meaning amongst each group of
unstructured data With just a few lines of code, you’ll get rich,
interactive, shareable dashboards `which you can see yourself
here <https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png>`__.
|image0|

.. |image0| image:: https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png

🔒 Data & Privacy
~~~~~~~~~~~~~~~~

We take security very seriously, and our cloud-hosted dashboard uses
industry standard best practices for encryption. Our team adhere to our
`strict privacy policy <https://relevance.ai/data-security-policy/>`__.

--------------

🪄 Install ``RelevanceAI`` library and authenticate the client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by installing the library and logging in to your account.

.. code:: ipython3

    !pip install RelevanceAI -qqq

.. code:: ipython3

    from relevanceai import Client

    # Instantiate the client and authenticate
    client = Client()

    # This will prompt a link to collect your API token which includes your project and API key


.. parsed-literal::

    Activation token (you can find it here: https://cloud.relevance.ai/sdk/api )

    Connecting to ap-southeast-2...
    You can view all your datasets at https://cloud.relevance.ai/datasets/
    Welcome to RelevanceAI. Logged in as fc103ba5498da02f86a3.


📩 Upload Some Data
~~~~~~~~~~~~~~~~~~

1️⃣. Open a new **Dataset**

2️⃣. **Insert** some documents

.. code:: ipython3

    dataset_id = "retail_reviews"  # The dataset name that we have decided, this can be whatever you want for your own data
    dataset = client.Dataset(dataset_id=dataset_id)  # Instantiate the dataset

.. code:: ipython3

    from relevanceai.utils.datasets import ExampleDatasets

    documents = ExampleDatasets._get_dummy_dataset("retail_reviews_small")

    dataset.insert_documents(documents)



.. parsed-literal::

      0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/retail_reviews/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


👨‍🔬 Vectorizing
---------------

💪 In order to better visualise clusters within our data, we must
vectorise the unstructured fields in a our clusters. In this dataset,
there are two important text fields, both located in the review body;
These are the ``reviews.text`` and ``reviews.title``. For the purposes
of this tutorial, we will be vectorizing ``reviews.text`` only.

🤔 Choosing a Vectorizer
~~~~~~~~~~~~~~~~~~~~~~~

An important part of vectorizing text is around choosing which
vectorizer to use. Relevance AI allows for a custom vectorizer from
vectorhub, but if you can’t decide, the default models for each type of
unstructured data are listed below.

-  Text: ``USE2Vec``
-  Images: ``Clip2Vec``

First we install the suite of vectorizers from vectorhub

.. code:: ipython3

    !pip install vectorhub[encoders-text-tfhub] -qqq

🤩 Vectorize in one line
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # The text fields here are the ones we wish to construct vector representations for
    text_fields = ["reviews.text"]
    vector_fields = dataset.vectorize(text_fields=text_fields)["added_vectors"]

😎 Custom Vectorizer
~~~~~~~~~~~~~~~~~~~

For this example we will encode text using ``SentenceTransformers``. If
following this tutorial, and you completed the above step, you can skip
vectorizing with ``SentenceTransformer2Vec``.

.. code:: ipython3

    # Other vectorizers will come from vectorhub should you wish to choose a different vectorizer
    from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec

    # For this example we will use the mpnet base
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer2Vec(model_name=model_name)

    # Same process of vectorizing as before, just add the `text_model` parameter
    text_fields = ["reviews.text"]
    dataset.vectorize(text_fields=text_fields, text_model=model)

✨ Cluster Application
---------------------

In one line of code, we can create a cluster application based on our
new vector field. This application is how we will discover insights
about the semantic groups in our data.

🤔 Choosing the Number of Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most clustering algorithms require you choose the number clusters you
wish to find. This can be tricky if you don’t know what the expect.
Luckily, RelevanceAI uses a clustering algorithm called community
detection that does not require the number of clusters to be set.
Instead, the algorithm will decide how many is right for you. To
discover more about other clustering methods, `read
here <https://relevanceai.readthedocs.io/en/latest/relevanceai.cluster_report.html>`__

.. code:: ipython3

    model = "community_detection"
    alias = "my_clustering"

    dataset.cluster(model=model, alias=alias, vector_fields=vector_fields)

🔗 The above step will produce a link to your first cluster app!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click the link provided to view your newly generated clusters in a
`dashboard
app <https://cloud.relevance.ai/dataset/retail_reviews/deploy/cluster/59066979f4876d91beea/QVdEaHJuOEJ5Qy1VVnVsVDhndjM6eG9HaVg2RGtTTUdWNXFFQjNhZUg0QQ/LZpGq38B8_iiYmskWDEn/us-east-1/>`__
|image0|

.. |image0| image:: https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png

Want to quickly create some example applications with Relevance AI?
Check out some other guides below! - `Text-to-image search with OpenAI’s
CLIP <https://docs.relevance.ai/docs/quickstart-text-to-image-search>`__
- `Hybrid Text search with Universal Sentence Encoder using
Vectorhub <https://docs.relevance.ai/docs/quickstart-text-search>`__ -
`Text search with Universal Sentence Encoder Question Answer from
Google <https://docs.relevance.ai/docs/quickstart-question-answering>`__

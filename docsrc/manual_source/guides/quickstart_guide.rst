🏃‍♀️ Quickstart
=============

Use `Relevance AI <https://cloud.relevance.ai/>`__ for clustering and
gaining meaning from your unstructured data.

✨ Example
----------

An example cluster app that showcases meaning amongst each group of
unstructured data With just a few lines of code, you’ll get rich,
interactive, shareable dashboards `which you can see yourself
here <https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png>`__.
|image1|

.. |image1| image:: https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png

🔒 Data & Privacy
~~~~~~~~~~~~~~~~~

We take security very seriously, and our cloud-hosted dashboard uses
industry standard best practices for encryption. Our team adhere to our
`strict privacy policy <https://relevance.ai/data-security-policy/>`__.

--------------

🪄 Install ``RelevanceAI`` library and authenticate the client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by installing the library and logging in to your account.

.. code:: ipython3

    !pip install RelevanceAI -qqq


.. parsed-literal::

    [K     |████████████████████████████████| 249 kB 15.8 MB/s
    [K     |████████████████████████████████| 255 kB 63.6 MB/s
    [K     |████████████████████████████████| 1.1 MB 56.6 MB/s
    [K     |████████████████████████████████| 58 kB 5.8 MB/s
    [K     |████████████████████████████████| 144 kB 70.9 MB/s
    [K     |████████████████████████████████| 271 kB 70.4 MB/s
    [K     |████████████████████████████████| 94 kB 2.9 MB/s
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.8.0 requires tf-estimator-nightly==2.8.0.dev2021122109, which is not installed.
    arviz 0.11.4 requires typing-extensions<4,>=3.7.4.3, but you have typing-extensions 4.0.1 which is incompatible.[0m
    [?25h

.. code:: ipython3

    from relevanceai import Client

    # Instantiate the client and authenticate
    client = Client()

    # This will prompt a link to collect your API token which includes your project and API key


.. parsed-literal::

    /usr/local/lib/python3.7/dist-packages/relevanceai/__init__.py:49: UserWarning: We noticed you don't have the latest version!
    We recommend updating to the latest version (1.4.3) to get all bug fixes and newest features!
    You can do this by running pip install -U relevanceai.
    Changelog: https://relevanceai.readthedocs.io/en/2.0.0/changelog.html.
      warnings.warn(MESSAGE)


.. parsed-literal::

    Activation token (you can find it here: https://cloud.relevance.ai/sdk/api )
    Activation token:··········
    Connecting to us-east-1...
    You can view all your datasets at https://cloud.relevance.ai/datasets/
    Welcome to RelevanceAI. Logged in as 59066979f4876d91beea.


📩 Upload Some Data
~~~~~~~~~~~~~~~~~~~

1️⃣. Open a new **Dataset**

2️⃣. **Insert** some documents

.. code:: ipython3

    dataset_id = "retail_reviews"  # The dataset name that we have decided, this can be whatever you want for your own data
    dataset = client.Dataset(dataset_id=dataset_id)  # Instantiate the dataset

.. code:: ipython3

    import gdown  # Since the example data is located in google drive, we use gdown to retrieve

    # In a real workload, this step can be substituted for loading your own .csv
    # dataset link: https://data.world/datafiniti/grammar-and-online-product-reviews

    dataset_small = "1SZ1EqBZQG132yaAaV0doxuGDZo7PdT2B"  # 5K files
    output = "data_small.zip"
    gdown.download(id=dataset_small, output=output, quiet=False)

    dataset_large = "1eQwJy4nbIontA7qEe344lgBl3Una5Vlg"  # 71K files
    output = "data_large.zip"
    gdown.download(id=dataset_large, output=output, quiet=False)


.. parsed-literal::

    Downloading...
    From: https://drive.google.com/uc?id=1SZ1EqBZQG132yaAaV0doxuGDZo7PdT2B
    To: /content/data_small.zip
    100%|██████████| 869k/869k [00:00<00:00, 121MB/s]
    Downloading...
    From: https://drive.google.com/uc?id=1eQwJy4nbIontA7qEe344lgBl3Una5Vlg
    To: /content/data_large.zip
    100%|██████████| 3.87M/3.87M [00:00<00:00, 150MB/s]




.. parsed-literal::

    'data_large.zip'



.. code:: ipython3

    !unzip data_small.zip # Our data is a .csv wrapped in .zip, so we must extract


.. parsed-literal::

    Archive:  data_small.zip
    replace data_small.csv? [y]es, [n]o, [A]ll, [N]one, [r]ename: y
      inflating: data_small.csv
    Archive:  data_large.zip
      inflating: data_large.csv


.. code:: ipython3

    dataset.insert_csv("data_small.csv")  # RelevanceAI uses one line of code to insert .csv


.. parsed-literal::

    while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/retail_reviews/dashboard/monitor/
    ✅ All documents inserted/edited successfully.




.. parsed-literal::

    {'failed_documents': [], 'failed_documents_detailed': [], 'inserted': 5000}



👨‍🔬 Vectorizing
--------------

💪 In order to better visualise clusters within our data, we must
vectorise the unstructured fields in a our clusters. In this dataset,
there are two important text fields, both located in the review body;
These are the ``reviews.text`` and ``reviews.title``. For the purposes
of this tutorial, we will be vectorizing ``reviews.text`` only.

🤔 Choosing a Vectorizer
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # The text fields here are the ones we wish to construct vector representations for
    text_fields = ["reviews.text"]
    vector_fields = dataset.vectorize(text_fields=text_fields)["added_vectors"]


.. parsed-literal::

    /usr/local/lib/python3.7/dist-packages/relevanceai/package_utils/version_decorators.py:20: UserWarning: This function currently in beta and may change in the future.
      warnings.warn("This function currently in beta and may change in the future.")



.. parsed-literal::

      0%|          | 0/50 [00:00<?, ?it/s]


.. parsed-literal::

    ✅ All documents inserted/edited successfully.
    The following vector was added: reviews.text_use_vector_


😎 Custom Vectorizer
~~~~~~~~~~~~~~~~~~~~

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
----------------------

In one line of code, we can create a cluster application based on our
new vector field. This application is how we will discover insights
about the semantic groups in our data.

🤔 Choosing the Number of Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


.. parsed-literal::

    Retrieving all documents



.. parsed-literal::

      0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    Fitting and predicting on all documents
    Updating the database...
    Inserting centroid documents...
    Build your clustering app here: https://cloud.relevance.ai/dataset/retail_reviews/deploy/recent/cluster




.. parsed-literal::

    <relevanceai.workflows.cluster_ops.clusterops.ClusterOps at 0x7f5054aa3150>



🔗 The above step will produce a link to your first cluster app!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click the link provided to view your newly generated clusters in a
`dashboard
app <https://cloud.relevance.ai/dataset/retail_reviews/deploy/cluster/59066979f4876d91beea/QVdEaHJuOEJ5Qy1VVnVsVDhndjM6eG9HaVg2RGtTTUdWNXFFQjNhZUg0QQ/LZpGq38B8_iiYmskWDEn/us-east-1/>`__
|image1|

.. |image1| image:: https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png

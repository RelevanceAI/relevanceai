|Open In Colab|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/getting-started/_notebooks/RelevanceAI-ReadMe-Quick-Feature-Tour.ipynb

1. Set up Relevance AI
~~~~~~~~~~~~~~~~~~~~~~

Get started using our RelevanceAI SDK and use of
`Vectorhub <https://hub.getvectorai.com/>`__\ ’s `CLIP
model <https://hub.getvectorai.com/model/text_image%2Fclip>`__ for
encoding.

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0
    # remove `!` if running the line in a terminal
    !pip install -U vectorhub[clip]


Follow the signup flow and get your credentials below otherwise, you can
sign up/login and find your credentials in the settings
`here <https://auth.relevance.ai/signup/?callback=https%3A%2F%2Fcloud.relevance.ai%2Flogin%3Fredirect%3Dcli-api>`__

.. image:: https://drive.google.com/uc?id=131M2Kpz5s9GmhNRnqz6b0l0Pw9DHVRWs

.. code:: python


    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()




.. image:: https://drive.google.com/uc?id=1owtvwZKTTcrOHBlgKTjqiMOvrN3DGrF6

2. Create a dataset and insert data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use one of our sample datasets to upload into your own project!

.. code:: python

    import pandas as pd
    from relevanceai.utils.datasets import get_ecommerce_dataset_clean

    # Retrieve our sample dataset. - This comes in the form of a list of documents.
    documents = get_ecommerce_dataset_clean()

    pd.DataFrame.from_dict(documents).head()


.. code:: python


    ds = client.Dataset("quickstart")
    ds.insert_documents(documents)


See your dataset in the dashboard

.. image:: https://drive.google.com/uc?id=1nloY4S8R1B8GY2_QWkb0BGY3bLrG-8D-

3. Encode data and upload vectors into your new dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encode a new product image vector using
`Vectorhub’s <https://hub.getvectorai.com/>`__ ``Clip2Vec`` models and
update your dataset with the resulting vectors. Please refer to
`Vectorhub <https://github.com/RelevanceAI/vectorhub>`__ for more
details.

.. code:: python

    from vectorhub.bi_encoders.text_image.torch import Clip2Vec

    model = Clip2Vec()

    # Set the default encode to encoding an image
    model.encode = model.encode_image
    documents = model.encode_documents(fields=['product_image'], documents=documents)



.. code:: python

    ds.upsert_documents(documents=documents)


.. code:: python

    ds.schema


Monitor your vectors in the dashboard

.. image:: https://drive.google.com/uc?id=1d2jhjhwvPucfebUphIiqGVmR1Td2uYzM

4. Run clustering on your vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run clustering on your vectors to better understand your data!

You can view your clusters in our clustering dashboard following the
link which is provided after the clustering is finished!

.. code:: python

    from sklearn.cluster import KMeans

    cluster_model = KMeans(n_clusters=10)
    ds.cluster(cluster_model, ["product_image_clip_vector_"])


You can see the new ``_cluster_`` field that is added to your document
schema. Clustering results are uploaded back to the dataset as an
additional field. The default ``alias`` of the cluster will be the
``kmeans_<k>``.

.. code:: python

    ds.schema


See your cluster centers in the dashboard

.. image:: https://drive.google.com/uc?id=1P0ZJcTd-Kl7TUwzFHEe3JuJpf_cTTP6J

4. Run a vector search
~~~~~~~~~~~~~~~~~~~~~~

Encode your query and find your image results!

Here our query is just a simple vector query, but our search comes with
out of the box support for features such as multi-vector, filters,
facets and traditional keyword matching to combine with your vector
search. You can read more about how to construct a multivector query
with those features
`here <https://docs.relevance.ai/docs/vector-search-prerequisites>`__.

See your search results on the dashboard here
https://cloud.relevance.ai/sdk/search.

.. code:: python


    query = "gifts for the holidays"
    query_vector = model.encode(query)
    multivector_query=[
        { "vector": query_vector, "fields": ["product_image_clip_vector_"]}
    ]
    results = ds.vector_search(
        multivector_query=multivector_query,
        page_size=10
    )


See your multi-vector search results in the dashboard

.. image:: https://drive.google.com/uc?id=1qpc7oK0uxj2IRm4a9giO5DBey8sm8GP8

Want to quickly create some example applications with Relevance AI?
Check out some other guides below! - `Text-to-image search with OpenAI’s
CLIP <https://docs.relevance.ai/docs/quickstart-text-to-image-search>`__
- `Hybrid Text search with Universal Sentence Encoder using
Vectorhub <https://docs.relevance.ai/docs/quickstart-text-search>`__ -
`Text search with Universal Sentence Encoder Question Answer from
Google <https://docs.relevance.ai/docs/quickstart-question-answering>`__

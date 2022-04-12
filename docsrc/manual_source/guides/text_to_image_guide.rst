🌇 Text To Image Search QuickStart
=================================

|Open In Colab|

`Try the image search live in Relevance AI
Dashboard <https://cloud.relevance.ai/demo/search/image-to-text>`__.

In this notebook we will show you how to create and experiment with a
powerful text to image search engine using OpenAI's CLIP and Relevance
AI.

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/getting-started/example-applications/_notebooks/RelevanceAI-ReadMe-Text-to-Image-Search.ipynb

What I Need
===========

-  Project & API Key (The SDK will link you to the corresponding page or
   you can grab your API key from https://cloud.relevance.ai/ in the
   settings area)
-  Python 3
-  Relevance AI Installed as shown below. For more information visit
   `Installation guide <https://docs.relevance.ai/docs>`__

Installation Requirements
-------------------------

.. code:: ipython3

    # Relevance AI installation
    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0
    !pip install ftfy regex tqdm
    !pip install git+https://github.com/openai/CLIP.git


Client Setup
------------

You can sign up/login and find your credentials here:
https://cloud.relevance.ai/sdk/api Once you have signed up, click on the
value under ``Activation token`` and paste it here

.. code:: ipython3

    from relevanceai import Client
    client = Client()

Text-to-image search
====================

To enable text-to-image search we will be using Relevance AI as the
vector database and OpenAI's CLIP as the vectorizer, to vectorize text
and images into CLIP vector embeddings.

1) Data
-------

For this quickstart we will be using a sample e-commerce dataset.
Alternatively, you can use your own dataset for the different steps.

.. code:: ipython3

    import pandas as pd
    from relevanceai.utils.datasets import get_ecommerce_dataset_clean

    # Retrieve our sample dataset. - This comes in the form of a list of documents.
    documents = get_ecommerce_dataset_clean()
    pd.DataFrame.from_dict(documents).head()

2) Encode / Vectorize with CLIP
-------------------------------

CLIP is a vectorizer from OpenAI that is trained to find similarities
between text and image pairs. In the code below we set up CLIP.

.. code:: ipython3

    import torch
    import clip
    import requests
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # First - let's encode the image based on CLIP
    def encode_image(image):
        # Let us download the image and then preprocess it
        image = preprocess(Image.open(requests.get(image, stream=True).raw)).unsqueeze(0).to(device)
        # We then feed our processed image through the neural net to get a vector
        with torch.no_grad():
          image_features = model.encode_image(image)
        # Lastly we convert it to a list so that we can send it through the SDK
        return image_features.tolist()[0]

    # Next - let's encode text based on CLIP
    def encode_text(text):
        # let us get text and then tokenize it
        text = clip.tokenize([text]).to(device)
        # We then feed our processed text through the neural net to get a vector
        with torch.no_grad():
            text_features = model.encode_text(text)
        return text_features.tolist()[0]



.. parsed-literal::

    100%|███████████████████████████████████████| 338M/338M [00:06<00:00, 52.0MiB/s]


We then encode the data we have into vectors, this will take a couple of
mins

.. code:: ipython3

    documents = documents[:500] # only 500 docs to make the process faster

.. code:: ipython3

    def encode_image_document(d):
      try:
        d['product_image_clip_vector_'] = encode_image(d['product_image'])
      except:
        pass

    # Let's import TQDM for a nice progress bar!
    from tqdm.auto import tqdm
    [encode_image_document(d) for d in tqdm(documents)]


3) Insert
---------

Uploading our documents into the dataset ``quickstart_clip``.

In case you are uploading your own dataset, keep in mind that each
document should have a field called '\_id'. Such an id can be easily
allocated using the uuid package:

::

    ds.insert_documents(documents, create_id=True)

.. code:: ipython3

    ds = client.Dataset("quickstart_clip")
    ds.insert_documents(documents)

Once we have uploaded the data, we can see the dataset on the
`dashboard <https://cloud.relevance.ai/dataset/quickstart_clip/dashboard/monitor/vectors>`__.

The dashboard provides users with a great overview and statistics of the
dataset as shown below.

4) Search
---------

This step is to run a simple vector search; you can read more about
vector search and how to construct a multi-vector query
`here <https://docs.relevance.ai/docs/hybrid-search>`__.

Note that our dataset includes vectors generated by the Clip encoder.
Therefore, in this step, we first vectorize the query using the same
encoder to be able to search among the similarly generated vectors.

.. code:: ipython3


    query = 'for my baby daughter'
    query_vector = encode_text(query)
    multivector_query=[
        { "vector": query_vector, "fields": ["product_image_clip_vector_"]}
    ]
    results = ds.vector_search(
        multivector_query=multivector_query,
        page_size=5
    )


You can use our json shower library to observe the search result in a
notebook as shown below:

.. code:: ipython3


    from relevanceai import show_json

    print('=== QUERY === ')
    print(query)

    print('=== RESULTS ===')
    show_json(results, image_fields=["product_image"], text_fields=["product_title"])




.. parsed-literal::

    === QUERY ===>   for my baby daughter




.. raw:: html

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_image</th>
          <th>product_title</th>
          <th>_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9392460/P16581616.jpg" width="60" ></td>
          <td>Crocs Girl (Infant) 'Littles Hover' Leather Athletic Shoe</td>
          <td>cdf48ecc-882a-45ab-b625-ba86bf8cffa4</td>
        </tr>
        <tr>
          <th>1</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9669945/P16850773.jpg" width="60" ></td>
          <td>The New York Doll Collection Double Stroller</td>
          <td>ae2915f9-d7bb-4e0c-8a05-65682cd5a6d3</td>
        </tr>
        <tr>
          <th>2</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/5158127/Badger-Basket-Envee-Baby-High-Chair-Play-Table-in-Pink-P12999228.jpg" width="60" ></td>
          <td>Badger Basket Envee Baby High Chair/ Play Table in Pink</td>
          <td>585e7877-95eb-4864-9d89-03d5369c08fa</td>
        </tr>
        <tr>
          <th>3</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9151116/P16330850.jpg" width="60" ></td>
          <td>Crocs Girl (Toddler) 'CC Magical Day Princess' Synthetic Casual Shoes (Size 6 )</td>
          <td>14c3ad94-3ecd-438b-b00e-1ce5b0eed4e3</td>
        </tr>
        <tr>
          <th>4</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9151116/P16330850.jpg" width="60" ></td>
          <td>Crocs Girl (Toddler) 'CC Magical Day Princess' Synthetic Casual Shoes (Size 6 )</td>
          <td>30809211-dbcd-4b15-8c0a-7702dfe9e30f</td>
        </tr>
      </tbody>
    </table>



Other Notebooks:

-  `Multivector search with your own
   vectors <doc:search-with-your-own-vectors>`__
-  `Text search using USE (VectorHub) <doc:quickstart-text-search>`__
-  `Question answering using USE QA (Tensorflow
   Hub) <doc:quickstart-question-answering>`__

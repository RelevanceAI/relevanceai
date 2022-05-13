.. raw:: html

   <h5>

Developer-first vector platform for ML teams

.. raw:: html

   </h5>

|Open In Colab|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI/blob/main/guides/advanced_search_guide.ipynb

🔍 Advanced Search
==================

Fast Search is Relevance AI’s most complex search endpoint. It combines
functionality to search using vectors, exact text search with ability to
boost your search results depending on your needs. The following
demonstrates a few dummy examples on how to quickly add complexity to
your search!

.. code:: ipython3

    !pip install -q -U RelevanceAI-dev[notebook]


.. parsed-literal::

    [K     |████████████████████████████████| 299 kB 16.0 MB/s 
    [K     |████████████████████████████████| 1.1 MB 61.6 MB/s 
    [K     |████████████████████████████████| 253 kB 53.2 MB/s 
    [K     |████████████████████████████████| 58 kB 6.5 MB/s 
    [K     |████████████████████████████████| 144 kB 49.1 MB/s 
    [K     |████████████████████████████████| 94 kB 3.1 MB/s 
    [K     |████████████████████████████████| 271 kB 48.8 MB/s 
    [K     |████████████████████████████████| 112 kB 47.4 MB/s 
    [?25h  Building wheel for fuzzysearch (setup.py) ... [?25l[?25hdone


.. code:: ipython3

    ## Let's use this CLIP popular model to encode text and image into same space https://github.com/openai/CLIP
    %%capture
    !conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
    !pip install ftfy regex tqdm
    !pip install git+https://github.com/openai/CLIP.git

You can sign up/login and find your credentials here:
https://cloud.relevance.ai/sdk/api Once you have signed up, click on the
value under ``Authorization token`` and paste it here

.. code:: ipython3

    import pandas as pd
    from relevanceai import Client
    client = Client()



.. parsed-literal::

    Activation Token: ··········


🚣 Inserting data
-----------------

We use a sample ecommerce dataset - with vectors
``product_image_clip_vector_`` and ``product_title_clip_vector_``
already encoded for us.

.. code:: ipython3

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded
    docs = get_ecommerce_dataset_encoded()

.. code:: ipython3

    ds = client.Dataset("advanced_search_guide")
    # ds.delete()
    ds.upsert_documents(docs)


.. parsed-literal::

    ✅ All documents inserted/edited successfully.


.. code:: ipython3

    ds.schema




.. parsed-literal::

    {'insert_date_': 'date',
     'price': 'numeric',
     'product_image': 'text',
     'product_image_clip_vector_': {'vector': 512},
     'product_link': 'text',
     'product_price': 'text',
     'product_title': 'text',
     'product_title_clip_vector_': {'vector': 512},
     'query': 'text',
     'source': 'text'}



.. code:: ipython3

    vector_fields = ds.list_vector_fields()
    vector_fields




.. parsed-literal::

    ['product_image_clip_vector_', 'product_title_clip_vector_']



Simple Text Search
------------------

.. code:: ipython3

    results = ds.advanced_search(
        query="nike", fields_to_search=["product_title"], select_fields=["product_title"]
    )
    pd.DataFrame(results["results"])




.. raw:: html

    
      <div id="df-28223bf4-f936-48dd-819e-9ae525fc8622">
        <div class="colab-df-container">
          <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_title</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Nike Women's Summerlite Golf Glove</td>
          <td>b37b2aea-800e-4662-8977-198f744d52bb</td>
          <td>7.590130</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Dura Feel Women's Golf Glove</td>
          <td>e725c79c-c2d2-4c6d-b77a-ed029f33813b</td>
          <td>7.148285</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Junior's Range Jr Golf Shoes</td>
          <td>0e7a5a3d-5d17-42c4-b607-7bf9bb2625a4</td>
          <td>7.148285</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike Sport Lite Women's Golf Bag</td>
          <td>3660e25b-8359-49b9-88c7-fca2dfd9053f</td>
          <td>7.148285</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Women's Tech Xtreme Golf Glove</td>
          <td>8b28e438-0726-4b58-98c7-7597a43d2433</td>
          <td>7.148285</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Nike Women's SQ Dymo Fairway Wood</td>
          <td>adab23fd-ded8-4068-b6a2-999bfe20e5e7</td>
          <td>7.148285</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>b655198b-4356-4ba9-b88e-1e1d6608f43e</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike Junior's Range Red/ White Golf Shoes</td>
          <td>d27e70f3-2884-4490-9742-133166795d0f</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>e1f3faf0-72fa-4559-9604-694699426cc2</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Air Men's Range WP Golf Shoes</td>
          <td>e8d2552f-3ca5-4d15-9ca7-86855025b183</td>
          <td>6.755055</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-28223bf4-f936-48dd-819e-9ae525fc8622')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-28223bf4-f936-48dd-819e-9ae525fc8622 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-28223bf4-f936-48dd-819e-9ae525fc8622');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




Simple Vector Search
--------------------

Let’s prepare some functions to help us encode our data!

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
        image = (
            preprocess(Image.open(requests.get(image, stream=True).raw))
            .unsqueeze(0)
            .to(device)
        )
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

    100%|███████████████████████████████████████| 338M/338M [00:06<00:00, 52.1MiB/s]


.. code:: ipython3

    # Encoding the query
    query_vector = encode_text("nike")
    
    results = ds.advanced_search(
        vector_search_query=[
            {"vector": query_vector, "field": "product_title_clip_vector_"}
        ],
        select_fields=["product_title"],
    )
    
    pd.DataFrame(results["results"])




.. raw:: html

    
      <div id="df-0917fed9-d37c-4e5c-be06-9b0aa4f46786">
        <div class="colab-df-container">
          <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_title</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>PS4 - Playstation 4 Console</td>
          <td>a24c46df-0a1b-49a5-80f4-5ad61bcc6370</td>
          <td>0.748447</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Men's 'Air Visi Pro IV' Synthetic Athleti...</td>
          <td>0435795a-899f-4cdf-89be-a0f3f189d69e</td>
          <td>0.747137</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Men's 'Air Max Pillar' Synthetic Athletic...</td>
          <td>57ca8324-3e8a-4926-9333-b10599edb17b</td>
          <td>0.733907</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Brica Drink Pod</td>
          <td>bbb623f6-485b-44b3-8739-1998b15ae60d</td>
          <td>0.725095</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Gear Head Mouse</td>
          <td>c945fe93-fff3-434b-a91f-18133ab28582</td>
          <td>0.712708</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Gear Head Mouse</td>
          <td>0f1e86a8-867f-4437-8fb0-2b95a37f0c22</td>
          <td>0.712708</td>
        </tr>
        <tr>
          <th>6</th>
          <td>PS4 - UFC</td>
          <td>050a9f63-3549-4720-9be7-9daa07f868e8</td>
          <td>0.702847</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike Women's 'Zoom Hyperquickness' Synthetic A...</td>
          <td>5536a97a-2183-4342-bc92-422aebbcbbc9</td>
          <td>0.697779</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Women's 'Zoom Hyperquickness' Synthetic A...</td>
          <td>00445000-a8ed-4523-b610-f70aa79d47f7</td>
          <td>0.695003</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Men's 'Jordan SC-3' Leather Athletic Shoe</td>
          <td>281d9edd-4be6-4c69-a846-502053f3d4e7</td>
          <td>0.694744</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-0917fed9-d37c-4e5c-be06-9b0aa4f46786')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-0917fed9-d37c-4e5c-be06-9b0aa4f46786 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-0917fed9-d37c-4e5c-be06-9b0aa4f46786');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




Combining Text And Vector Search (Hybrid)
-----------------------------------------

Combining text and vector search allows users get the best of both exact
text search and contextual vector search. This can be done as shown
below.

.. code:: ipython3

    results = ds.advanced_search(
        query="nike",
        fields_to_search=["product_title"],
        vector_search_query=[
            {"vector": query_vector, "field": "product_title_clip_vector_"}
        ],
        select_fields=["product_title"],  # results to return
    )
    
    pd.DataFrame(results["results"])




.. raw:: html

    
      <div id="df-2ee11e7b-1ff0-47f3-808f-81c738ffe817">
        <div class="colab-df-container">
          <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_title</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Nike Women's Summerlite Golf Glove</td>
          <td>b37b2aea-800e-4662-8977-198f744d52bb</td>
          <td>8.140370</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Junior's Range Jr Golf Shoes</td>
          <td>0e7a5a3d-5d17-42c4-b607-7bf9bb2625a4</td>
          <td>7.816567</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Sport Lite Women's Golf Bag</td>
          <td>3660e25b-8359-49b9-88c7-fca2dfd9053f</td>
          <td>7.704053</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike Women's SQ Dymo Fairway Wood</td>
          <td>adab23fd-ded8-4068-b6a2-999bfe20e5e7</td>
          <td>7.700504</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Dura Feel Women's Golf Glove</td>
          <td>e725c79c-c2d2-4c6d-b77a-ed029f33813b</td>
          <td>7.696908</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Nike Women's Tech Xtreme Golf Glove</td>
          <td>8b28e438-0726-4b58-98c7-7597a43d2433</td>
          <td>7.643136</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>8cb26a3e-7de4-4af3-ae40-272450fa9b4d</td>
          <td>7.445704</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>968a9319-fdd4-45ca-adc6-940cd83a204a</td>
          <td>7.440268</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Women's SQ Dymo STR8-FIT Driver</td>
          <td>ff52b64a-0567-4181-8753-763da7044f2f</td>
          <td>7.410513</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>0614f0a9-adcb-4c6c-939c-e7869525549c</td>
          <td>7.408814</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-2ee11e7b-1ff0-47f3-808f-81c738ffe817')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-2ee11e7b-1ff0-47f3-808f-81c738ffe817 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-2ee11e7b-1ff0-47f3-808f-81c738ffe817');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




Adjust the weighting of your vector search results
--------------------------------------------------

Adjust the weighting of your vector search results to make it easier for
you! Simply add a ``weight`` parameter your dictionary inside
``vector_search_query``.

.. code:: ipython3

    results = ds.advanced_search(
        query="nike",
        fields_to_search=["product_title"],
        vector_search_query=[
            {"vector": query_vector, "field": "product_title_clip_vector_", "weight": 0.5}
        ],
        select_fields=["product_title"],  # results to return
    )
    
    pd.DataFrame(results["results"])




.. raw:: html

    
      <div id="df-b8c85355-7961-40db-be70-8d8ab54af2c7">
        <div class="colab-df-container">
          <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_title</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Nike Women's Summerlite Golf Glove</td>
          <td>b37b2aea-800e-4662-8977-198f744d52bb</td>
          <td>7.865250</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Junior's Range Jr Golf Shoes</td>
          <td>0e7a5a3d-5d17-42c4-b607-7bf9bb2625a4</td>
          <td>7.482427</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Sport Lite Women's Golf Bag</td>
          <td>3660e25b-8359-49b9-88c7-fca2dfd9053f</td>
          <td>7.426169</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike Women's SQ Dymo Fairway Wood</td>
          <td>adab23fd-ded8-4068-b6a2-999bfe20e5e7</td>
          <td>7.424395</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Dura Feel Women's Golf Glove</td>
          <td>e725c79c-c2d2-4c6d-b77a-ed029f33813b</td>
          <td>7.422597</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Nike Women's Tech Xtreme Golf Glove</td>
          <td>8b28e438-0726-4b58-98c7-7597a43d2433</td>
          <td>7.395711</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>8cb26a3e-7de4-4af3-ae40-272450fa9b4d</td>
          <td>7.100379</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>968a9319-fdd4-45ca-adc6-940cd83a204a</td>
          <td>7.097662</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Women's SQ Dymo STR8-FIT Driver</td>
          <td>ff52b64a-0567-4181-8753-763da7044f2f</td>
          <td>7.082784</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>0614f0a9-adcb-4c6c-939c-e7869525549c</td>
          <td>7.081935</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-b8c85355-7961-40db-be70-8d8ab54af2c7')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-b8c85355-7961-40db-be70-8d8ab54af2c7 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-b8c85355-7961-40db-be70-8d8ab54af2c7');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




Multi-Vector Search Across Multiple Fields
------------------------------------------

You can easily add more to your search by extending your vector search
query as belows.

.. code:: ipython3

    from PIL import Image
    import requests
    import numpy as np
    
    image_url = "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/e6ea66d1-fd36-4436-bcac-72ed14d8308d/wearallday-younger-shoes-5bnMmp.png"

.. raw:: html

   <h5>

Sample Query Image

.. raw:: html

   </h5>

.. code:: ipython3

    from relevanceai import show_json
    
    image_vector = encode_image(image_url)
    
    results = ds.advanced_search(
        query="nike",
        fields_to_search=["product_title"],
        vector_search_query=[
            {"vector": query_vector, "field": "product_title_clip_vector_", "weight": 0.2},
            {
                "vector": image_vector,
                "field": "product_image_clip_vector_",
                "weight": 0.8,
            },  ## weight the query more on the image vector
        ],
        select_fields=[
            "product_title",
            "product_image",
            "query",
            "product_price",
        ],  # results to return
        queryConfig={"weight": 0.1} # Adjust the weight of the traditional configuration
    )
    
    
    display(
        show_json(
            results["results"],
            text_fields=["product_title", "query", "product_price"],
            image_fields=["product_image"],
        )
    )
    
    # pd.DataFrame(results['results'])



.. raw:: html

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_image</th>
          <th>product_title</th>
          <th>query</th>
          <th>product_price</th>
          <th>_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9684998/P16863510.jpg" width="60" ></td>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>nike womens</td>
          <td>$145.99</td>
          <td>8cb26a3e-7de4-4af3-ae40-272450fa9b4d</td>
        </tr>
        <tr>
          <th>1</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9684998/P16863510.jpg" width="60" ></td>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>nike shoes</td>
          <td>$145.99</td>
          <td>968a9319-fdd4-45ca-adc6-940cd83a204a</td>
        </tr>
        <tr>
          <th>2</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7706311/7706311/Nike-Juniors-Range-Jr-Golf-Shoes-P15113285.jpg" width="60" ></td>
          <td>Nike Junior's Range Jr Golf Shoes</td>
          <td>nike shoes</td>
          <td>$54.99</td>
          <td>0e7a5a3d-5d17-42c4-b607-7bf9bb2625a4</td>
        </tr>
        <tr>
          <th>3</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7957922/7957922/Nike-Ladies-Lunar-Duet-Sport-Golf-Shoes-P15330010.jpg" width="60" ></td>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>nike womens</td>
          <td>$81.99 - $88.07</td>
          <td>80210247-6f40-45be-8279-8743b327f1dc</td>
        </tr>
        <tr>
          <th>4</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9576057/P16765291.jpg" width="60" ></td>
          <td>Nike Mens Lunar Mont Royal Spikeless Golf Shoes</td>
          <td>nike shoes</td>
          <td>$100.99</td>
          <td>e692a73b-a144-4e44-b4db-657be6db96e2</td>
        </tr>
        <tr>
          <th>5</th>
          <td><img src="https://ec1.ostkcdn.com/images/products/9576059/P16765293.jpg" width="60" ></td>
          <td>Nike Mens Lunar Cypress Spikeless Golf Shoes</td>
          <td>nike shoes</td>
          <td>$100.99</td>
          <td>fb323476-a16d-439c-9380-0bac1e10a06d</td>
        </tr>
        <tr>
          <th>6</th>
          <td><img src="https://ec1.ostkcdn.com/images/products/7957922/7957922/Nike-Ladies-Lunar-Duet-Sport-Golf-Shoes-P15330010.jpg" width="60" ></td>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>nike shoes</td>
          <td>$81.99 - $88.07</td>
          <td>b655198b-4356-4ba9-b88e-1e1d6608f43e</td>
        </tr>
        <tr>
          <th>7</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/8952218/Nike-Womens-Lunaracer-3-Mesh-Athletic-Shoe-P16163941.jpg" width="60" ></td>
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>nike shoes</td>
          <td>$107.99</td>
          <td>0614f0a9-adcb-4c6c-939c-e7869525549c</td>
        </tr>
        <tr>
          <th>8</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/8952218/Nike-Womens-Lunaracer-3-Mesh-Athletic-Shoe-P16163941.jpg" width="60" ></td>
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>nike womens</td>
          <td>$107.99</td>
          <td>7baea34f-fb0a-47da-9edd-d920abddccf5</td>
        </tr>
        <tr>
          <th>9</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7481848/7481848/Nike-Air-Mens-Range-WP-Golf-Shoes-P14927541.jpg" width="60" ></td>
          <td>Nike Air Men's Range WP Golf Shoes</td>
          <td>nike shoes</td>
          <td>$90.99 - $91.04</td>
          <td>e8d2552f-3ca5-4d15-9ca7-86855025b183</td>
        </tr>
      </tbody>
    </table>


Chunk Search Guide
------------------

Chunk search allows users to search within a a *chunk* field. Chunk
search allows users to search more fine-grained. A sample chunk search
query is shown below.

.. code:: ipython3

    from relevanceai import mock_documents
    documents = mock_documents()
    
    ds = client.Dataset("mock_dataset")
    ds.upsert_documents(documents)


.. parsed-literal::

    ✅ All documents inserted/edited successfully.


.. code:: ipython3

    ds.schema




.. parsed-literal::

    {'_chunk_': 'chunks',
     '_chunk_.label': 'text',
     '_chunk_.label_chunkvector_': {'chunkvector': 5},
     'insert_date_': 'date',
     'sample_1_description': 'text',
     'sample_1_label': 'text',
     'sample_1_value': 'numeric',
     'sample_1_vector_': {'vector': 5},
     'sample_2_description': 'text',
     'sample_2_label': 'text',
     'sample_2_value': 'numeric',
     'sample_2_vector_': {'vector': 5},
     'sample_3_description': 'text',
     'sample_3_label': 'text',
     'sample_3_value': 'numeric',
     'sample_3_vector_': {'vector': 5}}



.. code:: ipython3

    # Provide a chunk search 
    ds.advanced_search(
        vector_search_query=[
            {
             "vector": [1, 1, 1, 1, 1],
             "field": "label_chunkvector_", 
             "weight": 1,
             "chunkConfig": {
                 "chunkField": "_chunk_",
                 "page": 0,
                 # the number of chunk results to return 
                 # - stored in `_chunk_results` key
                 "pageSize": 3 
             }
            },
        ],
    
    )




.. parsed-literal::

    {'afterId': [],
     'aggregateStats': {},
     'aggregates': {},
     'aggregations': {},
     'results': [{'_chunk_': [{'label': 'label_1',
         'label_chunkvector_': [0.9714655321220234,
          0.7128316097400133,
          0.6781037943929558,
          0.6488623491829022,
          0.775330428892935]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_1'}]}},
       '_id': '0fba3159-44ed-3303-ae3e-8763af736d82',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': 'WRZGB',
       'sample_1_label': 'label_1',
       'sample_1_value': 95,
       'sample_1_vector_': [0.010111141119929168,
        0.8100269908459344,
        0.8450143601010813,
        0.5200637988452348,
        0.6807143398905711],
       'sample_2_description': '27MA4',
       'sample_2_label': 'label_2',
       'sample_2_value': 62,
       'sample_2_vector_': [0.8158557111159398,
        0.7079708018800909,
        0.040442267483184136,
        0.2550053832057586,
        0.6655286701296413],
       'sample_3_description': '1NJGR',
       'sample_3_label': 'label_0',
       'sample_3_value': 16,
       'sample_3_vector_': [0.8319698111146892,
        0.2970554960820262,
        0.7053962091476822,
        0.7616721137875679,
        0.33539644279489944]},
      {'_chunk_': [{'label': 'label_2',
         'label_chunkvector_': [0.17573371062486798,
          0.557943855238517,
          0.697754222989297,
          0.9786125118059382,
          0.7922094154419312]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_2'}]}},
       '_id': '51a2eb0f-94c6-3035-89b0-027b2379b3d7',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': 'MH2FZ',
       'sample_1_label': 'label_2',
       'sample_1_value': 4,
       'sample_1_vector_': [0.13122902838701556,
        0.3479630189944891,
        0.7020069274564608,
        0.28257296541486776,
        0.15930197109337352],
       'sample_2_description': 'KU20B',
       'sample_2_label': 'label_0',
       'sample_2_value': 58,
       'sample_2_vector_': [0.20753358564393043,
        0.7285124067578301,
        0.9003748477567735,
        0.912483293611922,
        0.23245362499843847],
       'sample_3_description': '62ZTY',
       'sample_3_label': 'label_5',
       'sample_3_value': 98,
       'sample_3_vector_': [0.6187077877648824,
        0.4248041940356846,
        0.48710139974254263,
        0.769860649556282,
        0.5785388950443682]},
      {'_chunk_': [{'label': 'label_4',
         'label_chunkvector_': [0.7338702708046809,
          0.41755372242176314,
          0.4912010324442426,
          0.0834347624193984,
          0.48279406238186817]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_4'}]}},
       '_id': 'b56dd78b-0c29-3a00-8c6d-387655ca0a2b',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': 'VPYRN',
       'sample_1_label': 'label_3',
       'sample_1_value': 13,
       'sample_1_vector_': [0.3828314864256408,
        0.36459459507507885,
        0.8940227989713352,
        0.8794642161978363,
        0.9682486851016051],
       'sample_2_description': 'ZYE1X',
       'sample_2_label': 'label_5',
       'sample_2_value': 66,
       'sample_2_vector_': [0.12136689372267317,
        0.462037834296147,
        0.5120688870564564,
        0.38689918710131,
        0.2805130330014971],
       'sample_3_description': 'TCC27',
       'sample_3_label': 'label_3',
       'sample_3_value': 19,
       'sample_3_vector_': [0.09914254554709134,
        0.920167083569516,
        0.11868940231964686,
        0.5438045792718624,
        0.43635676728310124]},
      {'_chunk_': [{'label': 'label_5',
         'label_chunkvector_': [0.33436906373438624,
          0.5380728845974861,
          0.23972813094355927,
          0.7919330405084691,
          0.2878108785508634]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_5'}]}},
       '_id': '086151be-e3e0-3c74-ace7-6292246f0fc9',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': '2COZY',
       'sample_1_label': 'label_2',
       'sample_1_value': 96,
       'sample_1_vector_': [0.7147183445018557,
        0.18066520347080173,
        0.9740064235203669,
        0.6258224799724947,
        0.3500929889622264],
       'sample_2_description': 'H638P',
       'sample_2_label': 'label_4',
       'sample_2_value': 16,
       'sample_2_vector_': [0.9450798492356538,
        0.4462449289257341,
        0.004355001860774199,
        0.25486874541800486,
        0.3482060493985143],
       'sample_3_description': 'F2T24',
       'sample_3_label': 'label_2',
       'sample_3_value': 15,
       'sample_3_vector_': [0.14630374114623268,
        0.12238406234925325,
        0.5542096939075382,
        0.0475748252915158,
        0.41292937921919615]},
      {'_chunk_': [{'label': 'label_2',
         'label_chunkvector_': [0.9465667341769131,
          0.8306490761371044,
          0.06366580368540398,
          0.4169022757966413,
          0.2879497402145924]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_2'}]}},
       '_id': '32abd915-60b5-373c-825b-22ba2b7e01bf',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': '5GFR7',
       'sample_1_label': 'label_5',
       'sample_1_value': 0,
       'sample_1_vector_': [0.03266482919634517,
        0.2184525410362036,
        0.4272720912279113,
        0.735584738472561,
        0.16534557670923755],
       'sample_2_description': '4F95A',
       'sample_2_label': 'label_2',
       'sample_2_value': 50,
       'sample_2_vector_': [0.5666292182319911,
        0.045574402067497854,
        0.20808912259919377,
        0.41197652736153034,
        0.9622611439423331],
       'sample_3_description': '50JO8',
       'sample_3_label': 'label_4',
       'sample_3_value': 91,
       'sample_3_vector_': [0.8349167635041148,
        0.9909929540761643,
        0.36585325598630203,
        0.635433668522285,
        0.28632200528034224]},
      {'_chunk_': [{'label': 'label_0',
         'label_chunkvector_': [0.04643479539353512,
          0.832710978356411,
          0.27875623750147294,
          0.4913456773422803,
          0.5388430545812762]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_0'}]}},
       '_id': 'c4ba6213-c9de-31e7-8102-b6078cecfeaf',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': '2GTHL',
       'sample_1_label': 'label_1',
       'sample_1_value': 80,
       'sample_1_vector_': [0.4934065280893306,
        0.599044030362021,
        0.23000529514903578,
        0.35262850141097246,
        0.447190046367118],
       'sample_2_description': 'Y7DEF',
       'sample_2_label': 'label_0',
       'sample_2_value': 60,
       'sample_2_vector_': [0.41032731851307735,
        0.11788099018533249,
        0.6375475627332368,
        0.27037361979827434,
        0.11434413934349097],
       'sample_3_description': 'RN2TG',
       'sample_3_label': 'label_5',
       'sample_3_value': 40,
       'sample_3_vector_': [0.28102035620163046,
        0.7421090875142067,
        0.09771653703658345,
        0.10015420429876987,
        0.13744357712958866]},
      {'_chunk_': [{'label': 'label_0',
         'label_chunkvector_': [0.44900339130825384,
          0.8856780512547253,
          0.5731744454632794,
          0.07634302009769145,
          0.126567766301261]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_0'}]}},
       '_id': '9dc452ec-e60c-35e5-ad50-8abc544d72f5',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': '6WOTX',
       'sample_1_label': 'label_5',
       'sample_1_value': 90,
       'sample_1_vector_': [0.36754556968019914,
        0.7570935190789245,
        0.07080217925165144,
        0.0377899628386521,
        0.010935468014863448],
       'sample_2_description': 'C9AQY',
       'sample_2_label': 'label_0',
       'sample_2_value': 66,
       'sample_2_vector_': [0.8841987637795244,
        0.5798869557821004,
        0.629484594620124,
        0.15513971487981038,
        0.06784721110008496],
       'sample_3_description': 'CB9MB',
       'sample_3_label': 'label_1',
       'sample_3_value': 19,
       'sample_3_vector_': [0.07807951748335318,
        0.7070506382865839,
        0.7331808226921382,
        0.13633307017391627,
        0.22967712634144954]},
      {'_chunk_': [{'label': 'label_1',
         'label_chunkvector_': [0.298123417344888,
          0.6109539928925158,
          0.594743730194975,
          0.2648613560137232,
          0.8339071789779628]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_1'}]}},
       '_id': 'cecb1a97-3540-3b0f-a4b2-b7bed4df0e10',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': 'QJHMH',
       'sample_1_label': 'label_3',
       'sample_1_value': 42,
       'sample_1_vector_': [0.664231043403755,
        0.47220553157818856,
        0.08584357353004624,
        0.008458751015532395,
        0.3591367465817318],
       'sample_2_description': '18HJ7',
       'sample_2_label': 'label_4',
       'sample_2_value': 31,
       'sample_2_vector_': [0.46163001848406293,
        0.530708764060759,
        0.9892401074533322,
        0.2565786433160304,
        0.36644315611129674],
       'sample_3_description': 'Q8HBC',
       'sample_3_label': 'label_0',
       'sample_3_value': 46,
       'sample_3_vector_': [0.15636989000338164,
        0.30213016734011644,
        0.5854349758809958,
        0.6564881895528701,
        0.7604572527984234]},
      {'_chunk_': [{'label': 'label_3',
         'label_chunkvector_': [0.2459123596946453,
          0.9324565094950896,
          0.27724503128111255,
          0.0943163583176111,
          0.9062322733100795]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_3'}]}},
       '_id': '83644e36-9aea-36bf-8921-6763245fe23a',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': 'Q8ZGC',
       'sample_1_label': 'label_0',
       'sample_1_value': 25,
       'sample_1_vector_': [0.36705678637922134,
        0.5030829146042314,
        0.27586504612917107,
        0.04638466153973042,
        0.6038331836372212],
       'sample_2_description': 'BKQOW',
       'sample_2_label': 'label_3',
       'sample_2_value': 69,
       'sample_2_vector_': [0.06599734377357402,
        0.7291538497710904,
        0.5723644440353702,
        0.6404097412423622,
        0.14369410325126808],
       'sample_3_description': '0D1UW',
       'sample_3_label': 'label_3',
       'sample_3_value': 6,
       'sample_3_vector_': [0.421115379610321,
        0.3275935294784218,
        0.058777940280584584,
        0.04186263256123568,
        0.6260049143683458]},
      {'_chunk_': [{'label': 'label_4',
         'label_chunkvector_': [0.08549435178564124,
          0.11520069704151803,
          0.43403327749130916,
          0.01974440345523576,
          0.14372394345151063]}],
       '_chunk_results': {'_chunk_': {'_relevance': 0,
         'results': [{'_relevance': 0, 'label': 'label_4'}]}},
       '_id': '3495a820-467a-30d8-895b-50befba38f99',
       '_relevance': 0,
       'insert_date_': '2022-05-13T01:21:24.679Z',
       'sample_1_description': 'MOEEO',
       'sample_1_label': 'label_0',
       'sample_1_value': 67,
       'sample_1_vector_': [0.8513889401103251,
        0.7485584349006119,
        0.7453551223300326,
        0.6314495537016419,
        0.25585253601766167],
       'sample_2_description': '78D55',
       'sample_2_label': 'label_1',
       'sample_2_value': 3,
       'sample_2_vector_': [0.29248363029303037,
        0.3989529263903293,
        0.2237003035286077,
        0.3232937426927007,
        0.535646801886282],
       'sample_3_description': '53VLV',
       'sample_3_label': 'label_1',
       'sample_3_value': 94,
       'sample_3_vector_': [0.24476704213234246,
        0.582106042132727,
        0.8711476351278145,
        0.540170037829761,
        0.6652872417327402]}],
     'resultsSize': 100}



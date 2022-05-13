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

    image_vector[0:5]




.. parsed-literal::

    [-0.1314697265625,
     -0.442626953125,
     0.0194549560546875,
     0.11602783203125,
     -0.405029296875]



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


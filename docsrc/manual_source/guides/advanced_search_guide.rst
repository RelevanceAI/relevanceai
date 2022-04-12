🔍 Advanced Search
=================

Fast Search is Relevance AI's most complex search endpoint. It combines
functionality to search using vectors, exact text search with ability to
boost your search results depending on your needs. The following
demonstrates a few dummy examples on how to quickly add complexity to
your search!

.. code:: ipython3

    !pip install -q RelevanceAI[notebook]

.. code:: ipython3

    ## Let's use this CLIP popular model to encode text and image into same space https://github.com/openai/CLIP

    !conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
    !pip install ftfy regex tqdm
    !pip install git+https://github.com/openai/CLIP.git


.. parsed-literal::

    /bin/bash: conda: command not found
    Requirement already satisfied: ftfy in /usr/local/lib/python3.7/dist-packages (6.1.1)
    Requirement already satisfied: regex in /usr/local/lib/python3.7/dist-packages (2019.12.20)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (4.63.0)
    Requirement already satisfied: wcwidth>=0.2.5 in /usr/local/lib/python3.7/dist-packages (from ftfy) (0.2.5)
    Collecting git+https://github.com/openai/CLIP.git
      Cloning https://github.com/openai/CLIP.git to /tmp/pip-req-build-8opfrum1
      Running command git clone -q https://github.com/openai/CLIP.git /tmp/pip-req-build-8opfrum1
    Requirement already satisfied: ftfy in /usr/local/lib/python3.7/dist-packages (from clip==1.0) (6.1.1)
    Requirement already satisfied: regex in /usr/local/lib/python3.7/dist-packages (from clip==1.0) (2019.12.20)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from clip==1.0) (4.63.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (from clip==1.0) (1.10.0+cu111)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.7/dist-packages (from clip==1.0) (0.11.1+cu111)
    Requirement already satisfied: wcwidth>=0.2.5 in /usr/local/lib/python3.7/dist-packages (from ftfy->clip==1.0) (0.2.5)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch->clip==1.0) (3.10.0.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torchvision->clip==1.0) (1.21.5)
    Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in /usr/local/lib/python3.7/dist-packages (from torchvision->clip==1.0) (7.1.2)


You can sign up/login and find your credentials here:
https://cloud.relevance.ai/sdk/api Once you have signed up, click on the
value under ``Authorization token`` and paste it here

.. code:: ipython3


    import pandas as pd
    from relevanceai import Client

    client = Client()



.. parsed-literal::

    Activation token (you can find it here: https://cloud.relevance.ai/sdk/api )

    Activation Token: ··········
    Connecting to ap-southeast-2...
    You can view all your datasets at https://cloud.relevance.ai/datasets/
    Welcome to RelevanceAI. Logged in as 22d8b47fbed6cf1391f0.


🚣 Inserting data
----------------

We use a sample ecommerce dataset - with vectors
``product_image_clip_vector_`` and ``product_title_clip_vector_``
already encoded for us.

.. code:: ipython3


    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    docs = get_ecommerce_dataset_encoded()
    docs[0].keys()





.. parsed-literal::

    dict_keys(['product_image', 'query', 'product_price', 'source', 'product_title', 'product_link', 'product_image_clip_vector_', 'product_title_clip_vector_', 'insert_date_', 'price', '_id'])



.. code:: ipython3

    ds = client.Dataset('advanced_search_guide')
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

    results = ds.advanced_search(query="nike",
                                fields_to_search=["product_title"],
                                 select_fields=['product_title']
                                 )
    pd.DataFrame(results['results'])




.. raw:: html


      <div id="df-4d475977-4f2a-4eb8-8631-4170ca25706b">
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
          <td>Nike Mens Lunar Cypress Spikeless Golf Shoes</td>
          <td>fb323476-a16d-439c-9380-0bac1e10a06d</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Women's SQ Dymo STR8-FIT Driver</td>
          <td>ff52b64a-0567-4181-8753-763da7044f2f</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>0614f0a9-adcb-4c6c-939c-e7869525549c</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike SolarSoft Golf Grill Room Black Shoes</td>
          <td>22871acd-fbc9-462e-8305-26df642c915c</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>6f85d037-7621-45ee-b5dc-dd0e88c58d4a</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>7baea34f-fb0a-47da-9edd-d920abddccf5</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>80210247-6f40-45be-8279-8743b327f1dc</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>8cb26a3e-7de4-4af3-ae40-272450fa9b4d</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>968a9319-fdd4-45ca-adc6-940cd83a204a</td>
          <td>6.755055</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Ladies Pink Lunar Duet Sport Golf Shoes</td>
          <td>c523a39a-82b1-4311-bf25-c572cb164a4b</td>
          <td>6.402832</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-4d475977-4f2a-4eb8-8631-4170ca25706b')"
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
              document.querySelector('#df-4d475977-4f2a-4eb8-8631-4170ca25706b button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';

            async function convertToInteractive(key) {
              const element = document.querySelector('#df-4d475977-4f2a-4eb8-8631-4170ca25706b');
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

Let's prepare some functions to help us encode our data!

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

    100%|███████████████████████████████████████| 338M/338M [00:15<00:00, 22.7MiB/s]


.. code:: ipython3


    # Encoding the query
    query_vector = encode_text('nike')

    results = ds.advanced_search(
        vector_search_query=[
        {
          "vector": query_vector,
          "field":'product_title_clip_vector_'
         }
    ],
    select_fields=['product_title'])

    pd.DataFrame(results['results'])





.. raw:: html


      <div id="df-87a1bb6d-0650-4bf0-8107-9072216c0afb">
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
          <td>Nike Women's 'Son Of Force Low' Leather Athlet...</td>
          <td>f0776d1d-58c2-40e1-a6a8-1389ab7c9097</td>
          <td>0.693284</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Classic Tote Bag</td>
          <td>89f74212-e9fd-46da-90f0-157d54a93693</td>
          <td>0.691830</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>8cb26a3e-7de4-4af3-ae40-272450fa9b4d</td>
          <td>0.690562</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike Men's 'Air Max '93' Leather Athletic Shoe</td>
          <td>d97d11df-0c37-4e33-8ac6-315e73884be0</td>
          <td>0.690420</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Men's 'Lunarglide 6' Synthetic Athletic Shoe</td>
          <td>968a9319-fdd4-45ca-adc6-940cd83a204a</td>
          <td>0.685140</td>
        </tr>
        <tr>
          <th>5</th>
          <td>PS4 - Destiny</td>
          <td>a5a6ee33-17da-4da8-b675-d18d4a43a6e4</td>
          <td>0.683079</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Panasonic Earbud Headphones</td>
          <td>83d1f654-2a47-44e7-994d-dc1c48c9abc6</td>
          <td>0.679970</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Panasonic Earbud Headphones</td>
          <td>ecd884ed-6acf-4bff-9dd4-d2ca1f82c4d6</td>
          <td>0.679800</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Panasonic Earbud Headphones</td>
          <td>d51b8c05-b5b2-4667-b482-68f16a8fc7c6</td>
          <td>0.679770</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Panasonic Earbud Headphones</td>
          <td>e694014a-f336-45d1-95a9-54ab55f676fc</td>
          <td>0.679770</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-87a1bb6d-0650-4bf0-8107-9072216c0afb')"
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
              document.querySelector('#df-87a1bb6d-0650-4bf0-8107-9072216c0afb button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';

            async function convertToInteractive(key) {
              const element = document.querySelector('#df-87a1bb6d-0650-4bf0-8107-9072216c0afb');
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
            {
            "vector": query_vector,
             "field":'product_title_clip_vector_'}
        ],
        select_fields = ["product_title"], # results to return
    )

    pd.DataFrame(results['results'])





.. raw:: html


      <div id="df-6673201c-bd24-4afe-b8f3-7df1e3ed9a36">
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
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>7baea34f-fb0a-47da-9edd-d920abddccf5</td>
          <td>7.408601</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Air Men's Range WP Golf Shoes</td>
          <td>e8d2552f-3ca5-4d15-9ca7-86855025b183</td>
          <td>7.405841</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>b655198b-4356-4ba9-b88e-1e1d6608f43e</td>
          <td>7.358681</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>80210247-6f40-45be-8279-8743b327f1dc</td>
          <td>7.358681</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Mens Lunar Cypress Spikeless Golf Shoes</td>
          <td>fb323476-a16d-439c-9380-0bac1e10a06d</td>
          <td>7.329358</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>e1f3faf0-72fa-4559-9604-694699426cc2</td>
          <td>7.314920</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>6f85d037-7621-45ee-b5dc-dd0e88c58d4a</td>
          <td>7.314820</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike SolarSoft Golf Grill Room Black Shoes</td>
          <td>22871acd-fbc9-462e-8305-26df642c915c</td>
          <td>7.280349</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Junior's Range Red/ White Golf Shoes</td>
          <td>d27e70f3-2884-4490-9742-133166795d0f</td>
          <td>7.264541</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Men's 'Air Max Pillar' Synthetic Athletic...</td>
          <td>57ca8324-3e8a-4926-9333-b10599edb17b</td>
          <td>7.136653</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-6673201c-bd24-4afe-b8f3-7df1e3ed9a36')"
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
              document.querySelector('#df-6673201c-bd24-4afe-b8f3-7df1e3ed9a36 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';

            async function convertToInteractive(key) {
              const element = document.querySelector('#df-6673201c-bd24-4afe-b8f3-7df1e3ed9a36');
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
            {
              "vector": query_vector,
             "field":'product_title_clip_vector_',
             "weight": 0.5
             }
        ],
        select_fields=["product_title"], # results to return
    )

    pd.DataFrame(results['results'])




.. raw:: html


      <div id="df-de6e771d-46c2-4222-b482-7912da0a64ca">
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
          <td>Nike Women's 'Lunaracer+ 3' Mesh Athletic Shoe</td>
          <td>7baea34f-fb0a-47da-9edd-d920abddccf5</td>
          <td>7.081828</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Air Men's Range WP Golf Shoes</td>
          <td>e8d2552f-3ca5-4d15-9ca7-86855025b183</td>
          <td>7.080448</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>b655198b-4356-4ba9-b88e-1e1d6608f43e</td>
          <td>7.056868</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>80210247-6f40-45be-8279-8743b327f1dc</td>
          <td>7.056868</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nike Mens Lunar Cypress Spikeless Golf Shoes</td>
          <td>fb323476-a16d-439c-9380-0bac1e10a06d</td>
          <td>7.042206</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>e1f3faf0-72fa-4559-9604-694699426cc2</td>
          <td>7.034987</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>6f85d037-7621-45ee-b5dc-dd0e88c58d4a</td>
          <td>7.034937</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Nike SolarSoft Golf Grill Room Black Shoes</td>
          <td>22871acd-fbc9-462e-8305-26df642c915c</td>
          <td>7.017702</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Nike Junior's Range Red/ White Golf Shoes</td>
          <td>d27e70f3-2884-4490-9742-133166795d0f</td>
          <td>7.009798</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Nike Men's 'Air Max Pillar' Synthetic Athletic...</td>
          <td>57ca8324-3e8a-4926-9333-b10599edb17b</td>
          <td>6.769743</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-de6e771d-46c2-4222-b482-7912da0a64ca')"
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
              document.querySelector('#df-de6e771d-46c2-4222-b482-7912da0a64ca button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';

            async function convertToInteractive(key) {
              const element = document.querySelector('#df-de6e771d-46c2-4222-b482-7912da0a64ca');
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


    image_url = 'https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/e6ea66d1-fd36-4436-bcac-72ed14d8308d/wearallday-younger-shoes-5bnMmp.png'

    image_vector = encode_image(image_url)

    results = ds.advanced_search(
        query="nike",
        fields_to_search=["product_title"],
        vector_search_query=[
            {"vector": query_vector, "field": 'product_title_clip_vector_'},
            {"vector": image_vector, "field": 'product_image_clip_vector_'}
        ],
        select_fields=["product_title", "product_image", "product_price", "query"], # results to return
    )

    from relevanceai import show_json


    display(
        show_json(
            results['results'],
            image_fields=['product_image'],
            text_fields=['query', 'product_title', 'product_price'],
        )
    )

    # pd.DataFrame(results['results'])




.. raw:: html

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_image</th>
          <th>query</th>
          <th>product_title</th>
          <th>product_price</th>
          <th>_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7481848/7481848/Nike-Air-Mens-Range-WP-Golf-Shoes-P14927541.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike Air Men's Range WP Golf Shoes</td>
          <td>$90.99 - $91.04</td>
          <td>e8d2552f-3ca5-4d15-9ca7-86855025b183</td>
        </tr>
        <tr>
          <th>1</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7957922/7957922/Nike-Ladies-Lunar-Duet-Sport-Golf-Shoes-P15330010.jpg" width="60" ></td>
          <td>nike womens</td>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>$81.99 - $88.07</td>
          <td>80210247-6f40-45be-8279-8743b327f1dc</td>
        </tr>
        <tr>
          <th>2</th>
          <td><img src="https://ec1.ostkcdn.com/images/products/7957922/7957922/Nike-Ladies-Lunar-Duet-Sport-Golf-Shoes-P15330010.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike Ladies Lunar Duet Sport Golf Shoes</td>
          <td>$81.99 - $88.07</td>
          <td>b655198b-4356-4ba9-b88e-1e1d6608f43e</td>
        </tr>
        <tr>
          <th>3</th>
          <td><img src="https://ec1.ostkcdn.com/images/products/9576059/P16765293.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike Mens Lunar Cypress Spikeless Golf Shoes</td>
          <td>$100.99</td>
          <td>fb323476-a16d-439c-9380-0bac1e10a06d</td>
        </tr>
        <tr>
          <th>4</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9572101/P16760787.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike SolarSoft Golf Grill Room Black Shoes</td>
          <td>$49.99</td>
          <td>22871acd-fbc9-462e-8305-26df642c915c</td>
        </tr>
        <tr>
          <th>5</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7706421/7706421/Nike-Juniors-Range-Red-White-Golf-Shoes-P15113324.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike Junior's Range Red/ White Golf Shoes</td>
          <td>$49.99</td>
          <td>d27e70f3-2884-4490-9742-133166795d0f</td>
        </tr>
        <tr>
          <th>6</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7709063/7709063/Nike-Womens-Lunar-Duet-Classic-Golf-Shoes-P15115286.jpg" width="60" ></td>
          <td>nike womens</td>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>$97.99</td>
          <td>6f85d037-7621-45ee-b5dc-dd0e88c58d4a</td>
        </tr>
        <tr>
          <th>7</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/7709063/7709063/Nike-Womens-Lunar-Duet-Classic-Golf-Shoes-P15115286.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike Women's Lunar Duet Classic Golf Shoes</td>
          <td>$97.99</td>
          <td>e1f3faf0-72fa-4559-9604-694699426cc2</td>
        </tr>
        <tr>
          <th>8</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/5136983/56/360/Nike-Womens-SQ-Dymo-STR8-FIT-Driver-P12982562.jpg" width="60" ></td>
          <td>nike womens</td>
          <td>Nike Women's SQ Dymo STR8-FIT Driver</td>
          <td>$146.99</td>
          <td>ff52b64a-0567-4181-8753-763da7044f2f</td>
        </tr>
        <tr>
          <th>9</th>
          <td><img src="https://ak1.ostkcdn.com/images/products/9576057/P16765291.jpg" width="60" ></td>
          <td>nike shoes</td>
          <td>Nike Mens Lunar Mont Royal Spikeless Golf Shoes</td>
          <td>$100.99</td>
          <td>e692a73b-a144-4e44-b4db-657be6db96e2</td>
        </tr>
      </tbody>
    </table>

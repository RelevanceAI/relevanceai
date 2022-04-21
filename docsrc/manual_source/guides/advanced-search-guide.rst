🔍 Advanced Search
==================

Fast Search is Relevance AI’s most complex search endpoint. It combines
functionality to search using vectors, exact text search with ability to
boost your search results depending on your needs. The following
demonstrates a few dummy examples on how to quickly add complexity to
your search!

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    import pandas as pd
    from relevanceai import Client

.. code:: ipython3

    client = Client()

.. code:: ipython3

    ds = client.Dataset("clothes")

Simple Text Search
==================

.. code:: ipython3

    results = ds.advanced_search(
        query="nike", fields_to_search=["prod_name"], select_fields=["prod_name"]
    )
    pd.DataFrame(results["results"])




.. raw:: html

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
          <th>prod_name</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NIKE</td>
          <td>44280a84-0e8d-4787-8aa6-d2d9506da894</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NIKE</td>
          <td>5e6a78eb-0e1a-4f84-99d1-7362ea4283f9</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NIKE</td>
          <td>5a7e502c-618f-42af-9f2b-6658587f8cc2</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NIKE</td>
          <td>bdc557f7-1d22-45bc-874d-f09b4fd76928</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NIKE</td>
          <td>bb9d74fb-e70f-4cb2-ab35-4925dae8ecac</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NIKE</td>
          <td>901b8471-f7af-4dba-a3f3-ecc494f5093a</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NIKE</td>
          <td>09b3b546-2b7d-4b44-9b5f-6d7350af2bff</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NIKE</td>
          <td>280cb2fb-fc40-4052-acc1-556c08493d24</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NIKE</td>
          <td>3c12c230-b2a0-4706-9f4d-7f929ffba714</td>
          <td>22.303717</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NIKE</td>
          <td>3b875970-43f3-4342-9372-a9163431c839</td>
          <td>22.303717</td>
        </tr>
      </tbody>
    </table>
    </div>



Simple Vector Search
====================

.. code:: ipython3

    # Create a simple mock vector for now
    vector = [1e-7] * 512
    results = ds.advanced_search(
        vector_search_query=[{"vector": vector, "field": "prod_name_use_vector_"}],
        select_fields=["prod_name"],
    )
    pd.DataFrame(results["results"])




.. raw:: html

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
          <th>prod_name</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>PIMA shell trousers</td>
          <td>73d95583-29cb-4dae-9066-14cf645195e7</td>
          <td>0.130984</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Rawley Chinos Slim</td>
          <td>4e879e37-af82-4a13-80f4-c22b7e9474dc</td>
          <td>0.128895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Rawley Chinos Slim</td>
          <td>46615788-a7af-42ab-9230-ec3087f45217</td>
          <td>0.128895</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Rawley Chinos Slim</td>
          <td>8f9ee4ba-81f1-4e95-ba02-3049694308ed</td>
          <td>0.128895</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Rawley Chinos Slim</td>
          <td>d4784d29-7aec-46da-9ae7-39431a47a9a1</td>
          <td>0.128895</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Rawley Chinos Slim</td>
          <td>d5617367-95e9-40c6-889a-b19b74bb8589</td>
          <td>0.128895</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Rawley Chinos Slim</td>
          <td>2362508c-f099-44d1-b14d-0c1490e8eb82</td>
          <td>0.128895</td>
        </tr>
        <tr>
          <th>7</th>
          <td>EDC Eli Kaftan</td>
          <td>c21bd3c2-9491-411f-8031-f071da8e0a50</td>
          <td>0.128436</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Ringhild earring pack</td>
          <td>c2ec8d6e-6fbd-4601-9a65-9145d784c614</td>
          <td>0.128367</td>
        </tr>
        <tr>
          <th>9</th>
          <td>2PACK SS Body TVP</td>
          <td>c6353a97-d8a0-4a5f-8a2d-5deb479a5b25</td>
          <td>0.126159</td>
        </tr>
      </tbody>
    </table>
    </div>



Combining Text And Vector Search (Hybrid)
=========================================

Combining text and vector search allows users get the best of both exact
text search and contextual vector search. This can be done as shown
below.

.. code:: ipython3

    results = ds.advanced_search(
        query="nike",
        fields_to_search=["prod_name"],
        vector_search_query=[{"vector": vector, "field": "prod_name_use_vector_"}],
        select_fields=["prod_name"],  # results to return
    )
    pd.DataFrame(results["results"])




.. raw:: html

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
          <th>prod_name</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NIKE</td>
          <td>3d13058f-fa09-4f00-bfb0-fecb2671d206</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NIKE</td>
          <td>011668c3-5546-458a-a57b-7e270c1dc987</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NIKE</td>
          <td>b203ebbb-f75b-45c8-8a45-1d9322f2750d</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NIKE</td>
          <td>8bba89a1-b1dd-4a2f-b4e8-68437d7b3c82</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NIKE</td>
          <td>890e1643-294e-4fdd-8787-1b0b325c6069</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NIKE</td>
          <td>c6def1ca-515d-43c4-8d05-d3de7ebea9b3</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NIKE</td>
          <td>a4480651-b9b4-4c02-8a59-c53a9a8f7d13</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NIKE</td>
          <td>81c74d7b-0f50-468b-b14e-ba36e9818ca4</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NIKE</td>
          <td>7c8f53cf-0c26-416c-891b-095761fb5d38</td>
          <td>22.365116</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NIKE</td>
          <td>e9a98454-ced3-4f79-96fd-894684465603</td>
          <td>22.365116</td>
        </tr>
      </tbody>
    </table>
    </div>



Adjust the weighting of your vector search results
==================================================

Adjust the weighting of your vector search results to make it easier for
you! Simply add a ``weight`` parameter your dictionary inside
``vector_search_query``.

.. code:: ipython3

    results = ds.advanced_search(
        query="nike",
        fields_to_search=["prod_name"],
        vector_search_query=[
            {"vector": vector, "field": "prod_name_use_vector_", "weight": 0.5}
        ],
        select_fields=["prod_name"],  # results to return
    )
    pd.DataFrame(results["results"])




.. raw:: html

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
          <th>prod_name</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NIKE</td>
          <td>3d13058f-fa09-4f00-bfb0-fecb2671d206</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NIKE</td>
          <td>011668c3-5546-458a-a57b-7e270c1dc987</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NIKE</td>
          <td>b203ebbb-f75b-45c8-8a45-1d9322f2750d</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NIKE</td>
          <td>8bba89a1-b1dd-4a2f-b4e8-68437d7b3c82</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NIKE</td>
          <td>890e1643-294e-4fdd-8787-1b0b325c6069</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NIKE</td>
          <td>c6def1ca-515d-43c4-8d05-d3de7ebea9b3</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NIKE</td>
          <td>a4480651-b9b4-4c02-8a59-c53a9a8f7d13</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NIKE</td>
          <td>81c74d7b-0f50-468b-b14e-ba36e9818ca4</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NIKE</td>
          <td>7c8f53cf-0c26-416c-891b-095761fb5d38</td>
          <td>22.334417</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NIKE</td>
          <td>e9a98454-ced3-4f79-96fd-894684465603</td>
          <td>22.334417</td>
        </tr>
      </tbody>
    </table>
    </div>



Multi-Vector Search Across Multiple Fields
==========================================

You can easily add more to your search by extending your vector search
query as belows.

.. code:: ipython3

    results = ds.advanced_search(
        query="nike",
        fields_to_search=["prod_name"],
        vector_search_query=[
            {"vector": vector, "field": "prod_name_use_vector_"},
            {"vector": vector, "field": "image_path_clip_vector_"},
        ],
        select_fields=["prod_name"],  # results to return
    )
    pd.DataFrame(results["results"])




.. raw:: html

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
          <th>prod_name</th>
          <th>_id</th>
          <th>_relevance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NIKE</td>
          <td>890e1643-294e-4fdd-8787-1b0b325c6069</td>
          <td>22.390835</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NIKE</td>
          <td>3c12c230-b2a0-4706-9f4d-7f929ffba714</td>
          <td>22.390250</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NIKE</td>
          <td>8bba89a1-b1dd-4a2f-b4e8-68437d7b3c82</td>
          <td>22.385850</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NIKE</td>
          <td>a4480651-b9b4-4c02-8a59-c53a9a8f7d13</td>
          <td>22.385597</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NIKE</td>
          <td>3b875970-43f3-4342-9372-a9163431c839</td>
          <td>22.383432</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NIKE</td>
          <td>280cb2fb-fc40-4052-acc1-556c08493d24</td>
          <td>22.383057</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NIKE</td>
          <td>81c74d7b-0f50-468b-b14e-ba36e9818ca4</td>
          <td>22.377310</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NIKE</td>
          <td>09b3b546-2b7d-4b44-9b5f-6d7350af2bff</td>
          <td>22.372906</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NIKE</td>
          <td>901b8471-f7af-4dba-a3f3-ecc494f5093a</td>
          <td>22.367360</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NIKE</td>
          <td>b203ebbb-f75b-45c8-8a45-1d9322f2750d</td>
          <td>22.365366</td>
        </tr>
      </tbody>
    </table>
    </div>



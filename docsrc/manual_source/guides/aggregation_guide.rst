🏘️ Aggregation QuickStart
=========================

|Open In Colab|

Installation
============

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/general-features/aggregations/_notebooks/RelevanceAI_ReadMe_Quickstart_Aggregations.ipynb

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0

Setup
=====

You can sign up/login and find your credentials here:
https://cloud.relevance.ai/sdk/api Once you have signed up, click on the
value under ``Activation token`` and paste it here

.. code:: python

    from relevanceai import Client

    client = Client()


.. parsed-literal::

    Activation token (you can find it here: https://cloud.relevance.ai/sdk/api )

    Activation Token: ··········
    Connecting to us-east-1...
    You can view all your datasets at https://cloud.relevance.ai/datasets/
    Welcome to RelevanceAI. Logged in as 334fe5fb667b3a64dada.


Data
====

.. code:: python

    import pandas as pd
    from relevanceai.utils.datasets import get_realestate_dataset

    # Retrieve our sample dataset. - This comes in the form of a list of documents.
    documents = get_realestate_dataset()

    # ToDo: Remove this cell when the dataset is updated

    for d in documents:
        if "_clusters_" in d:
            del d["_clusters_"]

    pd.DataFrame.from_dict(documents).head()




.. raw:: html


      <div id="df-c6baa545-5f21-4092-b3a6-110875a3a28f">
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
          <th>image_url_4_vector_</th>
          <th>hasFloorplan</th>
          <th>image_url_vector_</th>
          <th>listingType</th>
          <th>image_url_2_vector_</th>
          <th>image_url_2</th>
          <th>propertyDetails</th>
          <th>listingSlug</th>
          <th>id</th>
          <th>headline</th>
          <th>...</th>
          <th>image_url_5_clip_vector_</th>
          <th>image_url_2_label</th>
          <th>image_url_4_label</th>
          <th>image_url_2_clip_vector_</th>
          <th>image_url_4_clip_vector_</th>
          <th>image_url_5_label</th>
          <th>image_url_clip_vector_</th>
          <th>image_url_label</th>
          <th>_cluster_</th>
          <th>_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>False</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>Rent</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>https://bucket-api.domain.com.au/v1/bucket/ima...</td>
          <td>{'area': 'Eastern Suburbs', 'carspaces': 2, 's...</td>
          <td>407-39-kent-street-mascot-nsw-2020-14806988</td>
          <td>14806988</td>
          <td>Stunning &amp; Modern Two Bedroom Apartment</td>
          <td>...</td>
          <td>[-0.4681514799594879, 0.08181382715702057, 0.1...</td>
          <td>hoosegow</td>
          <td>clubrooms</td>
          <td>[-0.4723101556301117, 0.012517078779637814, -0...</td>
          <td>[-0.6319758296012878, 0.1783788651227951, 0.13...</td>
          <td>mudrooms</td>
          <td>[-0.37417566776275635, 0.05725931376218796, -0...</td>
          <td>showrooms</td>
          <td>{'image_url_vector_': {'default': 0}, 'image_t...</td>
          <td>-0JggHcBgSy8FC2yCzRU</td>
        </tr>
        <tr>
          <th>1</th>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>False</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>Rent</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>https://bucket-api.domain.com.au/v1/bucket/ima...</td>
          <td>{'area': 'Eastern Suburbs', 'streetNumber': '2...</td>
          <td>2-256-new-south-head-double-bay-nsw-2028-14816127</td>
          <td>14816127</td>
          <td>Two Bedrooms Apartments just newly renovated</td>
          <td>...</td>
          <td>[-0.4457785189151764, 0.14002937078475952, -0....</td>
          <td>viewings</td>
          <td>mudroom</td>
          <td>[-0.37797173857688904, 0.04217493161559105, -0...</td>
          <td>[-0.6865466833114624, 0.19351454079151154, 0.1...</td>
          <td>mudroom</td>
          <td>[-0.5267254114151001, 0.22717250883579254, -0....</td>
          <td>appartements</td>
          <td>{'image_url_vector_': {'default': 0}, 'image_t...</td>
          <td>-0JggHcBgSy8FC2yCzVU</td>
        </tr>
        <tr>
          <th>2</th>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>True</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>Rent</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>https://bucket-api.domain.com.au/v1/bucket/ima...</td>
          <td>{'area': 'Eastern Suburbs', 'streetNumber': '1...</td>
          <td>19-11-21-flinders-street-surry-hills-nsw-2010-...</td>
          <td>14842628</td>
          <td>Iconic lifestyle pad in Urbis building</td>
          <td>...</td>
          <td>[-0.06582163274288177, 0.10252979397773743, 0....</td>
          <td>appartements</td>
          <td>backsplash</td>
          <td>[0.060137778520584106, 0.31164053082466125, 0....</td>
          <td>[-0.20558945834636688, 0.6132649183273315, 0.0...</td>
          <td>serigraph</td>
          <td>[-0.2266240119934082, 0.3205014765262604, 0.19...</td>
          <td>appartements</td>
          <td>{'image_url_vector_': {'default': 0}, 'image_t...</td>
          <td>-0JggHcBgSy8FC2ykDbk</td>
        </tr>
        <tr>
          <th>3</th>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>False</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>Rent</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>https://bucket-api.domain.com.au/v1/bucket/ima...</td>
          <td>{'area': 'Inner West', 'streetNumber': '13', '...</td>
          <td>13-formosa-st-drummoyne-nsw-2047-14828984</td>
          <td>14828984</td>
          <td>Heritage Semi to rent</td>
          <td>...</td>
          <td>[-0.334237277507782, 0.140365868806839, -0.236...</td>
          <td>kitchen</td>
          <td>entryway</td>
          <td>[-0.32477402687072754, 0.4767194986343384, 0.1...</td>
          <td>[0.12064582854509354, 0.3271999657154083, -0.2...</td>
          <td>appartements</td>
          <td>[-0.11818409711122513, 0.09542372077703476, -0...</td>
          <td>pub</td>
          <td>{'image_url_vector_': {'default': 0}, 'image_t...</td>
          <td>-0JggHcBgSy8FC2ykDfk</td>
        </tr>
        <tr>
          <th>4</th>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>False</td>
          <td>[0.0394604466855526, 0, 5.5613274574279785, 0....</td>
          <td>Rent</td>
          <td>[0.24612084031105042, 0.347802996635437, 0.574...</td>
          <td>https://bucket-api.domain.com.au/v1/bucket/ima...</td>
          <td>{'area': 'St George', 'carspaces': 1, 'streetN...</td>
          <td>103-11-17-woodville-street-hurstville-nsw-2220...</td>
          <td>14741619</td>
          <td>UNIQUE APARTMENT IN PRIME LOCATION</td>
          <td>...</td>
          <td>[-0.3391430079936981, 0.024984989315271378, -0...</td>
          <td>kitchen</td>
          <td>sideman</td>
          <td>[-0.3949810862541199, 0.3241899311542511, -0.1...</td>
          <td>[1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-...</td>
          <td>vitrine</td>
          <td>[-0.28189733624458313, 0.061684366315603256, -...</td>
          <td>cornlofts</td>
          <td>{'image_url_vector_': {'default': 5}, 'image_t...</td>
          <td>-0JhgHcBgSy8FC2y9TjX</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 34 columns</p>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-c6baa545-5f21-4092-b3a6-110875a3a28f')"
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
              document.querySelector('#df-c6baa545-5f21-4092-b3a6-110875a3a28f button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';

            async function convertToInteractive(key) {
              const element = document.querySelector('#df-c6baa545-5f21-4092-b3a6-110875a3a28f');
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




.. code:: python

    ds = client.Dataset("quickstart_aggregation")
    ds.insert_documents(documents)


.. parsed-literal::

    while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/quickstart_aggregation/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


1. Grouping the Data
====================

In general, the group-by field is structured as

::

   {"name": ALIAS,
   "field": FIELD,
   "agg": TYPE-OF-GROUP}

Categorical Data
----------------

.. code:: python

    location_group = {
        "name": "location",
        "field": "propertyDetails.area",
        "agg": "category",
    }

Numerical Data
--------------

.. code:: python

    bedrooms_group = {
        "name": "bedrooms",
        "field": "propertyDetails.bedrooms",
        "agg": "numeric",
    }

Putting it Together
-------------------

.. code:: python

    groupby = [location_group, bedrooms_group]

2. Creating Aggregation Metrics
===============================

In general, the aggregation field is structured as

::

   {"name": ALIAS,
   "field": FIELD,
   "agg": TYPE-OF-AGG}

Average, Minimum and Maximum
----------------------------

.. code:: python

    avg_price_metric = {"name": "avg_price", "field": "priceDetails.price", "agg": "avg"}
    max_price_metric = {"name": "max_price", "field": "priceDetails.price", "agg": "max"}
    min_price_metric = {"name": "min_price", "field": "priceDetails.price", "agg": "min"}

Sum
---

.. code:: python

    sum_bathroom_metric = {
        "name": "bathroom_sum",
        "field": "propertyDetails.bathrooms",
        "agg": "sum",
    }

Putting it Together
-------------------

.. code:: python

    metrics = [avg_price_metric, max_price_metric, min_price_metric, sum_bathroom_metric]

3. Combining Grouping and Aggregating
=====================================

.. code:: python

    results = ds.aggregate(metrics=metrics, groupby=groupby)

.. code:: python

    from jsonshower import show_json

    show_json(results, text_fields=list(results["results"][0].keys()))




.. raw:: html

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>frequency</th>
          <th>location</th>
          <th>bedrooms</th>
          <th>avg_price</th>
          <th>max_price</th>
          <th>min_price</th>
          <th>bathroom_sum</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>10</td>
          <td>Eastern Suburbs</td>
          <td>2</td>
          <td>670.000000</td>
          <td>780.0</td>
          <td>580.0</td>
          <td>17</td>
        </tr>
        <tr>
          <th>1</th>
          <td>8</td>
          <td>Eastern Suburbs</td>
          <td>1</td>
          <td>554.000000</td>
          <td>670.0</td>
          <td>450.0</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Eastern Suburbs</td>
          <td>3</td>
          <td>850.000000</td>
          <td>900.0</td>
          <td>800.0</td>
          <td>5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>9</td>
          <td>North Shore - Lower</td>
          <td>1</td>
          <td>516.666667</td>
          <td>600.0</td>
          <td>450.0</td>
          <td>9</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7</td>
          <td>North Shore - Lower</td>
          <td>2</td>
          <td>525.000000</td>
          <td>525.0</td>
          <td>525.0</td>
          <td>9</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2</td>
          <td>North Shore - Lower</td>
          <td>3</td>
          <td>900.000000</td>
          <td>900.0</td>
          <td>900.0</td>
          <td>4</td>
        </tr>
        <tr>
          <th>6</th>
          <td>8</td>
          <td>Inner West</td>
          <td>2</td>
          <td>447.500000</td>
          <td>495.0</td>
          <td>400.0</td>
          <td>11</td>
        </tr>
        <tr>
          <th>7</th>
          <td>4</td>
          <td>Inner West</td>
          <td>1</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4</td>
        </tr>
        <tr>
          <th>8</th>
          <td>3</td>
          <td>Inner West</td>
          <td>3</td>
          <td>1070.000000</td>
          <td>1070.0</td>
          <td>1070.0</td>
          <td>7</td>
        </tr>
        <tr>
          <th>9</th>
          <td>1</td>
          <td>Inner West</td>
          <td>4</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1</td>
        </tr>
        <tr>
          <th>10</th>
          <td>5</td>
          <td>Northern Suburbs</td>
          <td>1</td>
          <td>460.000000</td>
          <td>500.0</td>
          <td>420.0</td>
          <td>5</td>
        </tr>
        <tr>
          <th>11</th>
          <td>5</td>
          <td>Northern Suburbs</td>
          <td>2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>8</td>
        </tr>
        <tr>
          <th>12</th>
          <td>3</td>
          <td>Northern Suburbs</td>
          <td>3</td>
          <td>620.000000</td>
          <td>680.0</td>
          <td>560.0</td>
          <td>6</td>
        </tr>
        <tr>
          <th>13</th>
          <td>1</td>
          <td>Northern Suburbs</td>
          <td>4</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1</td>
        </tr>
        <tr>
          <th>14</th>
          <td>4</td>
          <td>St George</td>
          <td>2</td>
          <td>370.000000</td>
          <td>370.0</td>
          <td>370.0</td>
          <td>5</td>
        </tr>
        <tr>
          <th>15</th>
          <td>2</td>
          <td>St George</td>
          <td>1</td>
          <td>340.000000</td>
          <td>350.0</td>
          <td>330.0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>16</th>
          <td>2</td>
          <td>St George</td>
          <td>3</td>
          <td>640.000000</td>
          <td>700.0</td>
          <td>580.0</td>
          <td>4</td>
        </tr>
        <tr>
          <th>17</th>
          <td>2</td>
          <td>St George</td>
          <td>4</td>
          <td>700.000000</td>
          <td>700.0</td>
          <td>700.0</td>
          <td>4</td>
        </tr>
        <tr>
          <th>18</th>
          <td>4</td>
          <td>Sydney City</td>
          <td>2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>6</td>
        </tr>
        <tr>
          <th>19</th>
          <td>3</td>
          <td>Sydney City</td>
          <td>1</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>3</td>
        </tr>
        <tr>
          <th>20</th>
          <td>1</td>
          <td>Sydney City</td>
          <td>3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
        </tr>
        <tr>
          <th>21</th>
          <td>3</td>
          <td>Parramatta</td>
          <td>2</td>
          <td>450.000000</td>
          <td>450.0</td>
          <td>450.0</td>
          <td>5</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1</td>
          <td>Parramatta</td>
          <td>1</td>
          <td>430.000000</td>
          <td>430.0</td>
          <td>430.0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>23</th>
          <td>3</td>
          <td>Canterbury/Bankstown</td>
          <td>2</td>
          <td>300.000000</td>
          <td>300.0</td>
          <td>300.0</td>
          <td>3</td>
        </tr>
        <tr>
          <th>24</th>
          <td>1</td>
          <td>Hills</td>
          <td>4</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
        </tr>
        <tr>
          <th>25</th>
          <td>1</td>
          <td>Northern Beaches</td>
          <td>3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
        </tr>
        <tr>
          <th>26</th>
          <td>1</td>
          <td>Western Sydney</td>
          <td>2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>

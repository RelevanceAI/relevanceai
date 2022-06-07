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

.. code:: ipython3

    In [1]: %load_ext autoreload

    In [2]: %autoreload 2

.. code:: ipython3

    from relevanceai import Client

    # Instantiate the client and authenticate
    client = Client()

    # This will prompt a link to collect your API token which includes your project and API key

📩 Upload Some Data
~~~~~~~~~~~~~~~~~~~

1️⃣. Open a new **Dataset**

2️⃣. **Insert** some documents

.. code:: ipython3

    from relevanceai.utils import example_documents

    documents = example_documents("retail_reviews_small", number_of_documents=100)

.. code:: ipython3

    dataset_id = "retail_reviews"
    # The dataset name that we have decided, this can be whatever you want for your own data
    dataset = client.Dataset(dataset_id=dataset_id)
    # Instantiate the dataset

.. code:: ipython3

    dataset.insert_documents(documents)


.. parsed-literal::

    while inserting, you can visit monitor the dataset at https://cloud.relevance.ai/dataset/retail_reviews/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


You can view your dataset quickly using ``dataset.head`` just like in
Pandas!

.. code:: ipython3

    dataset.head()


.. parsed-literal::

    https://cloud.relevance.ai/dataset/retail_reviews/dashboard/data?page=1




.. raw:: html

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Unnamed: 0</th>
          <th>id</th>
          <th>name</th>
          <th>asins</th>
          <th>brand</th>
          <th>categories</th>
          <th>keys</th>
          <th>manufacturer</th>
          <th>reviews.date</th>
          <th>reviews.dateAdded</th>
          <th>reviews.dateSeen</th>
          <th>reviews.didPurchase</th>
          <th>reviews.doRecommend</th>
          <th>reviews.id</th>
          <th>reviews.numHelpful</th>
          <th>reviews.rating</th>
          <th>reviews.sourceURLs</th>
          <th>reviews.text</th>
          <th>reviews.title</th>
          <th>reviews.userCity</th>
          <th>reviews.userProvince</th>
          <th>reviews.username</th>
          <th>insert_date_</th>
          <th>_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1408</td>
          <td>AVpjEN4jLJeJML43rpUe</td>
          <td>Brand New Amazon Kindle Fire 16gb 7 Ips Display Tablet Wifi 16 Gb Blue,,,</td>
          <td>B018Y225IA</td>
          <td>Amazon</td>
          <td>Computers/Tablets & Networking,Tablets & eBook Readers,Computers & Tablets,Tablets,All Tablets</td>
          <td>841667103143,0841667103143,brandnewamazonkindlefire16gb7ipsdisplaytabletwifi16gbblue/5025500,brandnewamazonkindlefire16gb7ipsdisplaytabletwifi16gbblue/b018y225ia,brandnewamazonkindlefire16gb7ipsdisplaytabletwifi16gbblue/201625338826,brandnewamazonkindlefire16gb7ipsdisplaytabletwifi16gbblue/362123960192,amazon/b018y225ia</td>
          <td>Amazon</td>
          <td>2017-05-18T00:00:00.000Z</td>
          <td>None</td>
          <td>2017-08-27T00:00:00Z,2017-08-09T00:00:00Z,2017-06-07T00:00:00Z,2017-07-08T00:00:00Z,2017-08-06T00:00:00Z,2017-08-19T00:00:00Z</td>
          <td>None</td>
          <td>True</td>
          <td>None</td>
          <td>0</td>
          <td>5</td>
          <td>http://reviews.bestbuy.com/3545/5025500/reviews.htm?format=embedded&page=15,http://reviews.bestbuy.com/3545/5025500/reviews.htm?format=embedded&page=14,http://reviews.bestbuy.com/3545/5025500/reviews.htm?format=embedded&page=3,http://reviews.bestbuy.com/3545/5025500/reviews.htm?format=embedded&page=11</td>
          <td>I bought this for my daughter for Christmas. I could set her as a user and make it safe for her age. I downloaded books and apps with ease. I'm glad that she is loving her Fire.</td>
          <td>Amazon Fire</td>
          <td>None</td>
          <td>None</td>
          <td>Renessa14</td>
          <td>2022-06-07T01:32:23.307Z</td>
          <td>00003142-ef66-39df-a84a-2f652c4c3e1c</td>
        </tr>
        <tr>
          <th>1</th>
          <td>563</td>
          <td>AVphgVaX1cnluZ0-DR74</td>
          <td>Fire Tablet, 7 Display, Wi-Fi, 8 GB - Includes Special Offers, Magenta</td>
          <td>B018Y229OU</td>
          <td>Amazon</td>
          <td>Fire Tablets,Tablets,Computers & Tablets,All Tablets,Electronics, Tech Toys, Movies, Music,Electronics,iPad & Tablets,Android Tablets,Frys</td>
          <td>firetablet7displaywifi8gbincludesspecialoffersmagenta/5025800,841667103105,0841667103105,amazon/b018y229ou,firetablet7displaywifi8gbincludesspecialoffersmagenta/b018y229ou</td>
          <td>Amazon</td>
          <td>2016-04-16T00:00:00.000Z</td>
          <td>2017-05-21T01:18:21Z</td>
          <td>2017-04-30T00:08:00.000Z,2017-06-07T08:18:00.000Z</td>
          <td>None</td>
          <td>True</td>
          <td>None</td>
          <td>1</td>
          <td>5</td>
          <td>http://reviews.bestbuy.com/3545/5025800/reviews.htm?format=embedded&page=767,http://reviews.bestbuy.com/3545/5025800/reviews.htm?format=embedded&page=800</td>
          <td>This was a gift for a senior citizen that is not a fan of computers and technology. After just a few minutes of instruction she picked right up on how much fun she could have with her down loaded games and she also learned how to reach out on the internet for information on subjects that interest her. She is very happy with her gift!</td>
          <td>Great tablet for first time user</td>
          <td>None</td>
          <td>None</td>
          <td>donfield</td>
          <td>2022-06-07T01:32:23.307Z</td>
          <td>00088906-e56b-3507-8300-d9205d0bee23</td>
        </tr>
        <tr>
          <th>2</th>
          <td>313</td>
          <td>AVphgVaX1cnluZ0-DR74</td>
          <td>Fire Tablet, 7 Display, Wi-Fi, 8 GB - Includes Special Offers, Magenta</td>
          <td>B018Y229OU</td>
          <td>Amazon</td>
          <td>Fire Tablets,Tablets,Computers & Tablets,All Tablets,Electronics, Tech Toys, Movies, Music,Electronics,iPad & Tablets,Android Tablets,Frys</td>
          <td>firetablet7displaywifi8gbincludesspecialoffersmagenta/5025800,841667103105,0841667103105,amazon/b018y229ou,firetablet7displaywifi8gbincludesspecialoffersmagenta/b018y229ou</td>
          <td>Amazon</td>
          <td>2016-01-08T00:00:00.000Z</td>
          <td>2017-05-21T02:00:31Z</td>
          <td>2017-04-30T00:14:00.000Z,2017-06-07T08:13:00.000Z</td>
          <td>None</td>
          <td>True</td>
          <td>None</td>
          <td>0</td>
          <td>4</td>
          <td>http://reviews.bestbuy.com/3545/5025800/reviews.htm?format=embedded&page=1049,http://reviews.bestbuy.com/3545/5025800/reviews.htm?format=embedded&page=1083</td>
          <td>NICE TABLET, BUT IT WANTS YOU TO PUT ALL YOUR BILLING INFORMATION IN AND A CREDIT CARD TO DO ALL PURCHASES. IS A HASSLE TO PULL UP PROGRAMS BECAUSE OF THIS. GOOD FOR A CHILD.</td>
          <td>NICE TABLET FOR CHILDREN</td>
          <td>None</td>
          <td>None</td>
          <td>FASTJAKE</td>
          <td>2022-06-07T01:32:23.307Z</td>
          <td>0012542b-3e5a-31e5-9cf3-0100f64666ae</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4752</td>
          <td>AVpfl8cLLJeJML43AE3S</td>
          <td>Amazon Fire Tv,,,\r\nAmazon Fire Tv,,,</td>
          <td>B00L9EPT8O,B01E6AO69U</td>
          <td>Amazon</td>
          <td>Stereos,Remote Controls,Amazon Echo,Audio Docks & Mini Speakers,Amazon Echo Accessories,Kitchen & Dining Features,Speaker Systems,Electronics,TVs Entertainment,Clearance,Smart Hubs & Wireless Routers,Featured Brands,Wireless Speakers,Smart Home & Connected Living,Home Security,Kindle Store,Home Automation,Home, Garage & Office,Home,Voice-Enabled Smart Assistants,Virtual Assistant Speakers,Portable Audio & Headphones,Electronics Features,Amazon Device Accessories,iPod, Audio Player Accessories,Home & Furniture Clearance,Consumer Electronics,Smart Home,Surveillance,Home Improvement,Smart Home & Home Automation Devices,Smart Hubs,Home Safety & Security,Voice Assistants,Alarms & Sensors,Amazon Devices,Audio,Holiday Shop</td>
          <td>echowhite/263039693056,echowhite/152558276095,echowhite/292178880467,echowhite/222588935706,echowhite/253120140398,echowhite/322577436254,echowhite/122597356284,echowhite/132263972952,echowhite/322586415668,echowhite/152626395386,echowhite/272724680159,echowhite/222587602421,echowhite/122474318097,echowhite/5588528,echowhite/112567699636,echowhite/272768463386,echowhite/332175902683,echowhite/311908601694,echowhite/292041139369,echowhite/192239032596,echowhite/272768869474,0841667112862,echowhite/222507973621,echowhite/112391858963,echowhite/291992370210,echowhite/b00l9ept8o,echowhite/112480241614,echowhite/b01e6ao69u,echowhite/322589755316,echowhite/322574315372,echowhite/253051886606,echowhite/382165760287,echowhite/222582493180,echowhite/282581384521,echowhite/112479310908,echowhite/302201691992,echowhite/201761456849,echowhite/amechow2k,echowhite/132262816901,echowhite/282571823011,echowhite/322511136772,841667112862,echowhite/232407174148,echowhite/322441917397,echowhite/amechow,echowhite/332296207643,echowhite/152610914446,echowhite/222578584785,echowhite/162591117080,echowhite/162593787621,echowhite/232407374203,echowhite/162595518416,echowhite/152623638099,amazon/b01e6ao69u</td>
          <td>Amazon</td>
          <td>2016-07-15T00:00:00.000Z</td>
          <td>None</td>
          <td>2017-09-28T00:00:00Z,2017-09-08T00:00:00Z,2017-09-12T00:00:00Z,2017-08-31T00:00:00Z,2017-08-08T00:00:00Z,2017-08-15T00:00:00Z,2017-08-01T00:00:00Z</td>
          <td>None</td>
          <td>True</td>
          <td>None</td>
          <td>0</td>
          <td>5</td>
          <td>http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=775,http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=702,http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=706,http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=676,http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=608,http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=615,http://reviews.bestbuy.com/3545/5588528/reviews.htm?format=embedded&page=600</td>
          <td>Echo is amazing still learning more things,great product</td>
          <td>It's great</td>
          <td>None</td>
          <td>None</td>
          <td>Igotit</td>
          <td>2022-06-07T01:32:23.307Z</td>
          <td>00151c99-bf82-312d-adad-d981019ba47b</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1927</td>
          <td>AVphgVaX1cnluZ0-DR74</td>
          <td>Fire Tablet, 7 Display, Wi-Fi, 8 GB - Includes Special Offers, Magenta</td>
          <td>B018Y229OU</td>
          <td>Amazon</td>
          <td>Fire Tablets,Tablets,Computers & Tablets,All Tablets,Electronics, Tech Toys, Movies, Music,Electronics,iPad & Tablets,Android Tablets,Frys</td>
          <td>firetablet7displaywifi8gbincludesspecialoffersmagenta/5025800,841667103105,0841667103105,amazon/b018y229ou,firetablet7displaywifi8gbincludesspecialoffersmagenta/b018y229ou</td>
          <td>Amazon</td>
          <td>2016-01-22T00:00:00.000Z</td>
          <td>2017-05-21T02:29:02Z</td>
          <td>2017-04-30T00:18:00.000Z,2017-06-07T08:13:00.000Z</td>
          <td>None</td>
          <td>True</td>
          <td>None</td>
          <td>0</td>
          <td>5</td>
          <td>http://reviews.bestbuy.com/3545/5025800/reviews.htm?format=embedded&page=925,http://reviews.bestbuy.com/3545/5025800/reviews.htm?format=embedded&page=959</td>
          <td>Exceeded my expectations would recommend to anyone who wants a good tablet for a good price.</td>
          <td>Well worth it</td>
          <td>None</td>
          <td>None</td>
          <td>Anice183</td>
          <td>2022-06-07T01:32:23.307Z</td>
          <td>0017e388-76b4-3c72-beb4-5e6ceece824e</td>
        </tr>
      </tbody>
    </table>



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

    # !pip install vectorhub[encoders-text-tfhub] -qqq

🤩 Vectorize in one line
~~~~~~~~~~~~~~~~~~~~~~~~

We support vectorizing text in just 1 line.

.. code:: ipython3

    # The text fields here are the ones we wish to construct vector representations for
    text_fields = ["reviews.text"]
    dataset.vectorize_text(fields=text_fields)

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

First, let us see what vector fields are availbale in the dataset.

.. code:: ipython3

    dataset.list_vector_fields()




.. parsed-literal::

    ['reviews.text_all-mpnet-base-v2_vector_']



.. code:: ipython3

    model = "kmeans"
    number_of_clusters = 20
    alias = "my_clustering"
    vector_fields = dataset.list_vector_fields()
    dataset.cluster(vector_fields=vector_fields, model=model, alias=alias)

🔗 The above step will produce a link to your first cluster app!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click the link provided to view your newly generated clusters in a
`dashboard
app <https://cloud.relevance.ai/dataset/retail_reviews/deploy/cluster/59066979f4876d91beea/QVdEaHJuOEJ5Qy1VVnVsVDhndjM6eG9HaVg2RGtTTUdWNXFFQjNhZUg0QQ/LZpGq38B8_iiYmskWDEn/us-east-1/>`__
|image1|

.. |image1| image:: https://i.gyazo.com/55a026bfe8e3becf06e7fceed4e146f2.png

Search Application
------------------

You can also build a search application in just 1 line of code.

This search application can be built by using

.. code:: ipython3

    dataset.launch_search_app()


.. parsed-literal::

    https://cloud.relevance.ai/dataset/retail_reviews/deploy/recent/search


You can view an example of our text search below.

.. figure:: text-search-gif.gif
   :alt: Text Search

   Text Search

Extract Sentiment
-----------------

You can add sentiment to your dataset. After adding sentiment

.. code:: ipython3

    dataset.extract_sentiment(text_fields=["reviews.text"]

Add Labels To Your Dataset
--------------------------

Labelling refers to when you apply a vector search from one tag to
another.

.. code:: ipython3

    labels = [{"label": "Furniture", "label": "Home office", "label": "Electronics"}]

.. code:: ipython3

    label_dataset.insert_documents(labels)


.. parsed-literal::

    while inserting, you can visit monitor the dataset at https://cloud.relevance.ai/dataset/retail-label/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


.. code:: ipython3

    # Vectorize like you would with a normal dataset
    label_dataset.vectorize_text(
        fields=['label'],
        output_fields=["label_vector_"]
    )



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Vector field is `label_vector_`
    ✅ All documents inserted/edited successfully.
    Storing operation metadata...
    ✅ You have successfully inserted metadata.




.. parsed-literal::

    <relevanceai.operations_new.vectorize.text.ops.VectorizeTextOps at 0x284649f70>



.. code:: ipython3

    dataset.label_from_dataset(
        vector_fields=dataset.list_vector_fields(),
        label_dataset=label_dataset
    )



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    ✅ All documents inserted/edited successfully.
    Storing operation metadata...
    ✅ You have successfully inserted metadata.




.. parsed-literal::

    <relevanceai.operations_new.label.ops.LabelOps at 0x1774ff5b0>



You can now see the labels on your dataset on Relevance AI.

.. figure:: attachment:image.png
   :alt: image.png

   image.png

Want to quickly create some example applications with Relevance AI?
Check out some other guides below! - `Text-to-image search with OpenAI’s
CLIP <https://docs.relevance.ai/docs/quickstart-text-to-image-search>`__
- `Hybrid Text search with Universal Sentence Encoder using
Vectorhub <https://docs.relevance.ai/docs/quickstart-text-search>`__ -
`Text search with Universal Sentence Encoder Question Answer from
Google <https://docs.relevance.ai/docs/quickstart-question-answering>`__

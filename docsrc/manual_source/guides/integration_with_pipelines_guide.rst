🤗 Hugging Face Pipelines
========================

.. code:: ipython3

    In [1]: %load_ext autoreload

    In [2]: %autoreload 2

Set-up
======

.. code:: ipython3

    from relevanceai import Client

.. code:: ipython3

    client = Client()

.. code:: ipython3

    ds = client.Dataset("ecommerce-example")

.. code:: ipython3

    # Uncomment out below if you want the e-commerce dataset
    # from relevanceai.utils.datasets import get_ecommerce_dataset_encoded
    # docs = get_ecommerce_dataset_encoded()
    # ds.upsert_documents(docs)

Installing Transformers
=======================

.. code:: ipython3

    # !pip install -q transformers

Here, we define a sample Transformers pipeline.

Transformers Pipeline
=====================

.. code:: ipython3

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = "My name is Wolfgang and I live in Berlin"

    ner_results = nlp(example)
    print(ner_results)


.. parsed-literal::

    [{'entity': 'B-PER', 'score': 0.9990139, 'index': 4, 'word': 'Wolfgang', 'start': 11, 'end': 19}, {'entity': 'B-LOC', 'score': 0.999645, 'index': 9, 'word': 'Berlin', 'start': 34, 'end': 40}]


Running Transformers
====================

.. code:: ipython3

    # We can apply HuggingFace Pipelines
    ds.apply_transformers_pipeline(
        text_fields=["product_title"], pipeline=nlp
    )



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    ✅ All documents inserted/edited successfully.
    Storing operation metadata...
    ✅ You have successfully inserted metadata.


Viewing NER Results
===================

We can see how they are stored below!

.. code:: ipython3

    ds.schema




.. parsed-literal::

    {'_ner_': 'dict',
     '_ner_.dslim/bert-base-NER': 'dict',
     '_ner_.dslim/bert-base-NER.product_title': 'dict',
     '_ner_.dslim/bert-base-NER.product_title.end': 'numeric',
     '_ner_.dslim/bert-base-NER.product_title.entity': 'text',
     '_ner_.dslim/bert-base-NER.product_title.index': 'numeric',
     '_ner_.dslim/bert-base-NER.product_title.score': 'numeric',
     '_ner_.dslim/bert-base-NER.product_title.start': 'numeric',
     '_ner_.dslim/bert-base-NER.product_title.word': 'text',
     'insert_date_': 'date',
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

    ds.head(select_fields=['product_title', '_ner_.dslim/bert-base-NER.product_title'])


.. parsed-literal::

    https://cloud.relevance.ai/dataset/ecommerce-example/dashboard/data?page=1




.. raw:: html

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>product_title</th>
          <th>_id</th>
          <th>_ner_.dslim/bert-base-NER.product_title</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Seville Classics 10-Drawer Organizer Cart</td>
          <td>0007a669-07e9-4a4a-b63c-40312690b381</td>
          <td>[{'score': 0.9612383842468262, 'start': 0, 'index': 1, 'end': 7, 'word': 'Seville', 'entity': 'B-MISC'}, {'score': 0.9937147498130798, 'start': 8, 'index': 2, 'end': 16, 'word': 'Classics', 'entity': 'I-MISC'}]</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Nike Women's 'Zoom Hyperquickness' Synthetic Athletic Shoe (Size 6 )</td>
          <td>00445000-a8ed-4523-b610-f70aa79d47f7</td>
          <td>[{'score': 0.9958500862121582, 'start': 0, 'index': 1, 'end': 4, 'word': 'Nike', 'entity': 'B-ORG'}, {'score': 0.5315994024276733, 'start': 19, 'index': 8, 'end': 20, 'word': 'H', 'entity': 'I-MISC'}, {'score': 0.5882592797279358, 'start': 24, 'index': 10, 'end': 27, 'word': '##qui', 'entity': 'I-MISC'}, {'score': 0.8706270456314087, 'start': 45, 'index': 17, 'end': 53, 'word': 'Athletic', 'entity': 'I-ORG'}, {'score': 0.8434967994689941, 'start': 54, 'index': 18, 'end': 55, 'word': 'S', 'entity': 'I-ORG'}, {'score': 0.6403887867927551, 'start': 55, 'index': 19, 'end': 58, 'word': '##hoe', 'entity': 'I-ORG'}]</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Men's DC Shoes Villain TX Black/Black/Black</td>
          <td>00a3d45e-2096-46aa-94c6-7d8480fb1436</td>
          <td>[{'score': 0.6886934041976929, 'start': 9, 'index': 5, 'end': 14, 'word': 'Shoes', 'entity': 'I-ORG'}, {'score': 0.4493587613105774, 'start': 32, 'index': 11, 'end': 37, 'word': 'Black', 'entity': 'B-LOC'}]</td>
        </tr>
        <tr>
          <th>3</th>
          <td>AGRA .5-ounce Under Eye and Neck Cream</td>
          <td>01317a4c-2136-4fa3-be56-c07d79a646b3</td>
          <td>[]</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Organize It All Black Storage Open Drawer Cube</td>
          <td>0165f12a-cc93-4306-8161-750511e9a997</td>
          <td>[{'score': 0.49954599142074585, 'start': 16, 'index': 5, 'end': 21, 'word': 'Black', 'entity': 'I-MISC'}, {'score': 0.6008133888244629, 'start': 22, 'index': 6, 'end': 24, 'word': 'St', 'entity': 'I-MISC'}, {'score': 0.762143075466156, 'start': 35, 'index': 9, 'end': 39, 'word': 'Draw', 'entity': 'I-MISC'}, {'score': 0.9884127974510193, 'start': 42, 'index': 11, 'end': 43, 'word': 'C', 'entity': 'I-MISC'}]</td>
        </tr>
      </tbody>
    </table>



We can also see how it can be found in our metadata!

.. code:: ipython3

    ds.metadata




.. parsed-literal::

    {'_operationhistory_': {'1653873505-991286': {'operation': 'dslim/bert-base-NER', 'parameters': "{'operation': 'dslim/bert-base-NER', 'values': {'text_fields': ['product_title'], 'pipeline': <transformers.pipelines.token_classification.TokenClassificationPipeline object at 0x2a09eed90>, 'task': 'ner', '_name': 'dslim/bert-base-NER', 'output_field': '_ner_.dslim/bert-base-NER.product_title'}}"}}}

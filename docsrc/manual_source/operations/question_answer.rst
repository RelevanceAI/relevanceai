Question Answer
==================

Basic
---------

The easiest way to add sentiment to a text field is using the 
`ds.question_answer` function.

Prior to adding sentiment, we will need to make sure to install HuggingFace's Transformers.

.. code-block::

   pip install -q transformers[sentencepiece]



.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("ecommerce")
    ds.question_answer(
        input_field="product_title",
        question="What brand shoes",
        output_field="_question_test",
        # Easily switch to a different HuggingFace model
        model_name="mrm8488/deberta-v3-base-finetuned-squadv2",
    )

For every document, you will get functions and formulas similar to the ones below:

.. code-block::

   {
      "_question_": {
         "what-brand-shoes": {
            "answer": "nike", # returns a string response
            "score": 0.48, # confidence of the answer
         }
      }
   }

API Reference
------------------

.. automodule:: relevanceai.operations.text.sentiment.sentiments
   :members:
   :exclude-members: __init__

.. automodule:: relevanceai.operations.text.sentiment.sentiment_workflow
   :members:
   :exclude-members: __init__

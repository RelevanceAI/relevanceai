Sentiment
=============

Basic
---------

The easiest way to add sentiment to a text field is using the
`ds.extract_sentiment` function.

Prior to adding sentiment, we will need to make sure to install HuggingFace's Transformers.

.. code-block::

   pip install -q transformers

.. code-block::

   from relevanceai import Client
   client = Client()
   ds = client.Dataset("sample")

   # Easily switch to a different HuggingFace model
   ds.extract_sentiment(
      text_fields=["sample_1_label"],
   )

For every document, you will get functions and formulas similar to the ones below:

.. code-block::

   {
      "_sentiment_": {
         "sample_1_label": {
            {
                "model_name": {
                "sentiment": sentiment, # positive / neutral / negative
                "score": np.round(float(scores[ranking[0]]), 4), # confidence of the sentiment
                "overall_sentiment_score": score if sentiment == "positive" else -score,
                # an overall sentiment score where -1 is negative and +1 is positive
            }
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

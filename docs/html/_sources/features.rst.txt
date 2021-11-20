Features
=============

Here is a list of the top features of the SDK.


Inserting with automatic multi-processing and multi-threading
----------------------------

Get multi-threading and multi-processing out of the box. The RelevanceAI Python package automatically gives you multi-threading and multi-processing out of the box!

.. code-block:: python

    client.insert_documents(
        dataset_id="match-example",
        docs=participant_frames,
        update_schema=True,
        overwrite=True,
    )

    def bulk_fn(docs):
        # bulk_fn receives a list of documents (python dictionaries)
        for d in docs:
            d["value_update"] = d["value"] + 2
        return docs

    client.insert_documents(
        dataset_id="match-example",
        docs=participant_frames,
        update_schema=True,
        overwrite=True,
        bulk_fn=bulk_fn)

Under the hood, we use multiprocessing for processing the `bulk_fn` and 
multi-threading to send data via network requests. However, if there is no `bulk_fn` supplied, it only multi-threads network requests.

Pull Update Push
--------------

Update documents within your collection based on a rule customised by you. The Pull-Update-Push Function loops through every document in your collection, brings it to your local computer where a function is applied (specified by you) and reuploaded to either an new collection or updated in the same collection. There is a logging functionality to keep track of which documents have been updated to save on network requests.

For example, consider a scenario where you have uploaded a dataset called 'test_dataset' containing integers up to 200.

.. code-block:: python
    original_collection = 'test_dataset'
    data = [{'_id': str(i)} for i in range(200)]
    client.datasets.bulk_insert(original_collection, data)

An example of sample data looks like this:

.. code-block:: python

    [{"_id": "0"}, {"_id": "1"}, ... {"_id": "199"}]

    def even_function(data):
        for i in data:
            if int(i['_id']) % 2 == 0:
                i['even'] = True
            else:
                i['even'] = False
        return data

This function is then included in the Pull-Update-Push Function to update every document in the uploaded collection.


.. code-block:: python

    client.pull_update_push(original_collection, even_function)

Alternatively, a new collection could be specified to direct where updated documents are uploaded into.

.. code-block:: python

    [{"_id": "0", "even": true}, {"_id": "1", "even": false}, ... {"_id": "199", "even": true}]

    client.delete_all_logs(original_collection)

Integration With VectorHub
-----------------------------

VectorHub is RelevanceAI's main vectorizer repository. 
For the models used here, we have abstracted away a lot of complexity from installation to encoding and have innate RelevanceAI support.

Using VectorHub models is as simple as (actual example):

.. code-block:: python

    # Insert in a dataframe
    import pandas as pd
    df = pd.read_csv("Grid view.csv")
    df['_id'] = df['sample']
    client.insert_df("sample-cn", df)

    # !pip install vectorhub[encoders-text-sentence-transformers]
    from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec
    model = SentenceTransformer2Vec()

    # Define an update function
    def encode_documents(docs):
        # Field and then the docs go here
        return model.encode_documents(
            ["current", "Longer"], docs)

    client.pull_update_push("sample-cn", encode_documents)

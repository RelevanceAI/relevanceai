# VecDB Python SDK 

This is the Python SDK of VecDB. This is the main Pythonic interface for the VecDB client.
Built mainly for users looking to experiment with vectors/embeddings without having to consistently
rely on the API.

## Installation 

The easiest way is to clone this package to your local directory, navigate to the directory and then run `pip install vecdb`.

This will automatically let you unlock the power of VecDB without having to rely on re-constructing API requests.

## How to use this package 

Instantiating the client.

```python
from vecdb import VecDBClient
vdb_client = VecDBClient(project, api_key)
```

Get multi-threading and multi-processing out of the box. The VecDB Python package automatically gives you multi-threading and multi-processing out of the box!

```python

def bulk_fn(docs):
    # bulk_fn receives a list of documents
    for d in docs:
        d["value_update"] = d["value"] + 2
    return docs

vdb_client.insert_documents(
    dataset_id="match-example",
    docs=participant_frames[0:50],
    update_schema=True,
    overwrite=True,
    bulk_fn=bulk_fn
)
```

Under the hood, our engineering team uses multiprocessing for processing the `bulk_fn` and then automatically uses multi-threading to send data via network requests. However, if there is no `bulk_fn` supplied, it automatically multi-threads network requests.


## Pull Update Push

Update documents within your collection based on a rule customised by you. The Pull-Update-Push Function loops through every document in your collection, brings it to your local computer where a function is applied (specified by you) and reuploaded to either an new collection or updated in the same collection. There is a logging functionality to keep track of which documents have been updated.

For example, consider a scenario where you have uploaded a dataset called 'test_dataset' containing integers up to 200. 

```python
original_collection = 'test_dataset'
data = [{'_id': i} for i in range(200)]
vec_client.datasets.bulk_insert(original_collection, data)
```

```json
Data: [{'_id': 0}, {'_id': 1}, ... {'_id': 199}]
```


Now, you want to create a new field in this dataset to flag whether the number is even or not. This function must input a list of documents (the original collection) and output another list of documents (to be updated).

```python
def even_function(data):
    for i in data:
        if int(i['_id']) % 2 == 0:
            i['even'] = True
        else:
            i['even'] = False
    return data
```
This function is then included in the Pull-Update-Push Function to update every document in the uploaded collection.
 
```python
vec_client.pull_update_push(original_collection, even_function)
```

```json
Updated Data: [{'_id': 0, 'even': True}, {'_id': 1, 'even': False}, ... {'_id': 199, 'even': True}]
```

Alternatively, a new collection could be specified to direct where updated documents are uploaded into.

```python
new_collection = 'updated_test_dataset'
vec_client.pull_update_push(original_collection, even_function, new_collection)
```
```json
New Data: [{'_id': 0, 'even': True}, {'_id': 1, 'even': False}, ... {'_id': 199, 'even': True}]
```

## Integration with VectorHub

VectorHub is RelevanceAI's main encoder repository. For the models used here, we have abstracted away a lot of 
complexity from installation to encoding and have innate VectorAI support. 

Using VectorHub models is as simple as (actual example): 

```python
# Inserting a dataframe
import pandas as pd
df = pd.read_csv("Grid view.csv")
df['_id'] = df['sample']
client.insert_df("sample-cn", df)

from vectorhub.encoders.text.tfhub import USE2Vec
model = USE2Vec()

# Define an update function
def encode_documents(docs):
    # Field and then the docs go here
    return model.encode_documents(["current", "Longer"], docs)

client.pull_update_push("sample-cn", encode_documents)

```

# relevanceai Python SDK 

For documentation about how to use this package, visit: https://docs.relevance.ai/docs

This is the Python SDK of relevanceai and is the main Pythonic interface for the relevanceai.

Built mainly for users looking to experiment with vectors/embeddings without having to consistently rely on the `requests` module.

## Installation 

The easiest way is to install this package is to run `pip install --upgrade relevanceai`.

## How to use the relevanceai client

For the relevanceai client, we want to ensure the SDK mirrors the API client.

For example:

```python
## To instantiate the client 
from relevanceai import Client
project = input("Your project goes here")
api_key = input("Your API key goes here")
client = Client(project, api_key)
```

To use the following endpoint: 

`/datasets/bulk_insert`

You can run: 

```python
# Bulk insert documents
client.datasets.bulk_insert(dataset_id, documents)
```

Or similarly, when you are trying to run 

`/services/search/vector`

You then write: 
```python
# Vector search in a dataset
client.services.search.vector(...)
```

## Features

### Out of the box multi-threading/multi-processing

Get multi-threading and multi-processing out of the box. The relevanceai Python package automatically gives you multi-threading and multi-processing out of the box!

```python

vdb_client.insert_documents(
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

vdb_client.insert_documents(
    dataset_id="match-example",
    docs=participant_frames,
    update_schema=True,
    overwrite=True,
    bulk_fn=bulk_fn
)
```

Under the hood, we use multiprocessing for processing the `bulk_fn` and multi-threading to send data via network requests. However, if there is no `bulk_fn` supplied, it only multi-threads network requests.

### Pull Update Push

Update documents within your collection based on a rule customised by you. The Pull-Update-Push Function loops through every document in your collection, brings it to your local computer where a function is applied (specified by you) and reuploaded to either an new collection or updated in the same collection. There is a logging functionality to keep track of which documents have been updated to save on network requests.

For example, consider a scenario where you have uploaded a dataset called 'test_dataset' containing integers up to 200. 

```python
original_collection = 'test_dataset'
data = [{'_id': str(i)} for i in range(200)]
client.datasets.bulk_insert(original_collection, data)
```

An example of sample data looks like this:

```
[{"_id": "0"}, {"_id": "1"}, ... {"_id": "199"}]
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
client.pull_update_push(original_collection, even_function)
```

An example of the data now: 
```
[{"_id": "0", "even": true}, {"_id": "1", "even": false}, ... {"_id": "199", "even": true}]
```

Alternatively, a new collection could be specified to direct where updated documents are uploaded into.

```python
new_collection = 'updated_test_dataset'
vec_client.pull_update_push(original_collection, even_function, new_collection)
```

An example of the data now: 
```
[{"_id": "0", "even": true}, {"_id": "1", "even": false}, ... {"_id": "199", "even": true}]
```

To delete all logs created by pull_update_push, use the `delete_all_logs` function.
```python
vec_client.delete_all_logs(original_collection)
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

# !pip install vectorhub[encoders-text-sentence-transformers]
from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec
model = SentenceTransformer2Vec()

# Define an update function
def encode_documents(docs):
    # Field and then the docs go here
    return model.encode_documents(["current", "Longer"], docs)

client.pull_update_push("sample-cn", encode_documents)

```

### Troubleshooting Pull Update Push

Sometimes Pull Update Push will fail for strange reasons (create a Github Issue!)

If this is the case, then you are free to use this: 

```python
from relevanceai import Client
url = "https://api-aueast.relevance.ai/v1/"

collection = ""
project = ""
api_key = ""
client = Client(project, api_key)
docs = client.datasets.documents.get_where(collection, select_fields=['title'])
while len(docs['documents']) > 0:
    docs['documents'] = model.encode_documents_in_bulk(['product_name'], docs['documents'])
    client.update_documents(collection, docs['documents'])
    docs = client.datasets.documents.get_where(collection, select_fields=['product_name'], cursor=docs['cursor'])
```

## Stop logging 

In order to stop all logging, you can just run this: 

```
client.logger.stop()
```

This can be helpful during client demos when you do not need to show the API endpoint being hit.

## Sample Datasets 

If you require a sample dataset, you can run the following to help:


```python
from relevanceai.datasets import get_ecommerce_dataset
docs = get_ecommerce_dataset()
```



## Development

### Getting Started

Setup your virtualenv, install requirements and package

```python
❯ python -m venv .venv
❯ source .venv/bin/activate
❯ pip install -r requirements-dev.txt  
```

Run your local tests in [`tests`](./tests)

```zsh
❯ pytest --cov=src -vv
❯ pytest <file_path> --cov=src -vv
```


```
Copyright (C) Relevance AI - All Rights Reserved
Unauthorized copying of this repository, via any medium is strictly prohibited
Proprietary and confidential
Relevance AI <dev@relevance.ai> 2021 
```
